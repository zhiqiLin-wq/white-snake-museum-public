"""FastAPI 路由 (v6 统一架构) — /chat, /skills, /health, /rebuild-index。

核心变更 (v4 → v6):
- _stream_chat(): 统一委托给 SupervisorGraph.astream_events() (U05-01)
- _execute_chat(): 统一委托给 SupervisorGraph.run() (U05-03)
- _direct_route_stream() 及所有 _stream_*_direct() 已删除 (U05-02)
- _format_event(): 简化为透传 (U04-04)
- 只有 routes.py 有权发出 done 事件 (U04-05)
"""
import asyncio
import json
import time
import logging
import uuid as _uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from .models import (
    ChatRequest,
    ChatResponse,
    SkillListResponse,
    HealthResponse,
    RebuildResponse,
    ChatResumeRequest,
    ResumePayload,
    CancelRequest,
)
from .events import SSEEventType, EVOLUTION_EVENT_PREFIX, WORKSPACE_EVENTS, _camelize_value
from ..config import settings
from ..security.auth import get_current_user
from ..llm.provider import LLMFatalError, RateLimitError, APITimeoutError
from ..observability.tracer import Tracer

logger = logging.getLogger(__name__)

router = APIRouter()

# ============================================================
#  全局引用 (由 main.py lifespan 注入)
# ============================================================
llm_provider = None
rag_pipeline = None
vector_store = None
embedder = None
skill_registry = None
prompt_registry = None
graph_registry = None
mcp_registry = None
metrics = None
node_client = None
supervisor = None  # v6: SupervisorGraph 单例
memory_system_factory = None  # per-user MemorySystem 工厂（由 main.py 注入）
start_time: float = time.monotonic()


# ============================================================
#  POST /chat
# ============================================================
@router.post("/chat")
async def chat(request: ChatRequest, user_id: str = Depends(get_current_user)):
    """对话入口 — 支持流式 (SSE) 和非流式 (JSON)。"""
    query = request.query.strip()
    if len(query) > settings.max_input_length:
        query = query[:settings.max_input_length] + "..."

    # Inject context tags into query so AgentLoop sees chapter/paragraph/annotation context
    # v13: 压缩注入文本，从 ~200 字减至 ~30 字。标注工具使用指南已在 AgentLoop system prompt 中
    if request.context_tags:
        parts = [query]
        for tag in request.context_tags:
            tag_data = tag.get("data", {})
            ch_num = tag_data.get("chapterNumber")
            ch_title = tag_data.get("chapterTitle", "")
            para_idx = tag_data.get("paragraphIndex")
            text = tag_data.get("text", "")
            if ch_num is not None:
                title_hint = f"({ch_title})" if ch_title else ""
                parts.append(f"[ctx: chapter_number={ch_num}{title_hint}]")
                if para_idx is not None:
                    parts.append(f"[ctx: p={para_idx}]")
            elif text:
                parts.append(f"[ctx: sel=\"{str(text)[:80]}\"]")
        query = " ".join(parts)

    if request.stream:
        return StreamingResponse(
            _stream_chat(query, request.skill, request.conversation_id,
                         request.new_session, request.history, request.paragraphs, user_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        result = await _execute_chat(query, request.skill, request.conversation_id, request.new_session, request.history, request.paragraphs, user_id)
        return result


# ============================================================
#  _execute_chat (非流式) — U05-03
# ============================================================
def _ctx_thread_id(user_id: str, conversation_id: Optional[str], fallback: str) -> str:
    """生成 context_manager 的隔离 key。

    有真实 conversation_id（UUID）时直接用，避免破坏会话连续性；
    conversation_id 为空（fallback）时，用 user_id 前缀隔离不同用户，
    防止多个匿名用户共享 "default"/"stream" 导致上下文串号 (P3-3)。
    """
    if conversation_id:
        return conversation_id
    if user_id:
        return f"{user_id}::{fallback}"
    return fallback


async def _get_context_manager(user_id: str):
    """按 user_id 获取 per-user ContextManager（记忆隔离）。

    user_id 为空或工厂未注入时返回 None，调用方回退到前端传入的 history。
    """
    if memory_system_factory is None or not user_id:
        return None
    ms = await memory_system_factory(user_id)
    return ms.context_manager if ms is not None else None


async def _execute_chat(
    query: str,
    skill: Optional[str],
    conversation_id: Optional[str],
    new_session: bool,
    history: Optional[list] = None,
    paragraphs: Optional[list] = None,
    user_id: str = "",
) -> ChatResponse:
    """非流式对话 — v15: ContextManager + 被动注入。"""
    thread_id = _ctx_thread_id(user_id, conversation_id, "default")

    if supervisor is None:
        return ChatResponse(answer="AI 服务尚未就绪，请稍后重试", workflow_used="none", degraded=True)

    # v15: ContextManager 管理跨请求上下文 + 被动注入 + Core Memory（per-user 隔离）
    memory_context = ""
    cm = await _get_context_manager(user_id)
    if cm is not None:
        await cm.load(thread_id)
        # 先入队当前 query（触发压缩 + 同步更新摘要/提取记忆），
        # 使本轮压缩的产物（新摘要、新长期记忆）本轮即可见，不再慢一拍 (P-fix)
        await cm.enqueue(thread_id, {"role": "user", "content": query})
        memory_context = cm.build_memory_context_text(thread_id, query)
        effective_history = cm.build_history_messages(thread_id)
    else:
        effective_history = history

    from ..langgraph.memory import get_user_memory
    user_memory = await get_user_memory(user_id) if user_id else None

    try:
        result = await supervisor.run(
            query, conversation_id=conversation_id,
            history=effective_history, user_id=user_id,
            user_memory=user_memory, system_context=memory_context,
        )

        # v14: 将 assistant 回复回写 FIFO
        answer_text = result.get("answer", "")
        if cm is not None and answer_text.strip():
            await cm.enqueue(
                thread_id, {"role": "assistant", "content": answer_text},
            )
            await cm.save(thread_id)

        return ChatResponse(
            answer=answer_text,
            sources=result.get("sources", []),
            workflow_used=result.get("workflow_used", "unknown"),
            conversation_id=conversation_id,
            degraded=result.get("degraded", False),
        )
    except (LLMFatalError, RateLimitError, APITimeoutError) as e:
        logger.error(f"LLM 服务不可用: {e}")
        return ChatResponse(answer=f"AI 服务暂时不可用：{str(e)}", workflow_used="none", degraded=True)
    except json.JSONDecodeError as e:
        logger.error(f"LLM 返回格式异常: {e}")
        return ChatResponse(answer="AI 返回了无法解析的响应，请重试或缩短问题。", workflow_used="none", degraded=True)


# ============================================================
#  _stream_chat (流式 SSE) — U05-01
# ============================================================
async def _sse_heartbeat(source, ping_interval: float = 15.0):
    """SSE 心跳包装 — 源流长时间无事件时发送 ping 事件。

    前端 SSEEventBus 收到任何事件（含 ping）都会刷新 lastEventTime；
    10 分钟完全无事件（真断流）才会触发前端超时 abort。
    防止慢 LLM 调用/长工具执行期间被代理或前端判定为连接死亡。
    """
    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    async def _pump():
        try:
            async for ev in source:
                await queue.put(ev)
        except Exception as e:
            await queue.put(e)  # 异常传给消费者，保持外层 except 行为
        finally:
            await queue.put(_SENTINEL)

    pump_task = asyncio.create_task(_pump())
    try:
        while True:
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=ping_interval)
            except asyncio.TimeoutError:
                # 必须与上游事件保持一致的 dict 格式（{"type","data"}），
                # 若直接 yield _sse() 格式化字符串，_stream_chat 循环对其调用 .get() 会抛
                # 'str' object has no attribute 'get'
                yield {"type": "ping", "data": {"ts": int(time.time() * 1000)}}
                continue
            if ev is _SENTINEL:
                break
            if isinstance(ev, Exception):
                raise ev
            yield ev
    finally:
        pump_task.cancel()


async def _stream_chat(
    query: str,
    skill: Optional[str],
    conversation_id: Optional[str],
    new_session: bool,
    history: Optional[list] = None,
    paragraphs: Optional[list] = None,
    user_id: str = "",
):
    """SSE 流式对话 — v15: ContextManager + 被动注入。"""
    if new_session:
        thread_id = str(_uuid.uuid4())
    else:
        thread_id = _ctx_thread_id(user_id, conversation_id, "stream")

    done_emitted = False

    # v15: ContextManager 管理跨请求上下文 + 被动注入 + Core Memory（per-user 隔离）
    memory_context = ""
    cm = await _get_context_manager(user_id)
    if cm is not None:
        await cm.load(thread_id)
        # 先入队当前 query（触发压缩 + 同步更新摘要/提取记忆），
        # 使本轮压缩的产物（新摘要、新长期记忆）本轮即可见，不再慢一拍 (P-fix)
        await cm.enqueue(thread_id, {"role": "user", "content": query})
        memory_context = cm.build_memory_context_text(thread_id, query)
        effective_history = cm.build_history_messages(thread_id)
    else:
        effective_history = history

    # 同步 conversation 元数据到 user.db（非阻塞，失败不影响对话）
    if node_client and thread_id != "stream" and user_id:
        try:
            await node_client.upsert_conversation(thread_id, title="", user_id=user_id)
        except Exception:
            pass  # degraded: conversation 元数据同步失败不阻断对话

    try:
        # ── 就绪检查 ──
        if rag_pipeline is None or not rag_pipeline.embedder.is_ready:
            load_err = rag_pipeline.embedder.load_error if rag_pipeline is not None else None
            if load_err:
                # 加载已永久失败：如实告知原因，而不是让用户无限等"初始化"
                yield _sse("thinking_start", {"message": "AI 服务模型加载失败"})
                yield _sse("error", {"code": "MODEL_LOAD_FAILED",
                                     "message": f"Embedding 模型加载失败，请查看服务端日志: {load_err[:300]}"})
            else:
                yield _sse("thinking_start", {"message": "正在初始化 AI 服务..."})
                yield _sse("status", {"message": "等待 Embedding 模型就绪..."})
                yield _sse("error", {"code": "NOT_READY", "message": "AI 服务正在初始化，请稍后再试"})
            yield _sse("done", {"workflow": "none"})
            return

        if supervisor is None:
            yield _sse("thinking_start", {"message": "正在初始化 Agent 服务..."})
            yield _sse("error", {"code": "NOT_READY", "message": "Agent 服务正在初始化，请稍后再试"})
            yield _sse("done", {"workflow": "none"})
            return

        # LLM trace: 为本次 chat 会话启动 trace（演化子图会自行重开自己的 trace）
        chat_trace = Tracer.start_trace(thread_id, query)

        # ── 技能匹配 ──
        resolved_skill = skill
        if not resolved_skill and skill_registry:
            matched = skill_registry.match(query)
            if matched:
                resolved_skill = matched[0].get("name")
                yield _sse("skill_activated", {
                    "name": resolved_skill,
                    "display_name": matched[0].get("display_name", ""),
                    "icon": matched[0].get("icon", "📖"),
                })

        # ── 统一流式入口: SupervisorGraph ──
        import asyncio as _asyncio
        from ..langgraph.memory import get_user_memory
        user_memory = await get_user_memory(user_id) if user_id else None

        yield _sse("thinking_start", {"message": "Agent 开始思考..."})

        collected_answer_parts = []

        async for event in _sse_heartbeat(supervisor.astream_events(query, conversation_id=conversation_id, history=effective_history, paragraphs=paragraphs, user_id=user_id, user_memory=user_memory, system_context=memory_context)):
            # 防御：上游事件必须是 dict（{"type","data"}），否则跳过并记录，避免整流崩溃
            if not isinstance(event, dict):
                logger.warning(f"[SSE] 跳过非 dict 事件: {type(event).__name__}: {str(event)[:200]}")
                continue
            ev_type = event.get("type", "")
            ev_data = event.get("data", "")

            # 对于没有 "data" 包装的事件（如 status, motif_matrix 等 build_sse_event 产出），
            # 整个 event dict 就是数据体，去掉 type/timestamp 后作为 ev_data
            if not ev_data and isinstance(event, dict):
                ev_data = {k: v for k, v in event.items() if k not in ("type", "timestamp")}

            # B-072: 子图的 done 事件已在 supervisor 层过滤，这里透传 supervisor 自身的 done
            if ev_type == SSEEventType.DONE:
                done_emitted = True
                yield _sse("done", ev_data)
                continue

            # B-074: SSE 事件 DEBUG 日志
            if ev_type == SSEEventType.CHUNK:
                data_len = len(str(ev_data)) if ev_data else 0
                logger.debug(f"[SSE] event=chunk size={data_len}")
            elif ev_type == SSEEventType.DONE:
                logger.debug(f"[SSE] event=done total_data={len(str(ev_data)) if ev_data else 0}")

            # B-068: 工作区事件 + 演化事件直接透传，不包装
            # interrupt 事件 (evolution_interrupt) 保留了完整外层结构
            # (interrupt_id/interruptId, thread_id/threadId, data)，必须整包发送，前端需要这些字段
            # build_sse_event 会将所有 key 转为 camelCase，所以这里同时检查两种命名
            if ev_type in WORKSPACE_EVENTS or ev_type.startswith(EVOLUTION_EVENT_PREFIX):
                if "interrupt_id" in event or "interruptId" in event:
                    ev_data = {k: v for k, v in event.items() if k != "type"}
                yield _sse(ev_type, ev_data)
            elif ev_type == SSEEventType.CHUNK:
                text = ev_data if isinstance(ev_data, str) else str(ev_data)
                collected_answer_parts.append(text)
                yield _sse("chunk", {"text": text})
            else:
                yield _sse(ev_type, ev_data)

        # v14: 将 assistant 回复回写 FIFO
        if cm is not None and collected_answer_parts:
            full_answer = "".join(collected_answer_parts)
            if full_answer.strip():
                await cm.enqueue(
                    thread_id, {"role": "assistant", "content": full_answer},
                )
                await cm.save(thread_id)

        # 确保 done 只发出一次
        if not done_emitted:
            yield _sse("done", {"workflow": "supervisor", "thread_id": thread_id})

    except Exception as e:
        logger.error(f"SSE 流式对话错误: {e}")
        yield _sse("error", {"code": "INTERNAL", "message": str(e)})
        yield _sse("done", {"workflow": "none", "status": "error"})
    finally:
        # 打印本次 chat 的 LLM trace 汇总（与演化分析的 trace summary 格式一致）
        try:
            summary = chat_trace.llm_summary()
            logger.info(f"[LLM_TRACE] chat trace summary: {json.dumps(summary, ensure_ascii=False)}")
        except Exception:
            pass


# ============================================================
#  DELETE /memory/conversation/{thread_id} — 删除对话记忆/召回
# ============================================================
@router.delete("/memory/conversation/{thread_id}")
async def delete_conversation_memory(thread_id: str, user_id: str = Depends(get_current_user)):
    """删除某对话产生的全部记忆/召回数据（跨存储联动）。

    由 Node 删除对话时回调（x-user-id + localhost），或前端直接调用。
    覆盖：长期记忆 + 召回消息 + FIFO 上下文 + ChromaDB 向量 + 提取状态。
    """
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    from ..context.memory_registry import delete_user_conversation_memory
    result = await delete_user_conversation_memory(user_id, thread_id)
    status = result.pop("status", "ok")
    if status == "degraded":
        return {"status": "degraded", **result}
    return {"status": "ok", **result}


# ============================================================
#  v19: 报告文件库 REST API — 用户专属报告存储区
#  存储位置: agent/reports/{user_id}/{report_id}.md + .meta.json
#  由 save_long_report 工具写入，这里暴露列表/读取/删除端点
# ============================================================
@router.get("/reports")
async def list_reports(user_id: str = Depends(get_current_user)):
    """列出当前用户的全部研究报告（按创建时间倒序）。"""
    from ..mcp.tools.save_long_report import handler as slr_handler
    result = await slr_handler(action="list", user_id=user_id)
    return result


@router.get("/reports/{report_id}")
async def get_report(report_id: str, user_id: str = Depends(get_current_user)):
    """读取某报告的完整 Markdown 内容（平台内预览）。"""
    from ..mcp.tools.save_long_report import handler as slr_handler
    result = await slr_handler(action="get", report_id=report_id, user_id=user_id)
    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])
    # 补充统计字段（meta 中只有 sections 数组，前端预览头需要直接可读的计数）
    result["sectionCount"] = len(result.get("sections", []))
    return result


@router.delete("/reports/{report_id}")
async def delete_report(report_id: str, user_id: str = Depends(get_current_user)):
    """删除报告的 .md 文件与 .meta.json（仅限本人报告）。"""
    from ..mcp.tools.save_long_report import _safe_id, _md_path, _meta_path
    uid = _safe_id(user_id)
    deleted = False
    for p in (_md_path(uid, report_id), _meta_path(uid, report_id)):
        if p.exists():
            p.unlink()
            deleted = True
    if not deleted:
        raise HTTPException(status_code=404, detail=f"报告不存在: {report_id}")
    return {"status": "ok", "reportId": report_id}


# ============================================================
#  SSE 工具函数
# ============================================================
def _sse(event: str, data) -> str:
    """构造 SSE 事件字符串。dict 类型 data 的键名自动转换为 camelCase。"""
    if isinstance(data, dict):
        data = _camelize_value(data)
    event_str = getattr(event, 'value', event)
    return f"event: {event_str}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# _format_event 已删除 (U04-04) — 事件格式化简化为直接透传
# _direct_route_stream 已删除 (U05-02)
# _stream_location_story, _stream_comparison, _stream_itinerary_direct, _stream_evolution_direct 已删除 (U05-02)


# ============================================================
#  _extract_state — 从多种 checkpointer 返回格式中提取状态 dict
# ============================================================
def _extract_state(state_raw) -> dict:
    """从 checkpointer 返回的各种格式中提取实际状态 dict。

    LangGraph 1.x: aget() 返回 dict，状态在 ['channel_values'] 中
    LangGraph 0.x: aget() 返回 CheckpointTuple，状态在 .checkpoint.channel_values 中
    MemorySaver: aget() 返回 dict，状态在 ['channel_values'] 中
    """
    if state_raw is None:
        return {}
    # LangGraph 0.x: CheckpointTuple 有 .checkpoint 属性
    if hasattr(state_raw, 'checkpoint'):
        cp = state_raw.checkpoint
        if hasattr(cp, 'channel_values'):
            return cp.channel_values if isinstance(cp.channel_values, dict) else {}
        if isinstance(cp, dict):
            return cp
        return {}
    # LangGraph 1.x / MemorySaver: dict 包含 'channel_values' key
    if isinstance(state_raw, dict):
        if 'channel_values' in state_raw and isinstance(state_raw['channel_values'], dict):
            return state_raw['channel_values']
        # StateSnapshot (某些版本): dict 包含 'values' key
        if 'values' in state_raw and isinstance(state_raw['values'], dict):
            return state_raw['values']
        return state_raw
    return {}


# ============================================================
#  B-091: POST /chat/resume — 中断恢复端点
# ============================================================
@router.post("/chat/resume")
async def resume_chat(request: ChatResumeRequest, user_id: str = Depends(get_current_user)):
    """恢复被中断的 LangGraph 图执行。"""
    logger.info(f"[RESUME] RECEIVED thread={request.thread_id} interrupt={request.interrupt_id} action={request.resume.action if request.resume else '?'}")
    thread_id = request.thread_id
    interrupt_id = request.interrupt_id

    # B-094: 验证 interrupt_id
    VALID_INTERRUPTS = ("confirm_motifs", "select_hypothesis", "decide_next_action")
    if interrupt_id not in VALID_INTERRUPTS:
        raise HTTPException(400, f"Unknown interrupt_id: {interrupt_id}")

    # 获取 checkpointer — 使用每用户独立的 agent_checkpoints.db
    from ..langgraph.memory import get_user_memory
    user_memory = await get_user_memory(user_id) if user_id else None
    checkpointer = user_memory.get_checkpointer() if user_memory and user_memory.is_ready else None

    if checkpointer is None:
        raise HTTPException(500, "Checkpointer 不可用，无法恢复会话")

    config = {"configurable": {"thread_id": thread_id}}

    # B-096: 检查 resume 超时
    timeout_error = await _check_resume_timeout(checkpointer, config)
    if timeout_error:
        status_code = 404 if "not found" in timeout_error else 410
        raise HTTPException(status_code, timeout_error)

    # B-095: 幂等性检查 — 验证图当前状态与中断点匹配
    try:
        state_raw = await checkpointer.aget(config) if hasattr(checkpointer, 'aget') else await checkpointer.get(config)
        if state_raw is None:
            raise HTTPException(404, f"thread_id '{thread_id}' not found")
        state = _extract_state(state_raw)
        current_phase = state.get("current_phase", "")
        logger.info(
            f"[resume] state_raw type={type(state_raw).__name__} "
            f"extracted state keys={list(state.keys())[:20]} "
            f"current_phase={current_phase!r}"
        )

        # 验证中断点匹配
        _INTERRUPT_PHASE_MAP = {
            "confirm_motifs": ("E1_complete",),
            "select_hypothesis": ("E6_complete", "E6_selected", "E7_decided"),
            "decide_next_action": ("E7_complete",),
        }
        expected_phases = _INTERRUPT_PHASE_MAP.get(interrupt_id, ())
        if expected_phases and current_phase not in expected_phases:
            raise HTTPException(
                409,
                f"会话状态不匹配: interrupt_id={interrupt_id} 期望 phase={expected_phases}, "
                f"实际 phase={current_phase}"
            )

        # 幂等性检查: 已完成的分析不可 resume
        if current_phase in ("E8_synthesized", "complete"):
            raise HTTPException(409, "会话已完成，无法恢复")
    except HTTPException:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as e:
        raise HTTPException(500, f"无法检查图状态: {e}")

    # B-092: 构建 resume command
    resume_data = _build_resume_command(request.resume, interrupt_id)

    return StreamingResponse(
        _stream_resume(config, resume_data, thread_id, interrupt_id, user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
#  B-092: _build_resume_command
# ============================================================
def _build_resume_command(resume: ResumePayload, interrupt_id: str) -> dict:
    """将前端的 resume 数据转为 resume dict（供 Command(resume=...) 使用）。"""
    if interrupt_id == "confirm_motifs":
        return {
            "action": "confirm",
            "edited_motifs": resume.edited_motifs,
        }
    elif interrupt_id == "select_hypothesis":
        return {
            "action": "select",
            "selected_hypothesis_ids": resume.selected_hypothesis_ids,
        }
    elif interrupt_id == "decide_next_action":
        # 兼容前端两种字段名: decision(后端规范) / selected_option_id(前端当前发送)
        decision_value = resume.decision or resume.selected_option_id
        if not decision_value:
            raise HTTPException(400, "decision is required for decide_next_action")
        return {
            "action": "decide",
            "decision": decision_value,
        }
    return {"action": resume.action}


# ============================================================
#  B-093: _stream_resume
# ============================================================
async def _stream_resume(config: dict, resume_data: dict, thread_id: str, interrupt_id: str, user_id: str = ""):
    """恢复流的生成器。

    B-071: 先 emit interrupt_resumed 事件。
    """
    from ..langgraph.evolution_workbench import EvolutionWorkbench

    # B-071: 恢复确认事件
    yield _sse("interrupt_resumed", {
        "interrupt_id": interrupt_id,
        "user_action": resume_data.get("action", "unknown"),
        "message": "已收到你的选择，继续分析...",
    })

    try:
        # Fast fail: checkpointer must be available for resume to work
        from ..langgraph.memory import get_user_memory as _get_user_mem
        _user_mem = await _get_user_mem(user_id) if user_id else None
        checkpointer = _user_mem.get_checkpointer() if _user_mem and _user_mem.is_ready else None
        if checkpointer is None:
            logger.error(
                f"[resume] checkpointer is None: _user_mem={_user_mem is not None} "
                f"mem_ready={_user_mem.is_ready if _user_mem else False}"
            )
            yield _sse("error", {
                "code": "NO_CHECKPOINTER",
                "message": "会话持久化服务未就绪，无法恢复演化分析。请重新发起演化追踪。",
            })
            return

        # Fast fail: verify the checkpoint exists before attempting resume
        try:
            existing = await checkpointer.aget(config)
            if existing is None:
                yield _sse("error", {
                    "code": "STATE_NOT_FOUND",
                    "message": f"未找到会话状态 (thread={thread_id})，可能已过期。请重新发起演化追踪。",
                })
                return
            existing_state = _extract_state(existing)
            logger.info(
                f"[resume] checkpoint found: thread={thread_id} "
                f"phase={existing_state.get('current_phase','?')}"
            )
        except Exception as ck_err:
            logger.error(f"[resume] checkpoint lookup failed: {ck_err}")
            yield _sse("error", {
                "code": "CHECKPOINT_ERROR",
                "message": f"无法读取会话状态: {ck_err}",
            })
            return

        # Fast fail: supervisor and dependencies
        if supervisor is None:
            yield _sse("error", {"code": "NO_SUPERVISOR", "message": "Agent 总控未初始化，请重启服务"})
            return
        if rag_pipeline is None or rag_pipeline.retriever is None:
            yield _sse("error", {"code": "NO_RETRIEVER", "message": "检索服务未初始化，无法恢复分析"})
            return
        if llm_provider is None:
            yield _sse("error", {"code": "NO_LLM", "message": "LLM 服务未初始化，无法恢复分析"})
            return

        wb = supervisor._subgraphs.get("evolution")
        if wb is None:
            from ..langgraph.evolution_workbench import EvolutionWorkbench
            wb = EvolutionWorkbench(
                llm=llm_provider,
                retriever=rag_pipeline.retriever,
                prompt_registry=prompt_registry,
                mcp_registry=mcp_registry,
                checkpointer=checkpointer,
                node_client=node_client,
            )
            supervisor._subgraphs["evolution"] = wb

        # 注册到活跃列表以支持用户取消
        supervisor._active_workbenches[thread_id] = wb

        try:
            logger.info(
                f"[resume] starting resume_stream: thread={thread_id} "
                f"interrupt={interrupt_id} action={resume_data.get('action','?')}"
            )
            event_count = 0
            async for event in wb.resume_stream(resume_data, config, thread_id):
                event_count += 1
                ev_type = event.get("type", "")
                logger.debug(f"[resume] event #{event_count}: type={ev_type}")
                if ev_type == SSEEventType.DONE:
                    continue
                if "interrupt_id" in event or "interruptId" in event:
                    ev_data = {k: v for k, v in event.items() if k != "type"}
                else:
                    ev_data = event.get("data", event)
                yield _sse(ev_type, ev_data)
            logger.info(f"[resume] resume_stream completed: {event_count} events yielded")
        finally:
            supervisor._active_workbenches.pop(thread_id, None)

    except Exception as e:
        logger.error(f"[resume] 流式执行错误: {e}", exc_info=True)
        yield _sse("error", {"code": "RESUME_ERROR", "message": str(e)[:300]})

    yield _sse("done", {"workflow": "evolution_workbench", "thread_id": thread_id})


# ============================================================
#  B-096: _check_resume_timeout
# ============================================================
async def _check_resume_timeout(checkpointer, config: dict) -> Optional[str]:
    """检查 resume 是否超时。

    Returns:
        None 表示未超时，str 表示错误信息
    """
    try:
        state = None
        if hasattr(checkpointer, 'aget'):
            state = await checkpointer.aget(config)
        elif hasattr(checkpointer, 'get'):
            state = await checkpointer.get(config)

        if state is None:
            return "thread not found"

        from datetime import datetime, timezone
        RESUME_TIMEOUT = 600  # 10 分钟

        state_dict = _extract_state(state)

        metadata = state_dict.get("metadata", {})
        interrupt_at = metadata.get("interrupt_created_at")

        if interrupt_at:
            created = datetime.fromisoformat(str(interrupt_at).replace("Z", "+00:00"))
            elapsed = (datetime.now(timezone.utc) - created).total_seconds()
            if elapsed > RESUME_TIMEOUT:
                return f"session timed out after {int(elapsed)}s (max {RESUME_TIMEOUT}s)"

        return None
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        # 已知可恢复异常：state 结构不兼容、字段缺失等
        logger.warning(f"_check_resume_timeout 检查失败（不阻塞）: {e}")
        return None


# ============================================================
#  GET /chat/evolution-state — 按 threadId 读取演化分析历史状态
#  LangGraph checkpointer（SQLite agent_checkpoints.db）中存有完整 state，
#  前端 localStorage 快照丢失（清缓存/过期/误删）时从此恢复
# ============================================================
@router.get("/chat/evolution-state")
async def get_evolution_state(thread_id: str, user_id: str = Depends(get_current_user)):
    if not thread_id or not thread_id.strip():
        raise HTTPException(status_code=400, detail="thread_id is required")

    from ..langgraph.memory import get_user_memory
    user_memory = await get_user_memory(user_id) if user_id else None
    checkpointer = user_memory.get_checkpointer() if user_memory and user_memory.is_ready else None
    if checkpointer is None:
        raise HTTPException(status_code=500, detail="Checkpointer 不可用")

    config = {"configurable": {"thread_id": thread_id}}
    try:
        state_raw = await checkpointer.aget(config) if hasattr(checkpointer, "aget") else await checkpointer.get(config)
    except Exception as e:
        logger.error(f"[evolution-state] 读取 checkpoint 失败 thread={thread_id}: {e}")
        raise HTTPException(status_code=500, detail=f"读取 checkpoint 失败: {e}")

    if state_raw is None:
        raise HTTPException(status_code=404, detail=f"thread_id '{thread_id}' 不存在")

    state = _extract_state(state_raw)
    if not state:
        raise HTTPException(status_code=404, detail=f"thread_id '{thread_id}' 状态为空")

    # ---- 组装前端 motifMatrix {rows, columns, cells}，与 E2 SSE 推送格式一致 ----
    from ..langgraph.evolution_workbench import _cn_to_arabic, _fmt_cell_value
    from ..api.events import _camelize_value

    motifs = state.get("motifs", []) or []
    dynasties = state.get("dynasties", []) or []
    matrix = state.get("motif_matrix", []) or []
    motif_names = [f"{m.get('id', '')}: {m.get('name', '')}" for m in motifs]
    cells = []
    for cell in matrix:
        occ = cell.get("occurrence_count", 0)
        docs = cell.get("documents", []) or []
        summary = cell.get("summary", "")
        cells.append({
            "motifId": cell.get("motif_id", ""),
            "era": cell.get("dynasty", ""),
            "status": "absent" if occ == 0 else "stable",
            "value": _fmt_cell_value(occ, summary),
            "tooltip": summary or None,
            "occurrenceCount": occ,
            "evidence": [
                {
                    "chapterNumber": _cn_to_arabic(
                        d.get("metadata", {}).get("chapter_number", ""),
                        d.get("metadata", {}).get("source_type", "primary_literature"),
                    ),
                    "chapterTitle": d.get("metadata", {}).get("chapter_title", ""),
                    "paragraphIndex": d.get("metadata", {}).get("paragraph_index", 0),
                    "excerpt": (d.get("content", "") or "")[:2000],
                    "dynasty": d.get("metadata", {}).get("dynasty", ""),
                }
                for d in docs
            ],
        })

    # 中断态 phase 统一映射为完成态（历史卡片只做展示，resume 走 /chat/resume）
    current_phase = state.get("current_phase", "") or ""
    phase_map = {"E8_synthesized": "E8_synthesize"}

    # claims 走 camelize：state 中是 snake_case (evidence_pro/claim_id)，
    # 前端模板期望 camelCase (evidencePro/claimId)，与 SSE 推送一致
    raw_claims = state.get("claims", []) or []
    camelized_claims = _camelize_value(raw_claims)

    payload = {
        "threadId": state.get("thread_id") or thread_id,
        "trackingTarget": state.get("tracking_target", ""),
        "currentPhase": phase_map.get(current_phase, current_phase),
        "motifMatrix": {"rows": dynasties, "columns": motif_names, "cells": cells},
        "phylogeny": _camelize_value(state.get("phylogeny", {}) or {"nodes": [], "edges": []}),
        "claimVerdicts": camelized_claims,
        "narrative": state.get("evolution_narrative", ""),
        "evidenceRefs": _camelize_value(state.get("evidence_refs", {}) or {}),
    }
    return payload


# ============================================================
#  POST /chat/cancel
# ============================================================
@router.post("/chat/cancel")
async def cancel_chat(request: CancelRequest):
    """取消正在进行的演化分析。"""
    if supervisor is None:
        raise HTTPException(status_code=503, detail="Agent 服务未初始化")
    ok = supervisor.cancel_evolution(request.thread_id)
    return {"status": "cancelled" if ok else "not_found", "thread_id": request.thread_id}


# ============================================================
#  GET /skills
# ============================================================
@router.get("/skills", response_model=SkillListResponse)
async def list_skills():
    if skill_registry is None:
        return SkillListResponse(skills=[])
    return SkillListResponse(skills=skill_registry.list_all())


# ============================================================
#  GET /health (U12: 增加 build_progress)
# ============================================================
# U12: 索引构建进度（由 chunker progress_callback 更新）
_index_build_progress: dict = {
    "current": 0, "total": 9, "phase": "idle", "current_title": "",
}


def _make_chunking_progress_callback():
    """创建 chunking 进度回调，更新 _index_build_progress。"""
    def callback(current: int, total: int, title: str):
        _index_build_progress["current"] = current
        _index_build_progress["total"] = total
        _index_build_progress["current_title"] = title
        _index_build_progress["phase"] = "chunking" if current < total else "indexing"
    return callback


@router.get("/health", response_model=HealthResponse)
async def health():
    llm_status = "configured" if settings.is_llm_configured else "missing_key"

    node_status = "unknown"
    if node_client:
        try:
            ok = await node_client.health_check()
            node_status = "connected" if ok else "unreachable"
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            logger.warning(f"Node API health check 失败: {e}")
            node_status = "unreachable"

    vs_status = "unknown"
    if vector_store:
        try:
            vs_status = "ready" if vector_store.is_healthy() else "error"
            if _index_build_progress["phase"] in ("chunking", "indexing"):
                vs_status = "building"
        except (ValueError, RuntimeError, ConnectionError) as e:
            logger.warning(f"Vector store health check 失败: {e}")
            vs_status = "error"
    elif settings.is_llm_configured:
        vs_status = "building"

    build_progress = dict(_index_build_progress)
    if build_progress["total"] > 0:
        build_progress["percent"] = round(build_progress["current"] / build_progress["total"], 2)
    else:
        build_progress["percent"] = 0.0

    resp = HealthResponse(
        status="ok" if llm_status == "configured" else "degraded",
        vector_store=vs_status,
        llm=llm_status,
        node_api=node_status,
        build_progress=build_progress,
        uptime_seconds=time.monotonic() - start_time,
    )

    # B-005: Checkpointer 健康信息（每用户模式，报告类型即可）
    memory_type = graph_registry.get("conversation_memory_type", "unknown") if graph_registry else "unknown"
    resp.checkpointer = {
        "checkpointer_type": memory_type,
        "checkpointer_ready": memory_type == "per_user_sqlite",
        "active_sessions": 0,
        "wal_mode": "unknown",
    }

    # B-126: 数据层健康信息
    data_health = {
        "retriever_ready": False,
        "chroma_collection_count": 0,
        "node_api_ready": node_status == "connected",
    }
    if vector_store:
        try:
            data_health["retriever_ready"] = vector_store.is_healthy()
            data_health["chroma_collection_count"] = vector_store.collection_count() if hasattr(vector_store, 'collection_count') else -1
        except (AttributeError, ValueError, RuntimeError) as e:
            logger.warning(f"Data layer health check 失败: {e}")
    resp.data_layer = data_health

    # B-007: 活跃 thread 数（每用户模式依赖各 ConversationMemory 实例统计）
    resp.active_threads = 0

    if embedder:
        resp.embedding_model = embedder.status_info
    if metrics:
        resp.metrics = metrics.snapshot
    return resp


# ============================================================
#  GET /trace/llm — 查看最近一次演化分析的 LLM trace
# ============================================================
@router.get("/trace/llm")
async def get_llm_trace(full: bool = False):
    """返回最近一次演化分析的 LLM 调用 trace。

    Args:
        full: true 时返回全量 prompt（system/user/response 完整原文）。
              false 时只返回汇总摘要 + 每次调用的元信息（不含 prompt 正文）。
    """
    from ..observability.tracer import Tracer
    trace = Tracer.get_current()
    if trace is None:
        return {"status": "no_active_trace", "message": "当前没有活跃的 trace"}
    summary = trace.llm_summary()
    if full:
        return {
            "summary": summary,
            "calls": trace.llm_full_export(),
        }
    # 非 full 模式：只返回元信息，不含 prompt 正文
    calls_meta = []
    for c in trace.llm_calls:
        calls_meta.append({
            "call_id": c.call_id,
            "timestamp": c.timestamp,
            "node": c.node,
            "purpose": c.purpose,
            "upstream": c.upstream,
            "downstream": c.downstream,
            "model": c.model,
            "system_prompt_len": len(c.system_prompt),
            "user_prompt_len": len(c.user_prompt),
            "response_len": len(c.response),
            "tokens_total": c.tokens_total,
            "latency_ms": round(c.latency_ms, 1),
            "cached": c.cached,
            "degraded": c.degraded,
            "error": c.error,
        })
    return {"summary": summary, "calls": calls_meta}


# ============================================================
#  POST /reload-templates
# ============================================================
@router.post("/reload-templates")
async def reload_templates():
    result = {"templates": 0, "skills": 0}
    if prompt_registry:
        prompt_registry.reload()
        result["templates"] = len(prompt_registry.templates)
    if skill_registry:
        from pathlib import Path
        from ..skills.loader import SkillLoader
        skill_dir = Path(__file__).parent.parent / "skills" / "definitions"
        loader = SkillLoader(skill_dir)
        for skill_def in loader.load_all():
            skill_registry.register(skill_def)
        result["skills"] = len(skill_registry.list_all())
    return {"status": "ok", **result}


# ============================================================
#  POST /rebuild-index (U11: 异步化 + 防并发)
# ============================================================
_rebuild_lock: asyncio.Lock | None = None
_rebuild_in_progress: bool = False


def _get_rebuild_lock() -> asyncio.Lock:
    global _rebuild_lock
    if _rebuild_lock is None:
        _rebuild_lock = asyncio.Lock()
    return _rebuild_lock


@router.post("/rebuild-index")
async def rebuild_index():
    global _rebuild_in_progress
    if vector_store is None:
        raise HTTPException(status_code=503, detail="向量存储未初始化")

    lock = _get_rebuild_lock()
    if lock.locked():
        raise HTTPException(status_code=409, detail="索引重建正在进行中")

    async with lock:
        _rebuild_in_progress = True
        try:
            asyncio.create_task(_rebuild_background())
            return {
                "status": "rebuilding",
                "estimated_seconds": 180,
                "message": "索引重建已启动，请通过 /health 端点查询进度",
            }
        except Exception as e:
            _rebuild_in_progress = False
            logger.error(f"启动重建任务失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))


async def _rebuild_background():
    global _rebuild_in_progress
    try:
        chunk_count = await asyncio.to_thread(vector_store.rebuild)
        logger.info(f"后台索引重建完成: {chunk_count} 个文本块")
    except (ValueError, RuntimeError, ConnectionError, MemoryError) as e:
        logger.error(f"后台重建索引失败: {e}")
    finally:
        _rebuild_in_progress = False


# ============================================================
#  POST /rebuild-memory-index (从 SQLite 全量重建 ChromaDB 记忆索引)
# ============================================================
@router.post("/rebuild-memory-index")
async def rebuild_memory_index(user_id: str = ""):
    """从 SQLite 全量重建记忆向量索引（最终一致性兜底）。

    传 user_id 重建指定用户；不传则遍历 data/users 重建所有用户。
    """
    if memory_system_factory is None:
        raise HTTPException(status_code=503, detail="记忆系统未初始化")

    async def _rebuild_one(uid: str) -> dict:
        ms = await memory_system_factory(uid)
        if ms is None or getattr(ms, "recall_store", None) is None:
            return {"error": "memory system unavailable"}
        return await asyncio.to_thread(ms.recall_store.rebuild_chroma_index)

    if user_id:
        result = await _rebuild_one(user_id)
        return {"status": "done", "user_id": user_id, **result}

    from ..config import PROJECT_ROOT as _prj_root
    users_dir = _prj_root.parent / "data" / "users"
    results = {}
    if users_dir.is_dir():
        for child in users_dir.iterdir():
            if child.is_dir():
                try:
                    results[child.name] = await _rebuild_one(child.name)
                except Exception as e:
                    results[child.name] = {"error": str(e)}
    return {"status": "done", "results": results}
