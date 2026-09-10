"""统一工具调度层 ToolDispatcher。

收敛分散在各编排模式（agent_loop / langgraph_supervisor / evolution_workbench /
skills_pipeline）的工具调用入口，解决以下问题：

1. user_id 注入不一致 — supervisor._exec_dynasty_compare 直接调 mcp.call_tool 漏注入
   user_id，导致 delete_annotations 等 per-user 工具鉴权失败。统一在此注入。
2. tool_call_context 分散 — 各编排模式各自 with tool_call_context(...)，口径不一。
   统一在此设置 caller/purpose/upstream/downstream，MCPRegistry.call_tool 据此把
   每次调用写入 logs/tool/tool_calls.log 并关联 trace_id。
3. 非 MCP 检索路径无埋点 — evolution_workbench._filtered_retrieve 直接调
   retriever.retrieve，绕过 MCPRegistry，需手动记录。统一在 dispatch_retrieve 中处理。

职责边界（避免过度耦合）：
- MCP 工具调用的统一入口 + user_id 注入 + context + 解包 + 截断 → ToolDispatcher
- summary / sse_events 生成是 agent_loop 特有的工作区事件格式化（依赖 node_client），
  通过 summarizer / sse_builder 回调注入，不在 ToolDispatcher 内硬编码
- 异常捕获策略（fast fail vs 降级返回）由调用方决定，dispatch 不吞异常
"""
import logging
import time
import traceback
from typing import Any, Awaitable, Callable, Optional

from ..llm.tool_call import ToolCallResult
from ..observability.tracer import Tracer, tool_call_context

logger = logging.getLogger(__name__)


def _safe_record(tool_name: str, arguments: dict, result, latency_ms: float,
                 success: bool, error=None, error_type=None):
    """安全记录工具调用到 logs/tool/tool_calls.log（埋点失败不影响主流程）。

    用于 dispatch_retrieve 这类不经 MCPRegistry 的检索路径，手动补埋点。
    MCP 工具路径（dispatch）由 MCPRegistry.call_tool 内部自动埋点，无需调此函数。
    """
    try:
        Tracer.record_tool_call(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            latency_ms=latency_ms,
            success=success,
            error=error,
            error_type=error_type,
            traceback_str=traceback.format_exc() if not success else None,
        )
    except Exception as e:
        logger.debug(f"[TOOL_TRACE] dispatch_retrieve 记录失败(忽略): {e}")


class ToolDispatcher:
    """统一工具调度层。

    使用方式：
        # 在编排模式 __init__ 中自建（依赖已注入的 mcp/retriever）
        self.dispatcher = ToolDispatcher(mcp_registry=self.mcp, retriever=self.retriever)

        # MCP 工具调用（agent_loop 路径，带 summary/sse 回调）
        result = await self.dispatcher.dispatch(
            tool_name, params, caller="agent_loop",
            user_id=self.user_id,
            summarizer=self._summarize_tool_result,
            sse_builder=self._build_tool_sse_events,
        )

        # MCP 工具调用（supervisor 路径，无 UI 回调）
        result = await self.dispatcher.dispatch(
            "dynasty_compare_elements", {...},
            caller="langgraph_supervisor", purpose="dynasty_compare",
            user_id=self._request_user_id,
            upstream="_exec_dynasty_compare",
        )

        # 非 MCP 检索路径（evolution_workbench）
        docs = await self.dispatcher.dispatch_retrieve(
            query, caller="evolution_workbench",
            top_k=top_k, filters=filters, skip_tagging=skip_tagging,
        )
    """

    # 工具级截断预算覆盖（字符数），未列出的工具走 settings.context_tool_result_max_chars。
    # get_chapter_full_text 返回整章文本（paragraphs 含全文+段落索引），标注任务要求
    # LLM 先读原文再定位段落，默认 2000/4000 字符截断会导致靠后段落对 LLM 不可见。
    TOOL_RESULT_CHAR_BUDGETS: dict[str, int] = {
        "get_chapter_full_text": 16000,
    }

    # 通用精简参数
    _STR_MIN_KEEP = 200       # 字符串字段截断后至少保留的字符数

    def __init__(self, mcp_registry=None, retriever=None, config=None):
        self.mcp = mcp_registry
        self.retriever = retriever
        self.config = config

    # ================================================================
    #  MCP 工具调用统一入口
    # ================================================================

    async def dispatch(
        self,
        tool_name: str,
        arguments: dict,
        caller: str,
        purpose: Optional[str] = None,
        user_id: str = "",
        inject_user_id: bool = True,
        upstream: str = "",
        downstream: str = "mcp_call_tool",
        summarizer: Optional[Callable[[str, Any], str]] = None,
        sse_builder: Optional[Callable[[str, Any], Awaitable[list[dict]]]] = None,
        on_progress: Optional[Callable[[str], Awaitable[None]]] = None,
        truncate: bool = True,
    ) -> ToolCallResult:
        """统一 MCP 工具调用入口。

        - 自动注入 user_id（确保 delete_annotations 等 per-user 工具鉴权通过）
        - 设置 tool_call_context（MCPRegistry.call_tool 内部 _safe_record_tool_call
          据此写入 logs/tool/tool_calls.log 并关联 trace_id）
        - 调 mcp.call_tool → 解包 raw → 调 summarizer/sse_builder（基于完整 raw）→ 截断
        - 异常不吞，由调用方决定 fast fail / 降级返回

        Args:
            tool_name: 工具名
            arguments: 工具参数
            caller: 调用方标识（agent_loop / langgraph_supervisor / skills_pipeline / evolution_workbench）
            purpose: 语义标签，默认同 tool_name
            user_id: 用户 ID，用于注入到 per-user 工具
            inject_user_id: 是否自动注入 user_id
            upstream/downstream: 链路上下游（写入 tool_calls.log 便于定位）
            summarizer: 可选，对完整 raw 生成简短摘要（供 LLM 下一轮决策）
            sse_builder: 可选，对完整 raw 生成额外 SSE 事件（agent_loop 工作区事件）
            on_progress: 可选，长耗时工具中途进度回调 async (message: str) -> None，
                经 registry 注入 handler（仅 handler 声明 progress_callback 时生效）
            truncate: 是否截断过大的 result（agent_loop 喂给 LLM context 需截断；
                supervisor 直接给 UI 的结果不应截断）

        Returns:
            ToolCallResult（truncate=True 时 result 已截断；summary/sse_events 基于完整 raw 生成）
        """
        call_id = f"tc_{int(time.monotonic() * 1000)}"
        if not self.mcp:
            return ToolCallResult(
                call_id=call_id, tool_name=tool_name,
                success=False, error="MCP registry 不可用",
            )

        params = dict(arguments) if arguments else {}
        # user_id 注入：LLM 不知道 user_id，由调度层统一注入，
        # 确保 delete_annotations / annotate_user_request 等 per-user 工具
        # 能通过 Node.js 鉴权中间件的 x-user-id 校验
        if inject_user_id and user_id and "user_id" not in params:
            params["user_id"] = user_id

        # tool_call_context：MCPRegistry.call_tool 内部 _safe_record_tool_call 会读取
        # 此上下文，把本次调用标注到对应 caller/purpose，并写入 logs/tool/tool_calls.log
        with tool_call_context(
            caller=caller, purpose=purpose or tool_name,
            upstream=upstream, downstream=downstream,
        ):
            result = await self.mcp.call_tool(tool_name, params,
                                              progress_callback=on_progress)

        # 解包：mcp.call_tool 成功返回 {"result": ...}，失败返回 {"error": ...}
        raw = result.get("result", result) if isinstance(result, dict) else result

        # summary / sse_events 基于完整 raw 生成（截断前），保留工作区事件所需细节
        summary = summarizer(tool_name, raw) if summarizer else ""
        sse_events = await sse_builder(tool_name, raw) if sse_builder else []

        # 截断过大的工具返回结果，防止单条 tool_result 撑爆 LLM 上下文
        # 注意：summary 和 sse_events 已基于完整 raw 生成，此处只截断存入 LLM context 的副本
        llm_raw = self._truncate(raw, tool_name=tool_name) if truncate else raw

        return ToolCallResult(
            call_id=call_id, tool_name=tool_name, success=True,
            result=llm_raw, summary=summary, sse_events=sse_events,
            # v19.2: 保留完整原始结果，REPORT_READY 等事件需要完整数据
            full_result=raw,
        )

    def _truncate(self, raw: Any, tool_name: str = "") -> Any:
        """截断过大的工具返回结果（写入 LLM context 的副本）。

        v18 修复：原实现对超长 dict 一律精简为 annotations_count/target/insights_count
        三个标注类字段，导致 search_paragraphs / web_search 等检索类工具超长时
        results 全部丢失，LLM 误以为"没有搜到"。现改为通用字段感知精简：

        - str：超长截头并追加截断标记
        - dict/list：递归精简——标量元信息字段（query/totalMatches/...）全保留，
          列表保留前 N 条，大字符串按预算截断，所有丢弃处带明确标记，
          LLM 能感知"还有多少内容被省略"并决定是否用更精确的参数重查。

        预算：优先取 TOOL_RESULT_CHAR_BUDGETS[tool_name]，否则 settings 默认值。
        """
        from ..config import settings as _cfg
        max_chars = self.TOOL_RESULT_CHAR_BUDGETS.get(
            tool_name, _cfg.context_tool_result_max_chars
        )
        return self._shrink_value(raw, max_chars)

    def _shrink_value(self, value: Any, budget: int) -> Any:
        """把任意 JSON 值精简到约 budget 字符内，保留结构与截断标记。"""
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            if len(value) <= budget:
                return value
            keep = max(self._STR_MIN_KEEP, budget - 80)
            return value[:keep] + f"…[已截断，原文共 {len(value)} 字符]"
        if isinstance(value, list):
            if not value:
                return value
            # 预算驱动：条目多时按预算均分单条预算；逐条累加，装不下才截断。
            # 不设硬性条数上限——get_chapter_full_text 的段落列表（整章正文）
            # 在预算内应全量保留，硬上限会把靠后段落对 LLM 隐藏。
            per_item = max(
                self._STR_MIN_KEEP, budget // max(1, min(len(value), 16))
            )
            kept = []
            used = 0
            for item in value:
                shrunk = self._shrink_value(item, per_item)
                size = len(str(shrunk))
                if kept and used + size > budget:
                    kept.append(
                        f"…[列表已截断：显示前 {len(kept)} 条 / 共 {len(value)} 条]"
                    )
                    break
                kept.append(shrunk)
                used += size
            return kept
        if isinstance(value, dict):
            # 字段感知：标量与短字符串（query/totalMatches/title 等元信息）
            # 直接保留不占预算，剩余预算按大字段体量加权分配，
            # 避免 sortOptions 等小结构挤占 results/content 的预算
            out: dict = {}
            sizes: dict = {}
            remaining = budget
            for k, v in value.items():
                if v is None or isinstance(v, (bool, int, float)):
                    out[k] = v
                elif isinstance(v, str) and len(v) <= 120:
                    remaining -= len(v)
                    out[k] = v
                else:
                    sizes[k] = max(1, len(str(v)))
            if sizes:
                total = sum(sizes.values())
                for k, sz in sizes.items():
                    share = max(
                        self._STR_MIN_KEEP, int(remaining * sz / total)
                    )
                    out[k] = self._shrink_value(value[k], share)
            return out
        return str(value)

    # ================================================================
    #  非 MCP 检索路径统一入口
    # ================================================================

    async def dispatch_retrieve(
        self,
        query: str,
        caller: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,
        purpose: str = "filtered_retrieve",
        upstream: str = "",
        downstream: str = "retriever.retrieve",
        skip_tagging: bool = False,
    ) -> list:
        """统一检索调用入口（不经 MCPRegistry，直接调 retriever.retrieve）。

        透明处理不支持硬过滤的检索器（如 TagPipeline）：
        - 检索器支持 filters → 直接传 filters 检索
        - 不支持 → 过滤值并入查询词 + 取回后按 metadata 本地严格过滤（绝不混入其他朝代）
        - skip_tagging=True 时对 TagPipeline 类检索器跳过在线 LLM 打标（演化查询是关键词串）

        手动补埋点（_safe_record）：检索不经 MCPRegistry，需手动记录输入输出到
        logs/tool/tool_calls.log，便于定位 E2/E7 检索零结果等问题。
        """
        import asyncio  # 局部导入，避免与模块级 asyncio 循环依赖

        t0 = time.monotonic()
        args = {"query": query, "filters": filters, "top_k": top_k, "skip_tagging": skip_tagging}

        with tool_call_context(
            caller=caller, purpose=purpose,
            upstream=upstream, downstream=downstream,
        ):
            if not filters or getattr(self.retriever, "supports_filters", True):
                kwargs = {"skip_tagging": True} if (
                    skip_tagging and getattr(self.retriever, "supports_skip_tagging", False)) else {}
                docs = await self.retriever.retrieve(query, top_k=top_k, filters=filters, **kwargs)
                _safe_record("filtered_retrieve", args, docs,
                             (time.monotonic() - t0) * 1000, success=True)
                return docs

            # 不支持硬过滤：过滤值并入查询词，取回后本地严格过滤
            filter_suffix = " ".join(str(v) for v in filters.values())
            boosted = f"{query} {filter_suffix}".strip()
            fetch_k = top_k or 20
            try:
                docs = await self.retriever.retrieve(
                    boosted, top_k=fetch_k * 2,
                    **({"skip_tagging": True} if getattr(self.retriever, "supports_skip_tagging", False) else {}),
                )
            except (ValueError, ConnectionError, RuntimeError, OSError, TimeoutError, asyncio.TimeoutError) as e:
                # B-164: 基础设施错误（OOM/模型未加载/连接失败/超时）必须向上传播，
                # 禁止 return [] 把故障伪装成"查无证据"——2026-09-09 E7 OOM 时
                # 本分支吞掉 RuntimeError 返回空列表，导致 6/6 主张 0 证据误判
                # insufficient。埋点保留后 re-raise，由调用方（E2 cell 失败兜底 /
                # E7 claim 标 retrieval_error）决定失败语义。
                logger.warning(f"dispatch_retrieve 失败 [{boosted[:40]}]: {type(e).__name__}: {str(e)[:150]}")
                _safe_record("filtered_retrieve", args, None,
                             (time.monotonic() - t0) * 1000, success=False,
                             error=str(e), error_type=type(e).__name__)
                raise
            out = []
            for d in docs or []:
                md = d.get("metadata", {}) or {}
                if all(str(md.get(k, "")) == str(v) for k, v in filters.items()):
                    out.append(d)
            out = out[:fetch_k] if fetch_k else out
            _safe_record("filtered_retrieve", args, out,
                         (time.monotonic() - t0) * 1000, success=True)
            return out
