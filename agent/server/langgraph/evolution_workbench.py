"""文脉演化工作台 v8 — 12 节点 · 3 中断 · 人机协同。

核心升级 (v6 -> v8):
- 12 节点替换 11 节点
- 3 个 interrupt_before 实现真正的人机协同
- 母题分解(E1) -> 矩阵扫描(E2) -> 变化分类(E3) -> 谱系推断(E4) -> 外部关联(E5)
  -> 假设生成(E6) -> 中断 #2 -> 主张验证(E7) -> 条件路由 -> 综合(E8)

fast fail: 所有节点只捕获具有明确降级策略的具体异常类型。
非关键路径（检索补充上下文、AI解读）失败不影响主流程。
LLM/网络不可用等致命错误直接向上传播，不静默吞掉。

B-030 ~ B-055: v8 完整实现
"""
import asyncio
import json
import logging
import re
import time
from typing import Optional, AsyncGenerator

from langgraph.graph import StateGraph, END
from langgraph.types import Command

from .state import EvolutionWorkbenchState
from ..llm.provider import LLMFatalError, RateLimitError, APITimeoutError
from ..llm.json_utils import parse_llm_json
from ..api.events import SSEEventType, build_sse_event
from ..observability.tracer import Tracer, llm_call_context
from ..dispatch.tool_dispatcher import ToolDispatcher

logger = logging.getLogger(__name__)


# LangGraph 版本检测 (B-047b)
try:
    from importlib.metadata import version as _get_pkg_version
    _LG_VERSION_STR = _get_pkg_version('langgraph')
    _LG_VERSION = tuple(int(x) for x in _LG_VERSION_STR.split(".")[:2])
except Exception:
    _LG_VERSION = (0, 0)
    _LG_VERSION_STR = 'unknown'

logger.info(f"LangGraph version: {_LG_VERSION_STR}, "
            f"interrupt_mode: {'native' if _LG_VERSION >= (0, 2) else 'fallback'}")

# B-096: Resume 超时（10 分钟）
RESUME_TIMEOUT_SECONDS = 600

# 已知 LLM 相关异常（这些是不可恢复的，应向上传播）
_LLM_FATAL = (LLMFatalError, RateLimitError, APITimeoutError)

# B-156: 流式队列哨兵 — astream drainer 完成时发送，主循环收到后退出
_STREAM_END = object()


class EvolutionWorkbench:
    """文脉演化工作台 v8 — 人机协同演化分析。"""

    def __init__(self, llm, retriever, prompt_registry, mcp_registry=None, checkpointer=None, node_client=None):
        self.llm = llm
        self.retriever = retriever
        self.prompts = prompt_registry
        self.mcp = mcp_registry
        # 统一工具调度层：检索路径（dispatch_retrieve）统一埋点 + 过滤值合并
        self.dispatcher = ToolDispatcher(mcp_registry=self.mcp, retriever=self.retriever)
        self._checkpointer = checkpointer
        self.node_client = node_client
        self._graph = self._build_graph()
        self._llm_call_count = 0
        self._cancel_event = asyncio.Event()
        # B-156: 节点内实时进度推送队列。run_stream/resume_stream 激活时，
        # 节点可通过 _emit_live() 在执行中途立即推送事件，而非等到 return。
        self._live_queue: asyncio.Queue | None = None

    def cancel(self):
        """用户取消演化分析。fast fail: 设置 cancel flag 后 run_stream 在下一个 astream yield 时退出。"""
        self._cancel_event.set()

    def _emit_live(self, event: dict):
        """B-156: 节点内实时推送 SSE 事件。

        仅当 run_stream/resume_stream 已激活 _live_queue 时生效（put_nowait 非阻塞）。
        用于 E2 矩阵扫描等长耗时单节点内部循环，让前端实时看到进度而非等到节点 return。
        """
        if self._live_queue is not None:
            try:
                self._live_queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    # ====================================================================
    #  B-046: v8 图组装
    # ====================================================================

    def _build_graph(self):
        workflow = StateGraph(EvolutionWorkbenchState)

        workflow.add_node("decompose_motifs", self._decompose_motifs)
        workflow.add_node("confirm_motifs", self._confirm_motifs)
        workflow.add_node("scan_motif_matrix", self._scan_motif_matrix)
        workflow.add_node("classify_changes", self._classify_changes)
        workflow.add_node("infer_phylogeny", self._infer_phylogeny)
        workflow.add_node("correlate_external", self._correlate_external)
        workflow.add_node("formulate_hypothesis", self._formulate_hypothesis)
        workflow.add_node("select_hypothesis", self._select_hypothesis)
        workflow.add_node("prepare_claims", self._prepare_claims)
        workflow.add_node("test_claims_retrieve", self._test_claims_retrieve)
        workflow.add_node("test_one_claim", self._test_one_claim)
        workflow.add_node("decide_next_action", self._decide_next_action)
        workflow.add_node("reflect_and_re_search", self._reflect_and_re_search)
        workflow.add_node("synthesize", self._synthesize)
        workflow.add_node("present", self._present)

        workflow.set_entry_point("decompose_motifs")

        workflow.add_edge("decompose_motifs", "confirm_motifs")
        workflow.add_edge("confirm_motifs", "scan_motif_matrix")
        workflow.add_edge("scan_motif_matrix", "classify_changes")
        workflow.add_edge("classify_changes", "infer_phylogeny")
        workflow.add_edge("infer_phylogeny", "correlate_external")
        workflow.add_edge("correlate_external", "formulate_hypothesis")
        workflow.add_edge("formulate_hypothesis", "select_hypothesis")
        workflow.add_edge("select_hypothesis", "prepare_claims")
        workflow.add_edge("prepare_claims", "test_claims_retrieve")
        workflow.add_edge("test_claims_retrieve", "test_one_claim")

        workflow.add_conditional_edges(
            "test_one_claim", self._route_after_one_claim,
            {"continue": "test_one_claim", "synthesize": "synthesize", "decide": "decide_next_action"}
        )

        workflow.add_conditional_edges(
            "decide_next_action", self._route_after_decision,
            {"investigate": "reflect_and_re_search", "reexplore": "scan_motif_matrix",
             "accept": "synthesize", "change_hypothesis": "select_hypothesis"}
        )

        workflow.add_edge("reflect_and_re_search", "test_one_claim")
        workflow.add_edge("synthesize", "present")
        workflow.add_edge("present", END)

        kwargs = {}
        if self._checkpointer is not None:
            kwargs["checkpointer"] = self._checkpointer

        return workflow.compile(
            interrupt_before=[
                "confirm_motifs",
                "select_hypothesis",
                "decide_next_action",
            ],
            **kwargs
        )

    # ====================================================================
    #  B-040 / B-042: 条件路由（纯函数，无异常可能）
    # ====================================================================

    @staticmethod
    def _route_after_one_claim(state: EvolutionWorkbenchState) -> str:
        """单条主张验证后的条件路由。

        - 还有未处理的 claim -> continue (循环回 test_one_claim)
        - 深入调查重判完成 -> 强制 decide（让用户再次选择）
        - 全部处理完 + 有 insufficient/refuted + 还有重试次数 -> decide
        - 全部处理完 + 全部 supported 或重试耗尽 -> synthesize
        """
        claims = state.get("claims", [])
        if not claims:
            return "synthesize"
        test_iter = state.get("test_iteration", 0)
        if test_iter >= 3:
            return "synthesize"
        total = len(claims)
        claim_index = state.get("claim_index", 0)
        if claim_index < total:
            return "continue"
        # 深入调查重判完成后强制回到决策面板
        if state.get("_from_deep_investigation"):
            return "decide"
        # 全部判定完成，检查结果
        supported = sum(1 for c in claims if c.get("verdict") == "supported")
        refuted = sum(1 for c in claims if c.get("verdict") == "refuted")
        insufficient = sum(1 for c in claims if c.get("verdict") == "insufficient")
        if (refuted > 0 or insufficient > 0) and test_iter < 3:
            return "decide"
        return "synthesize"

    @staticmethod
    def _route_after_decision(state: EvolutionWorkbenchState) -> str:
        decision = state.get("decision") or state.get("user_decision", "accept")
        valid = {"investigate", "reexplore", "accept", "change_hypothesis"}
        if decision not in valid:
            logger.warning("E7 route: 未知 decision '%s'，回退到 accept", decision)
            decision = "accept"
        return decision

    # ====================================================================
    #  B-031: decompose_motifs (E1)
    # ====================================================================

    async def _decompose_motifs(self, state: EvolutionWorkbenchState) -> dict:
        """E1: 调用 decompose_motifs MCP 工具 -> motifs[]。

        fast fail: MCP 不可用或 LLM 返回空 -> 使用内置 fallback motifs。
        """
        tracking_target = state.get("tracking_target", "")
        target_type = state.get("target_type", "location")

        if not self.mcp:
            logger.warning("E1: MCP 不可用，使用 fallback motifs")
            motifs = _build_fallback_motifs(tracking_target, target_type)
            return _build_e1_result(tracking_target, motifs, [], state.get("thread_id", ""), degraded=True)

        # B-156: E1 开始即推送，让用户知道正在分解追踪目标
        self._emit_live(build_sse_event(
            SSEEventType.THINKING_STEP,
            step="decompose_motifs",
            message=f"E1 分解母题: 正在将追踪目标「{tracking_target[:20]}」分解为叙事母题（LLM 分析中）..."
        ))
        try:
            result = await asyncio.wait_for(
                self.mcp.call_tool("decompose_motifs", {
                    "tracking_target": tracking_target,
                    "target_type": target_type,
                }),
                timeout=45.0
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"E1: decompose_motifs MCP 调用超时 (45s)，使用 fallback motifs "
                f"tracking_target={tracking_target}"
            )
            motifs = _build_fallback_motifs(tracking_target, target_type)
            return _build_e1_result(tracking_target, motifs, [], state.get("thread_id", ""), degraded=True)
        data = result.get("result", result)

        if data.get("error"):
            logger.error(
                f"E1: decompose_motifs MCP 工具返回错误: {data['error'][:200]}\n"
                f"   tracking_target={tracking_target} target_type={target_type}"
            )
            motifs = _build_fallback_motifs(tracking_target, target_type)
            return _build_e1_result(tracking_target, motifs, [], state.get("thread_id", ""), degraded=True)

        motifs = data.get("motifs", [])
        suggested = data.get("suggested_additions", [])

        if len(motifs) == 0:
            logger.error(
                f"E1: LLM 返回 0 个母题! tracking_target={tracking_target} "
                f"raw_keys={list(data.keys())}"
            )
            motifs = _build_fallback_motifs(tracking_target, target_type)
            return _build_e1_result(tracking_target, motifs, suggested, state.get("thread_id", ""), degraded=True)

        if len(motifs) < 3:
            logger.warning(f"E1: 仅生成 {len(motifs)} 个母题，低于预期的 3-7 个")

        # B-156: LLM 输出预览 — 让用户看到分解出了哪些母题
        _preview = "、".join(
            f"{m.get('id', '?')}:{(m.get('name', '') or '')[:10]}" for m in motifs[:6]
        )
        self._emit_live(build_sse_event(
            SSEEventType.THINKING_STEP,
            step="decompose_motifs",
            message=f"E1 LLM 输出: 分解出 {len(motifs)} 个母题 — {_preview}"
        ))

        return _build_e1_result(tracking_target, motifs, suggested, state.get("thread_id", ""), degraded=False)

    # ====================================================================
    #  B-032: confirm_motifs (Interrupt #1 handler)
    # ====================================================================

    async def _confirm_motifs(self, state: EvolutionWorkbenchState) -> dict:
        # 检查 state 中是否有通过 Command(resume=...) 注入的编辑后母题
        # LangGraph 0.6+: resume 数据以额外 key 形式合并到 state 中
        edited_motifs = state.get("edited_motifs")  # resume 中的 key 可能被直接合并
        if edited_motifs and isinstance(edited_motifs, list) and len(edited_motifs) > 0:
            # 校验每个母题的必要字段
            valid_motifs = []
            for i, m in enumerate(edited_motifs):
                if not isinstance(m, dict):
                    logger.warning(f"E1 confirm: 跳过非 dict 元素 [{i}]: {type(m).__name__}")
                    continue
                mid = m.get("id") or m.get("name")
                if not mid:
                    logger.warning(f"E1 confirm: 跳过缺少 id/name 的母题 [{i}]")
                    continue
                if "name" not in m:
                    m["name"] = str(mid)
                if "id" not in m:
                    m["id"] = f"M{i+1}"
                valid_motifs.append(m)
            if valid_motifs:
                logger.info(
                    f"E1 confirm: 应用用户编辑的 {len(valid_motifs)} 个母题 "
                    f"(原始: {len(state.get('motifs', []))} 个, 丢弃: {len(edited_motifs)-len(valid_motifs)} 个)"
                )
                return {
                    "motifs": valid_motifs,
                    "current_phase": "E1_confirmed",
                    "pending_events": [
                        build_sse_event(SSEEventType.STATUS,
                                        message=f"母题已确认（{len(valid_motifs)} 个），开始扫描母题矩阵...")
                    ]
                }
        return {"current_phase": "E1_confirmed",
                "pending_events": [
                    build_sse_event(SSEEventType.STATUS,
                                    message=f"母题已确认（{len(state.get('motifs', []))} 个），开始扫描母题矩阵...")
                ]}

    async def _filtered_retrieve(self, query: str, filters: dict | None = None,
                                 top_k: int | None = None, skip_tagging: bool = False):
        """B-152: 检索代理 — 委托 ToolDispatcher.dispatch_retrieve 统一调度。

        透明处理不支持硬过滤的检索器（如 TagPipeline）：过滤值并入查询词 +
        取回后按 metadata 本地严格过滤。埋点（logs/tool/tool_calls.log）由
        dispatch_retrieve 内部统一完成。
        """
        return await self.dispatcher.dispatch_retrieve(
            query, caller="evolution_workbench",
            top_k=top_k, filters=filters, skip_tagging=skip_tagging,
            upstream="evolution_node", downstream="retriever.retrieve",
        )

    # ====================================================================
    #  B-033: scan_motif_matrix (E2)
    # ====================================================================

    async def _scan_motif_matrix(self, state: EvolutionWorkbenchState) -> dict:
        """E2: 母题 x 朝代 并行检索。

        fast fail: 单细胞检索失败 -> 该细胞 count=0，不阻塞整体网格。

        reexplore 时使用更宽的检索策略以发现新的证据模式。
        """
        motifs = state.get("motifs", [])
        dynasties = state.get("dynasties") or ["唐", "宋", "明", "清"]
        logger.info(f"E2 START: motifs={[m.get('id','?')+':'+m.get('name','?')[:20] for m in motifs]} dynasties={dynasties}")
        reexplore_count = state.get("reexplore_count", 0)
        is_reexplore = reexplore_count > 0

        if not motifs:
            return {"motif_matrix": [], "dynasties": dynasties, "current_phase": "E2_complete"}

        # ★ 方向2: LLM 桥接 — 为每个母题生成古籍中可能出现的搜索关键词
        # 母题名称是分析概念(如"镇压与囚禁"), embedding 跟文言文不互通
        # 让 fast LLM 把概念"翻译"成古籍关键词, 提高检索命中率
        keyword_done = 0  # B-156: 关键词生成进度计数
        async def generate_keywords(motif: dict) -> dict:
            nonlocal keyword_done
            motif_copy = dict(motif)
            name = motif_copy.get("name", "")
            if motif_copy.get("_search_keywords"):
                keyword_done += 1
                self._emit_live(build_sse_event(
                    SSEEventType.THINKING_STEP,
                    step="scan_motif_matrix",
                    message=f"E2 关键词生成 {keyword_done}/{len(motifs)}: [{name}] 已有，跳过"
                ))
                return motif_copy
            core = motif_copy.get("core", "")
            from ..config import settings as _cfg
            system = (
                "你是古籍文献检索专家。给定一个文学分析概念，输出5-8个最可能在古籍原文（文言文/古白话）中出现的"
                "搜索关键词，用空格分隔。关键词必须是可能在古文中出现的具体词语，不要现代学术术语。只输出关键词。"
            )
            user = f"分析概念: {name}\n概念含义: {core}\n\n古籍搜索关键词(5-8个, 空格分隔):"
            try:
                with llm_call_context("scan_motif_matrix", "generate_keywords",
                                      upstream="decompose_motifs",
                                      downstream="scan_motif_matrix"):
                    resp = await asyncio.wait_for(
                        self.llm.generate(system, user, model=_cfg.effective_fast_model,
                                       max_tokens=100, temperature=0.4, thinking_disabled=True),
                        timeout=15,
                    )
                keywords = resp.content.strip().replace("\n", " ").replace("，", " ").replace(",", " ")
                # B-152: trace 实锤 4/6 输出是指令复述（"我们要求输出5-8个…"）或
                # 单字垃圾（"降"），直接当查询词会污染检索。含指令特征词或过短则回退。
                if len(keywords) < 4 or any(w in keywords for w in
                                            ("要求", "针对", "概念", "输出", "关键词", "含义", "分析")):
                    logger.warning(f"E2 keywords [{name}] 疑似指令复述/垃圾输出，回退母题名: {keywords[:60]}")
                    keywords = f"{name} {core[:60]}"
                motif_copy["_search_keywords"] = keywords
                logger.info(f"E2 keywords [{name}]: {keywords[:80]}")
                # B-156: 逐题推送进度，避免9个母题并行生成期间前端长时间无反馈
                keyword_done += 1
                self._emit_live(build_sse_event(
                    SSEEventType.THINKING_STEP,
                    step="scan_motif_matrix",
                    message=f"E2 关键词生成 {keyword_done}/{len(motifs)}: [{name}] → {keywords[:80]}"
                ))
            except Exception as e:
                logger.warning(f"E2 keywords generation failed for [{name}]: {e}")
                motif_copy["_search_keywords"] = f"{name} {core[:60]}"
                self._emit_live(build_sse_event(
                    SSEEventType.THINKING_STEP,
                    step="scan_motif_matrix",
                    message=f"E2 关键词生成: [{name}] 失败，回退母题名"
                ))
            return motif_copy

        # B-156: 关键词生成阶段（LLM 调用）先推送开始事件，让用户知道在干什么
        round_label_pre = f"第{reexplore_count}轮" if is_reexplore else "首次"
        self._emit_live(build_sse_event(
            SSEEventType.THINKING_STEP,
            step="scan_motif_matrix",
            message=f"E2 矩阵扫描（{round_label_pre}）: 为 {len(motifs)} 个母题生成古籍检索关键词中..."
        ))
        motifs_with_keywords = await asyncio.gather(*[generate_keywords(m) for m in motifs])

        if is_reexplore:
            logger.info(f"E2 reexplore round {reexplore_count}: 使用更宽检索策略")

        total_cells = len(motifs_with_keywords) * len(dynasties)
        # B-156: 关键词就绪，开始扫描单元格
        round_label = f"第{reexplore_count}轮重新探索" if is_reexplore else "首次扫描"
        self._emit_live(build_sse_event(
            SSEEventType.THINKING_STEP,
            step="scan_motif_matrix",
            message=f"关键词就绪，开始扫描 {total_cells} 个单元格（{len(motifs_with_keywords)} 母题 × {len(dynasties)} 朝代）"
        ))
        sem = asyncio.Semaphore(8)
        async def scan_with_limit(motif, dynasty, idx):
            async with sem:
                try:
                    res = await self._scan_one_cell(motif, dynasty, reexplore=is_reexplore)
                    return idx, res
                except BaseException as e:
                    return idx, e
        # 带索引的任务，as_completed 完成即推送实时进度（每格直推，B-156 经 _emit_live 直达）
        scan_tasks = []
        idx = 0
        # idx → (母题名, 朝代) 映射，用于细粒度进度文案
        idx_info: dict = {}
        for motif in motifs_with_keywords:
            _mname = (motif.get("name", "") or "")[:14]
            for dynasty in dynasties:
                scan_tasks.append(scan_with_limit(motif, dynasty, idx))
                idx_info[idx] = (_mname, dynasty)
                idx += 1

        results_by_idx: dict = {}
        completed = 0
        for coro in asyncio.as_completed(scan_tasks):
            cidx, result = await coro
            results_by_idx[cidx] = result
            completed += 1
            _mname, _dy = idx_info.get(cidx, ("?", "?"))
            # 细粒度：每格完成即推送，含母题名+朝代+结果数，让用户实时看到"在扫什么、找到了几处"
            if isinstance(result, dict):
                cnt = result.get("count", 0)
                found_txt = f"找到{cnt}处证据" if cnt > 0 else "未发现"
            elif isinstance(result, BaseException):
                found_txt = "检索异常"
            else:
                found_txt = "未知结果"
            self._emit_live(build_sse_event(
                SSEEventType.THINKING_STEP,
                step="scan_motif_matrix",
                message=f"扫描 {_mname}→{_dy}: {found_txt}（{completed}/{total_cells}，{completed * 100 // total_cells}%）"
            ))

        matrix = []
        idx = 0
        for motif in motifs_with_keywords:
            for dynasty in dynasties:
                result = results_by_idx.get(idx)
                if isinstance(result, BaseException):
                    logger.warning(f"E2 scan_cell {motif.get('id', '')}@{dynasty} 失败: {result}")
                    matrix.append({
                        "motif_id": motif.get("id", ""),
                        "motif_name": motif.get("name", ""),
                        "dynasty": dynasty,
                        "occurrence_count": 0,
                        "documents": [],
                        "summary": str(result),
                    })
                elif isinstance(result, dict):
                    matrix.append({
                        "motif_id": motif.get("id", ""),
                        "motif_name": motif.get("name", ""),
                        "dynasty": dynasty,
                        "occurrence_count": result.get("count", 0),
                        "documents": result.get("docs", []),
                        "summary": result.get("summary", ""),
                    })
                idx += 1

        # Build structured cells for frontend {rows, columns, cells} format
        # 前端 findCell() 用 split(':')[0] 提取 motifId 再匹配 cell.motifId
        # 所有母题统一用 "ID: 名称" 格式，确保 findCell 能正确匹配
        motif_names = [
            f"{m.get('id', '')}: {m.get('name', '')}"
            for m in motifs_with_keywords
        ]
        cells = []
        for cell in matrix:
            occ = cell.get("occurrence_count", 0)
            docs = cell.get("documents", [])
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
                        "chapterNumber": _cn_to_arabic(d.get("metadata", {}).get("chapter_number", ""), d.get("metadata", {}).get("source_type", "primary_literature")),
                        "chapterTitle": d.get("metadata", {}).get("chapter_title", ""),
                        "paragraphIndex": d.get("metadata", {}).get("paragraph_index", 0),
                        "excerpt": (d.get("content", "") or "")[:2000],
                        "dynasty": d.get("metadata", {}).get("dynasty", ""),
                    }
                    for d in docs
                ],
            })

        non_empty = sum(1 for c in matrix if c.get('occurrence_count', 0) > 0)
        total_cells = len(matrix)
        if non_empty == 0:
            logger.warning(
                f"E2 DONE: {len(motifs_with_keywords)}x{len(dynasties)}={total_cells} cells, "
                f"ALL EMPTY — retrieval failure suspected"
            )
        elif non_empty < total_cells * 0.25:
            logger.warning(
                f"E2 DONE: {len(motifs_with_keywords)}x{len(dynasties)}={total_cells} cells, "
                f"only {non_empty} non-empty ({non_empty * 100 // total_cells}%)"
            )
        else:
            logger.info(
                f"E2 DONE: {len(motifs_with_keywords)}x{len(dynasties)}={total_cells} cells, "
                f"{non_empty} non-empty"
            )
        return {
            "motif_matrix": matrix, "dynasties": dynasties, "motifs": motifs_with_keywords,
            "current_phase": "E2_complete",
            "pending_events": [
                build_sse_event(SSEEventType.THINKING_STEP,
                                step="scan_motif_matrix",
                                message=f"母题矩阵扫描完成: {len(motifs_with_keywords)}x{len(dynasties)}={len(motifs_with_keywords) * len(dynasties)} 个单元格"),
                build_sse_event(SSEEventType.MOTIF_MATRIX,
                                rows=dynasties, columns=motif_names, cells=cells,
                                message=f"母题矩阵扫描完成: {len(motifs_with_keywords)}x{len(dynasties)}={len(motifs_with_keywords) * len(dynasties)} 个单元格")
            ]
        }

    async def _scan_one_cell(self, motif: dict, dynasty: str, reexplore: bool = False) -> dict:
        """扫描单个母题 x 朝代单元格。

        多级检索策略（从精确到宽松）：
        1. LLM关键词 + motif名 + core + dynasty filter
        2. motif名 + core + dynasty filter
        3. LLM关键词 + dynasty filter
        4. motif名 + core（无 dynasty filter）
        5. motif名（无 dynasty filter，最宽松）

        fast fail: 单级检索失败继续下一级。全部失败返回 count=0。

        B-158 检索防御（不改变检索语义，只加时间护栏）：
        - 单次检索超时 30s：正常单次 3-5s，超时说明调度链路异常（embedding/向量库
          卡死等），放弃该级继续下一级，避免一个格子拖住整个 E2；
        - 单格总预算 150s：约 10 倍于典型耗时（5 级 × 3s），超预算即返回已收集的
          部分证据（深度语料仍可攒满 top_k 条，不受影响），防止病态格子在
          8 路并发下拖慢全场。

        B-161 早停 + 过滤前移：
        - 每级取回后立即按 cell 朝代过滤 + 去重（原 B-152k 在循环外统一过滤，
          会导致无效文档占用 top_k 额度）；
        - 连续 3 级过滤后无新增即停（语料现实：双硬过滤下单次检索只回 1-3 条，
          跑满 5 级多为无效功）。
        """
        motif_name = motif.get('name', '')
        motif_core = motif.get('core', '')
        search_keywords = motif.get('_search_keywords', '')

        top_k = 30 if reexplore else 20
        # B-158: 时间护栏（防御措施，不改检索策略与早停语义）
        _RETRIEVE_TIMEOUT_S = 30.0
        _CELL_BUDGET_S = 150.0
        cell_deadline = time.monotonic() + _CELL_BUDGET_S

        # 按优先级从精确到宽松排列
        queries: list[tuple[str, bool]] = []
        if search_keywords:
            queries = [
                (f"{search_keywords} {motif_name} {motif_core[:80]}", True),
                (f"{motif_name} {motif_core[:80]}", True),
                (f"{search_keywords}", True),
                (f"{motif_name} {motif_core[:80]}", False),
                (f"{motif_name}", False),
            ]
        else:
            queries = [
                (f"{motif_core} {motif_name}", True),
                (f"{motif_name}", True),
                (f"{motif_name} {motif_core}", False),
                (f"{motif_name}", False),
            ]

        docs: list[dict] = []
        seen: set[str] = set()
        # B-161: 连续 N 级"过滤后新增 = 0"即早停 — 语料现实是单次检索只回 1-3 条
        # （primary_literature + dynasty 双硬过滤），跑满 5 级多为无效功；
        # 保守取 3（多保留一轮宽松检索机会），配合下方朝代过滤前移生效。
        _EARLY_STOP_IDLE_LEVELS = 3
        idle_levels = 0

        for q_text, use_filter in queries:
            if len(docs) >= top_k:
                break
            # B-158: 单格预算耗尽 → 返回已收集的部分证据，不再发起新检索
            if time.monotonic() >= cell_deadline:
                logger.warning(
                    f"E2 scan cell budget({_CELL_BUDGET_S:.0f}s) exhausted {motif.get('id','')}@{dynasty}, "
                    f"returning partial {len(docs)} docs"
                )
                break
            try:
                search_query = f"{q_text} {dynasty}" if use_filter else q_text
                # B-152i: 矩阵只检索主文献 — 研究文献无朝代归属，
                # 混入会导致每个朝代 cell 都塞满研究文献（日志实锤"唐"网格泛滥）
                _filters = {"source_type": "primary_literature"}
                if use_filter:
                    _filters["dynasty"] = dynasty
                # B-158: 单次检索超时防御 — wait_for 到点取消，该级按无结果处理
                batch = await asyncio.wait_for(
                    self._filtered_retrieve(
                        search_query,
                        filters=_filters,
                        top_k=top_k,
                        skip_tagging=True,  # B-155: 关键词串打标是噪声，跳过
                    ),
                    timeout=_RETRIEVE_TIMEOUT_S,
                )
                # B-161: 朝代过滤前移 — B-152k 原先在循环外统一过滤，前几级攒的
                # 文档（尤其第4-5级无 dynasty filter 的）大量最终被扔，白白占用
                # top_k 额度并拖长扫描。改为每级取回后立即过滤+去重，早停判断
                # 基于真实有效新增。
                level_new = 0
                for d in batch:
                    if str((d.get("metadata", {}) or {}).get("dynasty", "")) != dynasty:
                        continue
                    key = d.get("content", "")[:80]
                    if key not in seen:
                        seen.add(key)
                        docs.append(d)
                        level_new += 1
                if level_new > 0:
                    idle_levels = 0
                else:
                    idle_levels += 1
                    if idle_levels >= _EARLY_STOP_IDLE_LEVELS:
                        logger.info(
                            f"E2 scan early-stop {motif.get('id','')}@{dynasty}: "
                            f"{idle_levels} consecutive levels with no new docs "
                            f"(collected {len(docs)}/{top_k})"
                        )
                        break
            except asyncio.TimeoutError:
                logger.warning(
                    f"E2 scan level TIMEOUT({_RETRIEVE_TIMEOUT_S:.0f}s) [{q_text[:40]}] "
                    f"{motif.get('id','')}@{dynasty} — 跳过该级"
                )
                continue
            except (ValueError, ConnectionError, RuntimeError, OSError, TimeoutError) as e:
                logger.warning(
                    f"E2 scan level failed [{q_text[:40]}] {motif.get('id','')}@{dynasty}: "
                    f"{type(e).__name__}"
                )

        if len(docs) == 0:
            logger.warning(
                f"E2 scan: ALL strategies failed for {motif.get('id','')}@{dynasty} "
                f"(name={motif_name[:30]})"
            )

        # 从检索到的文档中提取摘要（拼接前3条文档的内容）
        excerpts = []
        for d in docs[:3]:
            content = (d.get("content", "") or "")[:500]
            ch_title = d.get("metadata", {}).get("chapter_title", "")
            ch_dynasty = d.get("metadata", {}).get("dynasty", "")
            if ch_dynasty:
                label = f"[{ch_dynasty}] {ch_title}"
            else:
                label = f"[{ch_title}]"
            if content:
                excerpts.append(f"{label}: {content}")
        summary = " | ".join(excerpts) if excerpts else ""

        return {"count": len(docs), "docs": docs, "summary": summary}

    # ====================================================================
    #  B-034: classify_changes (E3)
    # ====================================================================

    async def _classify_changes(self, state: EvolutionWorkbenchState) -> dict:
        """E3: 母题矩阵 -> 变化类型分类（并行化）。

        fast fail: MCP 不可用 -> 使用 fallback 分类。
        单条 classify 失败 -> 该条使用 fallback_classify。
        """
        matrix = state.get("motif_matrix", [])
        dynasties = state.get("dynasties", ["唐", "宋", "明", "清"])

        # 构建 dict 索引: (motif_id, dynasty) -> cell
        cell_index = {(c["motif_id"], c["dynasty"]): c for c in matrix}

        # 保持 motifs 的首次出现顺序
        motifs_seen = list(dict.fromkeys(c.get("motif_id", "") for c in matrix))

        # 收集所有待分类任务
        tasks = []
        task_params = []
        for motif_id in motifs_seen:
            for i in range(len(dynasties) - 1):
                d_from = dynasties[i]
                d_to = dynasties[i + 1]
                cell_from = cell_index.get((motif_id, d_from))
                cell_to = cell_index.get((motif_id, d_to))
                if not cell_from and not cell_to:
                    continue

                text_from = cell_from.get("summary", "") if cell_from else ""
                text_to = cell_to.get("summary", "") if cell_to else ""
                text_from = text_from or f"{d_from}代{cell_from.get('occurrence_count', 0) if cell_from else 0}次出现"
                text_to = text_to or f"{d_to}代{cell_to.get('occurrence_count', 0) if cell_to else 0}次出现"

                task_params.append((motif_id, d_from, d_to, cell_from, cell_to))
                if self.mcp:
                    tasks.append(self.mcp.call_tool("classify_change_type", {
                        "motif_id": motif_id,
                        "dynasty_from": d_from,
                        "dynasty_to": d_to,
                        "text_from": text_from,
                        "text_to": text_to,
                    }))

        # 并行执行所有 MCP 调用，逐条流式推送进度
        # B-156: 开始即推送实时进度
        self._emit_live(build_sse_event(
            SSEEventType.THINKING_STEP,
            step="classify_changes",
            message=f"开始变化分类: 并行处理 {len(task_params)} 处跨朝代变化..."
        ))
        # 母题 id→名 映射，用于进度文案
        _mname_map = {m.get("id", ""): (m.get("name", "") or "")[:14]
                      for m in state.get("motifs", [])}

        # 为每个任务打 idx 标签，as_completed 完成即推送细粒度进度
        async def _classify_one(idx: int, params, coro):
            try:
                r = await coro
                return idx, r, params, None
            except BaseException as e:
                return idx, None, params, e

        tagged_tasks = [
            _classify_one(i, p, t)
            for i, (p, t) in enumerate(zip(task_params, tasks))
        ]

        results_map: dict = {}
        completed = 0
        total_changes = len(task_params)
        for coro in asyncio.as_completed(tagged_tasks) if tagged_tasks else []:
            idx, result, params, exc = await coro
            motif_id, d_from, d_to, cell_from, cell_to = params
            if exc is not None:
                result = exc
            results_map[idx] = result
            completed += 1
            # 细粒度：每条变化分类完成即告知类型，让用户看到"分到第几条、是什么变化"
            if isinstance(result, BaseException):
                _ctype = "异常→兜底"
            else:
                _data = result.get("result", result) if isinstance(result, dict) else {}
                if _data.get("error"):
                    _ctype = "出错→兜底"
                else:
                    _ctype = _data.get("change_type", "?")
            _mlabel = _mname_map.get(motif_id, motif_id)
            self._emit_live(build_sse_event(
                SSEEventType.THINKING_STEP,
                step="classify_changes",
                message=f"分类 {_mlabel} {d_from}→{d_to}: {_ctype}（{completed}/{total_changes}）"
            ))

        # 构建 changes 列表（按原顺序）
        changes = []
        for idx, (motif_id, d_from, d_to, cell_from, cell_to) in enumerate(task_params):
            result = results_map.get(idx)
            if result is not None and not isinstance(result, BaseException):
                data = result.get("result", result) if isinstance(result, dict) else {}
                if data.get("error"):
                    logger.warning("E3 classify MCP error %s %s->%s: %s",
                                   motif_id, d_from, d_to, str(data["error"])[:100])
                    change_data = self._fallback_classify(cell_from, cell_to)
                else:
                    change_data = data
            else:
                change_data = self._fallback_classify(cell_from, cell_to)

            changes.append({
                "motif_id": motif_id,
                "dynasty_from": d_from,
                "dynasty_to": d_to,
                "change_type": change_data.get("change_type", "substitution"),
                "confidence": change_data.get("confidence", "low"),
                "explanation": change_data.get("explanation", ""),
            })

        logger.info(f"E3 DONE: {len(changes)} changes classified")
        return {"changes": changes, "current_phase": "E3_complete",
                "pending_events": [
                    build_sse_event(SSEEventType.THINKING_STEP,
                                    step="classify_changes",
                                    message=f"变化分类完成: 识别到 {len(changes)} 处跨朝代变化")
                ]}

    @staticmethod
    def _fallback_classify(cell_from, cell_to):
        count_from = cell_from.get("occurrence_count", 0) if cell_from else 0
        count_to = cell_to.get("occurrence_count", 0) if cell_to else 0
        if count_from == 0 and count_to > 0:
            return {"change_type": "emergence", "confidence": "medium", "explanation": ""}
        if count_from > 0 and count_to == 0:
            return {"change_type": "disappearance", "confidence": "medium", "explanation": ""}
        return {"change_type": "substitution", "confidence": "low", "explanation": "fallback classification"}

    # ====================================================================
    #  B-035: infer_phylogeny (E4)
    # ====================================================================

    async def _infer_phylogeny(self, state: EvolutionWorkbenchState) -> dict:
        """E4: 母题矩阵 -> 谱系推断。

        fast fail: MCP 不可用 -> 直接调用 LLM 生成谱系。
        LLM 也失败 -> 空谱系，不阻塞后续。
        """
        matrix = state.get("motif_matrix", [])
        dynasties = state.get("dynasties", [])
        changes = state.get("changes", [])
        motifs = state.get("motifs", [])

        # B-156: 开始即推送实时进度
        self._emit_live(build_sse_event(
            SSEEventType.THINKING_STEP,
            step="infer_phylogeny",
            message=f"正在基于 {len(matrix)} 个矩阵单元与 {len(changes)} 处变化推断文本谱系..."
        ))

        phylogeny = None

        # B-152: MCP 传瘦身矩阵 — cell["docs"] 携带全文（20 docs × 2000 字 × 24 格），
        # 曾致 prompt 达 25.8 万 token。谱系推断只需要计数与摘要。
        slim_matrix = [
            {k: cell.get(k) for k in ("motif_id", "dynasty", "occurrence_count", "summary")}
            for cell in matrix
        ]

        # 优先通过 MCP 工具调用
        if self.mcp:
            result = await self.mcp.call_tool("infer_phylogeny", {
                "motif_matrix": {"matrix": slim_matrix, "dynasties": dynasties},
            })
            data = result.get("result", result)
            if not data.get("error"):
                phylogeny = {"nodes": data.get("nodes", []), "edges": data.get("edges", [])}
            else:
                logger.warning("E4: infer_phylogeny MCP 工具返回错误: %s", str(data["error"])[:200])

        # MCP 不可用或失败时，直接调用 LLM 作为 fallback
        if phylogeny is None:
            logger.info("E4: MCP 不可用，使用 LLM fallback 生成谱系")
            try:
                phylogeny = await self._llm_infer_phylogeny(matrix, dynasties, changes, motifs)
            except _LLM_FATAL as e:
                logger.error(f"E4 LLM fallback 也失败: {e}")
                phylogeny = {"nodes": [], "edges": []}

        logger.info(f"E4 DONE: phylogeny nodes={len(phylogeny.get('nodes', []))} edges={len(phylogeny.get('edges', []))}")
        return {
            "phylogeny": phylogeny, "current_phase": "E4_complete",
            "pending_events": [
                build_sse_event(SSEEventType.THINKING_STEP,
                                step="infer_phylogeny",
                                message=f"谱系推断完成: {len(phylogeny.get('nodes', []))} 个节点, {len(phylogeny.get('edges', []))} 条传承边"),
                build_sse_event(SSEEventType.PHYLOGENY_DATA, **phylogeny)
            ]
        }

    async def _llm_infer_phylogeny(self, matrix: list, dynasties: list,
                                   changes: list, motifs: list) -> dict:
        """LLM 直接推断谱系（MCP fallback）。"""
        # 构建精简的矩阵摘要（不含完整文档内容，避免 token 爆炸）
        matrix_summary = []
        for cell in matrix:
            summary = (cell.get("summary", "") or "")[:200]
            matrix_summary.append(
                f"{cell.get('motif_id', '')}@{cell.get('dynasty', '')}: "
                f"出现{cell.get('occurrence_count', 0)}次"
                + (f" ({summary})" if summary else "")
            )

        change_summary = []
        for c in changes[:10]:
            change_summary.append(
                f"{c.get('motif_id', '')}: {c.get('dynasty_from', '')}->{c.get('dynasty_to', '')} "
                f"{c.get('change_type', '')}"
            )

        motif_names = [f"{m.get('id', '')}:{m.get('name', '')}" for m in motifs]

        system = (
            "你是文本谱系学专家。基于母题在朝代之间的共享/转移模式，推断文本的传承关系。\n"
            "每个节点需要指定 type 字段为以下三者之一:\n"
            "- extant: 现存文本（有明确文献依据的章节）\n"
            "- missing: 失传环节（推测存在但文本已失传的中间环节）\n"
            "- external_influence: 外部影响（非文本传承，来自社会/文化的外部因素）\n\n"
            "只返回 JSON: {\"nodes\": [{\"id\": \"...\", \"label\": \"...\", \"era\": \"...\", "
            "\"type\": \"extant|missing|external_influence\"}], "
            "\"edges\": [{\"from\": \"...\", \"to\": \"...\", \"style\": \"solid|dashed|wavy\"}]}"
        )
        user = (
            f"母题: {', '.join(motif_names[:6])}\n"
            f"朝代: {', '.join(dynasties)}\n"
            f"矩阵摘要:\n" + "\n".join(matrix_summary[:15]) + "\n"
            f"变化:\n" + "\n".join(change_summary)
        )

        from ..config import settings as app_settings
        model = app_settings.effective_model
        with llm_call_context("infer_phylogeny", "infer_phylogeny_llm",
                              upstream="classify_changes",
                              downstream="correlate_external"):
            resp = await self.llm.generate(system, user, model=model, max_tokens=2048,
                                           temperature=0.5, thinking_disabled=True)
        self._llm_call_count += 1
        content = resp.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]

        try:
            data = parse_llm_json(content)
        except json.JSONDecodeError as e:
            logger.error(f"E4 LLM fallback JSON 解析失败: {e}")
            return {"nodes": [], "edges": []}

        _e4_nodes = data.get("nodes", [])
        _e4_edges = data.get("edges", [])
        # B-156: LLM 输出预览 — 让用户看到谱系推断结果
        self._emit_live(build_sse_event(
            SSEEventType.THINKING_STEP,
            step="infer_phylogeny",
            message=f"E4 LLM 输出: 推断出 {len(_e4_nodes)} 个谱系节点、{len(_e4_edges)} 条传承关系"
        ))
        return {
            "nodes": _e4_nodes,
            "edges": [{"from": e.get("from", e.get("source", "")),
                       "to": e.get("to", e.get("target", "")),
                       "style": e.get("style", "solid")}
                      for e in _e4_edges],
        }

    # ====================================================================
    #  B-036: correlate_external (E5)
    # ====================================================================

    async def _correlate_external(self, state: EvolutionWorkbenchState) -> dict:
        """E5: 变化 -> 外部历史关联。

        fast fail: Token 预算超 80% -> 跳过（B-051）。
        MCP 不可用 -> 空 correlations，不阻塞。
        """
        changes = state.get("changes", [])

        # B-156: 开始即推送实时进度
        self._emit_live(build_sse_event(
            SSEEventType.THINKING_STEP,
            step="correlate_external",
            message=f"正在关联 {min(len(changes), 5)} 处变化与历史/社会背景..."
        ))

        if not self.mcp:
            return {"correlations": [], "current_phase": "E5_complete",
                    "pending_events": [
                        build_sse_event(SSEEventType.THINKING_STEP,
                                        step="correlate_external",
                                        message="跳过外部关联（MCP 不可用）")
                    ]}

        change_descriptions = [
            f"{c.get('motif_id', '')}: {c.get('dynasty_from', '')} -> {c.get('dynasty_to', '')}"
            for c in changes[:5]
        ]
        if not change_descriptions:
            return {"correlations": [], "current_phase": "E5_complete",
                    "pending_events": [
                        build_sse_event(SSEEventType.THINKING_STEP,
                                        step="correlate_external",
                                        message="外部关联: 无可用变化记录")
                    ]}

        dynasty = changes[0].get("dynasty_to", "清") if changes else "清"

        result = await self.mcp.call_tool("correlate_external", {
            "change_descriptions": change_descriptions,
            "dynasty": dynasty,
        })
        data = result.get("result", result)
        if data.get("error"):
            logger.warning("E5: correlate_external MCP 工具返回错误: %s", str(data["error"])[:200])
            return {"correlations": [], "current_phase": "E5_complete",
                    "pending_events": [
                        build_sse_event(SSEEventType.THINKING_STEP,
                                        step="correlate_external",
                                        message="外部关联失败，跳过此步骤")
                    ]}
        _e5_corrs = data.get("correlations", [])
        # B-156: 输出预览 — 让用户看到关联到了多少历史背景
        self._emit_live(build_sse_event(
            SSEEventType.THINKING_STEP,
            step="correlate_external",
            message=f"E5 输出: 关联到 {len(_e5_corrs)} 条历史/社会背景"
        ))
        logger.info(f"E5 DONE: {len(data.get('correlations', []))} external correlations")
        return {"correlations": data.get("correlations", []), "current_phase": "E5_complete",
                "pending_events": [
                    build_sse_event(SSEEventType.THINKING_STEP,
                                    step="correlate_external",
                                    message=f"外部关联完成: 找到 {len(data.get('correlations', []))} 条历史背景关联")
                ]}

    # ====================================================================
    #  B-037: formulate_hypothesis (E6)
    # ====================================================================

    async def _formulate_hypothesis(self, state: EvolutionWorkbenchState) -> dict:
        """E6: 分析结果 -> 2-4 条候选假设。

        fast fail: LLM JSON 解析失败 -> 使用 fallback 假设。
        LLM 不可用 -> 向上传播 _LLM_FATAL。

        reexplore 时包含上一轮假设，要求 LLM 生成不同的假设方向。
        """
        target = state.get("tracking_target", "")
        matrix = state.get("motif_matrix", [])
        changes = state.get("changes", [])
        previous_hypotheses = state.get("previous_hypotheses") or []
        is_reexplore = len(previous_hypotheses) > 0

        system = (
            "你是白蛇传演化分析专家。基于母题矩阵和变化分类，生成 2-4 条关于文本演变的假设。"
            "每条假设包含 statement、confidence (1-5) 和 reasoning。"
            "只返回 JSON: {\"hypotheses\": [{\"id\": \"H1\", \"statement\": \"...\", \"confidence\": 3, \"reasoning\": \"...\"}]}"
        )

        if is_reexplore:
            prev_text = "\n".join(
                f"- [{h.get('id', '')}] {h.get('statement', '')}"
                for h in previous_hypotheses
            )
            system += (
                f"\n\n重要：以下是上一轮已生成的假设，请你务必生成与这些完全不同的新假设方向，"
                f"从不同角度、不同朝代、不同母题出发进行推断:\n{prev_text}"
            )

        matrix_summary = ""
        for cell in matrix[:10]:
            summary = (cell.get("summary", "") or "")[:150]
            matrix_summary += (
                f"[{cell.get('motif_id', '')}@{cell.get('dynasty', '')}] "
                f"出现{cell.get('occurrence_count', 0)}次"
                + (f": {summary}" if summary else "") + "\n"
            )

        change_summary = ""
        for c in changes[:5]:
            change_summary += f"[{c.get('motif_id', '')}] {c.get('dynasty_from', '')}->{c.get('dynasty_to', '')}: {c.get('change_type', '')}\n"

        user = f"追踪目标: {target}\n\n母题矩阵:\n{matrix_summary}\n\n变化分类:\n{change_summary}"

        from ..config import settings as app_settings
        model = app_settings.effective_model
        temperature = 0.8 if is_reexplore else 0.6
        # B-156: E6 开始即推送，让用户知道正在生成假设
        _e6_label = "重新生成不同方向假设" if is_reexplore else "生成候选假设"
        self._emit_live(build_sse_event(
            SSEEventType.THINKING_STEP,
            step="formulate_hypothesis",
            message=f"E6 提出假设: 基于 {len(matrix)} 格矩阵 + {len(changes)} 处变化 {_e6_label}（LLM 分析中）..."
        ))
        try:
            with llm_call_context("formulate_hypothesis", "formulate_hypothesis",
                                  upstream="correlate_external",
                                  downstream="select_hypothesis"):
                resp = await self.llm.generate(system, user, model=model, max_tokens=2048,
                                               temperature=temperature, thinking_disabled=True)
            content = resp.content.strip()
            self._llm_call_count += 1
            # B-156: LLM 输出预览 — 让用户看到 LLM 生成了什么
            _e6_preview = (content or "")[:80].replace("\n", " ")
            self._emit_live(build_sse_event(
                SSEEventType.THINKING_STEP,
                step="formulate_hypothesis",
                message=f"E6 LLM 输出: {_e6_preview}..."
            ))
        except _LLM_FATAL as e:
            logger.error(f"E6 LLM 调用失败，使用 fallback 假设: {e}")
            hypotheses = [
                {"id": "H1", "statement": f"{target}的叙事功能在朝代间发生了显著变化", "confidence": 3, "reasoning": ""},
                {"id": "H2", "statement": f"{target}的象征意义从唐代到清代经历了多次重新诠释", "confidence": 2, "reasoning": ""},
            ]
            return {
                "candidate_hypotheses": hypotheses, "current_phase": "E6_complete",
                "pending_events": [
                    build_sse_event(SSEEventType.THINKING_STEP,
                                    step="formulate_hypothesis",
                                    message=_hypothesis_summary(target, hypotheses, fallback=True)),
                    build_sse_event(SSEEventType.EVOLUTION_INTERRUPT,
                                    interruptId="select_hypothesis",
                                    interruptType="select_hypothesis",
                                    threadId=state.get("thread_id", ""),
                                    data={
                                        "candidateHypotheses": hypotheses,
                                        "allowMultiSelect": True,
                                        "phase": "E6_complete",
                                    })
                ]
            }

        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]

        try:
            data = parse_llm_json(content)
            hypotheses = data.get("hypotheses", [])
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"E6 LLM JSON 异常（降级到 fallback）: {e}")
            hypotheses = []

        if len(hypotheses) < 2:
            # reexplore 时 fallback 也应避免与上一轮完全相同
            if is_reexplore:
                hypotheses = [
                    {"id": "H1", "statement": f"{target}的文本表现可能存在未被注意的逆向传播路径", "confidence": 3, "reasoning": ""},
                    {"id": "H2", "statement": f"{target}在不同文体（志怪、话本、传奇、戏曲）中的叙事差异反映了媒介对内容的塑造", "confidence": 2, "reasoning": ""},
                ]
            else:
                hypotheses = [
                    {"id": "H1", "statement": f"{target}的叙事功能在朝代间发生了显著变化", "confidence": 3, "reasoning": ""},
                    {"id": "H2", "statement": f"{target}的象征意义从唐代到清代经历了多次重新诠释", "confidence": 2, "reasoning": ""},
                ]

        logger.info(f"E6 DONE: {len(hypotheses)} candidate hypotheses")
        return {
            "candidate_hypotheses": hypotheses, "current_phase": "E6_complete",
            "pending_events": [
                build_sse_event(SSEEventType.THINKING_STEP,
                                step="formulate_hypothesis",
                                message=_hypothesis_summary(target, hypotheses, fallback=False)),
                build_sse_event(SSEEventType.EVOLUTION_INTERRUPT,
                                interruptId="select_hypothesis",
                                interruptType="select_hypothesis",
                                threadId=state.get("thread_id", ""),
                                data={
                                    "candidateHypotheses": hypotheses,
                                    "allowMultiSelect": True,
                                    "phase": "E6_complete",
                                })
            ]
        }

    # ====================================================================
    #  B-038: select_hypothesis (Interrupt #2 handler)
    # ====================================================================

    async def _select_hypothesis(self, state: EvolutionWorkbenchState) -> dict:
        selected_ids = state.get("selected_hypothesis_ids", [])
        candidates = state.get("candidate_hypotheses", [])
        if not selected_ids and candidates:
            selected_ids = [candidates[0].get("id", "H1")]
            logger.warning(
                "E6 select: 未收到用户选择的假设，自动选择第一个: %s (共%d个候选)",
                selected_ids[0], len(candidates)
            )
        elif not selected_ids:
            logger.warning("E6 select: 无候选假设且无用户选择")
        return {
            "selected_hypothesis_ids": selected_ids,
            "current_phase": "E6_selected",
            "pending_events": [
                build_sse_event(SSEEventType.STATUS,
                                message=f"已选择 {len(selected_ids)} 条假设，开始生成验证主张...")
            ]
        }

    # ====================================================================
    #  B-039a: prepare_claims (E7 前半 — 生成主张 + 状态通知)
    # ====================================================================

    async def _prepare_claims(self, state: EvolutionWorkbenchState) -> dict:
        """E7a: 并行生成子主张，发送状态通知告知用户即将验证的数量。

        并行调用 _generate_claims 减少串行等待时间。
        """
        hypotheses = state.get("candidate_hypotheses", [])
        selected_ids = state.get("selected_hypothesis_ids", [])
        target = state.get("tracking_target", "")

        selected = [h for h in hypotheses if h.get("id") in selected_ids] if selected_ids else hypotheses[:1]

        if not selected:
            logger.error("E7a: 无法匹配任何候选假设到 selected_ids=%s", selected_ids)
            return {
                "claims": [],
                "claim_index": 0,
                "current_phase": "E7_testing",
                "pending_events": [
                    build_sse_event(SSEEventType.STATUS,
                                    message="未能生成主张：没有匹配的候选假设")
                ]
            }

        # 并行生成所有选中假设的子主张
        claim_lists = await asyncio.gather(*[
            self._generate_claims(h.get("statement", ""), target)
            for h in selected
        ], return_exceptions=True)

        claims = []
        for h, sub_claims in zip(selected, claim_lists):
            if isinstance(sub_claims, BaseException):
                logger.warning(f"generate_claims failed for {h.get('id')}: {sub_claims}")
                sub_claims = [h.get("statement", "")]
            for i, sc in enumerate(sub_claims):
                claims.append({
                    "claim_id": f"{h.get('id', 'H')}_c{i + 1}",
                    "statement": sc,
                    "evidence_pro": [],
                    "evidence_con": [],
                    "verdict": "pending",
                    "confidence": "medium",
                    "gap_detected": False,
                })

        total = len(claims)
        # 列出主张预览，让用户看到即将验证什么
        claim_preview = "；".join(
            c.get("statement", "")[:24] for c in claims[:4] if c.get("statement")
        )
        if total > 4:
            claim_preview += " 等"
        return {
            "claims": claims,
            "claim_index": 0,
            "current_phase": "E7_testing",
            "pending_events": [
                build_sse_event(SSEEventType.THINKING_STEP,
                                step="test_claims",
                                message=f"已生成 {total} 条可检验主张：{claim_preview}。开始逐条检索证据..."),
                build_sse_event(SSEEventType.STATUS,
                                message=f"已生成 {total} 条主张，开始检索证据...")
            ]
        }

    # ====================================================================
    #  B-039r: test_claims_retrieve (E7 前半 — 并行检索所有证据)
    # ====================================================================

    async def _test_claims_retrieve(self, state: EvolutionWorkbenchState) -> dict:
        """E7b-1: 并行检索所有主张的正反证据。

        所有 retriever 调用并发执行，快速返回检索结果。
        完成后发送状态通知，让用户看到进度。
        """
        claims = state.get("claims", [])
        target = state.get("tracking_target", "")

        if not claims:
            return {
                "claims": [], "current_phase": "E7_testing",
                "pending_events": [
                    build_sse_event(SSEEventType.STATUS, message="无主张需要验证")
                ]
            }

        total = len(claims)

        async def retrieve_for_claim(claim: dict) -> dict:
            """检索主张相关证据 — 按朝代分别检索，确保唐/宋/明/清全覆盖。

            核心问题：唐代仅 7 个 chunk（1.3%），单次语义检索几乎不可能排进 top-20。
            改为对每个朝代单独检索 top_k=5，保证每朝至少 5 条证据进入候选池。
            研究文献仅作补充。
            """
            statement = claim.get("statement", "")
            target = state.get("tracking_target", "")

            # B-152: claims 里混入英文概念词（emergence/transvaluation 等）会稀释
            # 检索查询，只保留 CJK 片段（trace 实锤首轮 0 证据的次因）
            query_text = re.sub(r"[A-Za-z0-9]+", " ", statement).strip()
            if not query_text:
                query_text = statement

            # 按朝代分别检索主文献，确保覆盖均匀
            _seen_keys: set[str] = set()
            all_primary_docs: list[dict] = []

            for dynasty in ("唐", "宋", "明", "清"):
                try:
                    batch = await self._filtered_retrieve(
                        f"{target} {query_text}",
                        filters={"source_type": "primary_literature", "dynasty": dynasty},
                        skip_tagging=True,  # B-155
                    )
                except Exception:
                    batch = []
                if isinstance(batch, BaseException):
                    batch = []
                for d in batch:
                    key = (d.get("content", "") or "")[:80]
                    if key and key not in _seen_keys:
                        _seen_keys.add(key)
                        all_primary_docs.append(d)

            # 补充一轮无朝代过滤的检索，兜底覆盖研究文献或其他朝代
            try:
                extra_batch = await self._filtered_retrieve(
                    f"{target} {query_text}",
                    filters={"source_type": "primary_literature"},
                    top_k=10,
                    skip_tagging=True,  # B-155
                )
            except Exception:
                extra_batch = []
            if isinstance(extra_batch, BaseException):
                extra_batch = []
            for d in extra_batch:
                key = (d.get("content", "") or "")[:80]
                if key and key not in _seen_keys:
                    _seen_keys.add(key)
                    all_primary_docs.append(d)

            # 第二轮：补充研究文献
            research_docs = []
            if len(all_primary_docs) < 15:
                try:
                    research_docs = await self._filtered_retrieve(
                        f"{target} {query_text}",
                        filters={"source_type": "research_literature"},
                        skip_tagging=True,  # B-155
                    )
                except Exception:
                    research_docs = []
                if isinstance(research_docs, BaseException):
                    research_docs = []
                for d in research_docs:
                    key = (d.get("content", "") or "")[:80]
                    if key and key not in _seen_keys:
                        _seen_keys.add(key)
                        research_docs.append(d)

            all_docs = all_primary_docs + research_docs

            # ---- P1-3: 反例检索（双向校验） ----
            # 当前只有正向检索（query=主张本身），导致 evidence_pool 偏向正例。
            # 如果语料中有与主张相悖的段落，但语义上不匹配正向 query，它们不会进池。
            # 修复: 用快模生成"反方查询"，再检索一轮，结果标 retrieval_side="con"。
            # _judge_claim 的 LLM 已有 support/oppose/neutral 分类能力，会把 con 候选
            # 正确归为 oppose。这里只负责"把反例送进池"，不改判定逻辑。
            con_query = await self._generate_opposing_query(statement, target)
            if con_query and con_query != query_text:
                con_docs: list[dict] = []
                for dynasty in ("唐", "宋", "明", "清"):
                    try:
                        batch = await self._filtered_retrieve(
                            f"{target} {con_query}",
                            filters={"source_type": "primary_literature", "dynasty": dynasty},
                            top_k=3,
                            skip_tagging=True,
                        )
                    except Exception:
                        batch = []
                    if isinstance(batch, BaseException):
                        batch = []
                    for d in batch:
                        key = (d.get("content", "") or "")[:80]
                        if key and key not in _seen_keys:
                            _seen_keys.add(key)
                            con_docs.append(d)
                if con_docs:
                    all_docs.extend(con_docs)
                    logger.info(
                        f"P1-3 反例检索: con_query='{con_query[:40]}' "
                        f"+{len(con_docs)} 条反例候选 → evidence_pool"
                    )

            # 统计各朝代分布
            # B-152f: dynasty 字段存在但值为 None 时 .get 的默认值不生效，
            # None 混入 sorted() 会抛 '<' not supported 导致整个 claim 检索作废
            dynasty_counts: dict[str, int] = {}
            for d in all_docs:
                d_dynasty = d.get("metadata", {}).get("dynasty") or "未知"
                dynasty_counts[d_dynasty] = dynasty_counts.get(d_dynasty, 0) + 1
            logger.info(
                f"E7 retrieve_for_claim: {len(all_primary_docs)} primary "
                f"({', '.join(f'{k}:{v}' for k, v in sorted(dynasty_counts.items()) if k != '未知')}) "
                f"+ {len(research_docs)} research "
                f"= {len(all_docs)} total for statement={statement[:60]}..."
            )

            # B-152f: metadata 字段可能为 None（研究文献块 dynasty=None 实锤），
            # 统一用 `or 默认值` + isinstance 防御；chunkId 透传供 synthesize 去重用
            # P1-3: 标记 retrieval_side（pro=正向检索 / con=反例检索），供 _judge_claim 参考
            con_ids = set()
            if con_query and 'con_docs' in locals():
                for cd in con_docs:
                    cid = (cd.get("chunk_id", "") or cd.get("metadata", {}).get("chunk_id", ""))
                    if cid:
                        con_ids.add(cid)
            claim["evidence_pool"] = []
            for d in all_docs:
                _meta = d.get("metadata", {}) or {}
                _p = _meta.get("paragraph_index")
                # B-152j: excerpt/excerptFull 分离 — excerpt 供 LLM(2000字截断省token)，
                # excerptFull 存完整 chunk 文本供前端弹窗不截断展示
                _content_full = (d.get("content", "") or "")
                _content = _content_full[:2000]
                _cid = (d.get("chunk_id", "") or _meta.get("chunk_id", "") or "")
                claim["evidence_pool"].append({
                    "text": _content,
                    "chapterNumber": _cn_to_arabic(_meta.get("chapter_number"), _meta.get("source_type") or "primary_literature"),
                    "chapterTitle": _meta.get("chapter_title") or _meta.get("source_title") or _meta.get("title") or (_meta.get("dynasty") or "") or "证据片段",
                    "paragraphIndex": _p if isinstance(_p, int) else -1,
                    "startPos": _meta.get("char_start", -1),
                    "chunkId": _cid,
                    "excerpt": _content,
                    "excerptFull": _content_full,
                    "dynasty": _meta.get("dynasty") or "",
                    "sourceType": _meta.get("source_type") or "primary_literature",
                    "retrievalSide": "con" if _cid in con_ids else "pro",
                })
            return claim

        retrieved_claims = await asyncio.gather(
            *[retrieve_for_claim(c) for c in claims],
            return_exceptions=True,
        )

        # 过滤掉异常结果
        valid_claims = []
        for i, result in enumerate(retrieved_claims):
            if isinstance(result, BaseException):
                logger.warning(f"retrieve_for_claim[{i}] 异常: {result}")
                valid_claims.append(claims[i])  # 保留原始 claim
            else:
                valid_claims.append(result)

        # 汇总每条主张检索到的证据数，让用户看到检索规模
        ev_counts = []
        for c in valid_claims:
            pro = len(c.get("evidence_pro", []) or [])
            con = len(c.get("evidence_con", []) or [])
            ev_counts.append(f"{pro+con}")
        ev_total = sum(int(x) for x in ev_counts if x.isdigit())

        return {
            "claims": valid_claims,
            "current_phase": "E7_testing",
            "pending_events": [
                build_sse_event(SSEEventType.THINKING_STEP,
                                step="test_claims",
                                message=f"证据检索完成：共 {total} 条主张，检索到 {ev_total} 条候选证据，开始AI逐条判定..."),
                build_sse_event(SSEEventType.STATUS,
                                message=f"证据检索完成 ({total}条主张)，正在进行AI判定...")
            ]
        }

    # ====================================================================
    #  B-039x: test_one_claim (逐条判定，条件循环，每次推送进度)
    # ====================================================================

    async def _test_one_claim(self, state: EvolutionWorkbenchState) -> dict:
        """逐条验证主张，每判定一条推送进度事件。

        支持跳过已判定 claim：只处理 verdict 为 "pending" 或 "pending_retest" 的 claim。
        """
        claims = state.get("claims", [])
        claim_index = state.get("claim_index", 0)
        total = len(claims)

        if not claims:
            return {
                "claims": claims,
                "test_iteration": state.get("test_iteration", 0) + 1,
                "current_phase": "E7_complete",
                "pending_events": [
                    build_sse_event(SSEEventType.EVIDENCE_DATA, claims=claims,
                                    message="无主张需要验证")
                ]
            }

        # 跳过已确定的 claim，找到下一个待判定的
        while claim_index < total:
            v = claims[claim_index].get("verdict", "")
            if v in ("pending", "pending_retest"):
                break
            claim_index += 1

        if claim_index >= total:
            test_iter = state.get("test_iteration", 0) + 1
            all_supported = sum(1 for c in claims if c.get("verdict") == "supported")
            all_insufficient = sum(1 for c in claims if c.get("verdict") == "insufficient")
            all_refuted = sum(1 for c in claims if c.get("verdict") == "refuted")

            events = [
                build_sse_event(SSEEventType.EVIDENCE_DATA, claims=claims,
                                message=f"判定完成 ({total}/{total}): "
                                        f"{all_supported} 支持, {all_insufficient} 不足, {all_refuted} 否定")
            ]

            if (all_refuted > 0 or all_insufficient > 0) and test_iter < 3:
                problematic = [c for c in claims if c.get("verdict") in ("insufficient", "refuted")]
                events.append(
                    build_sse_event(SSEEventType.EVOLUTION_INTERRUPT,
                                    interruptId="decide_next_action",
                                    interruptType="decide_next_action",
                                    threadId=state.get("thread_id", ""),
                                    data={
                                        "summary": f"已验证{total}条主张: {all_supported}条支持, {all_insufficient}条证据不足, {all_refuted}条被否定",
                                        "status": {"total": total, "supported": all_supported,
                                                   "insufficient": all_insufficient, "refuted": all_refuted},
                                        "problematicClaims": problematic[:5],
                                        "options": [
                                            {"id": "investigate", "label": "深入调查",
                                             "description": "对证据不足的主张扩大检索范围，尝试补充证据",
                                             "action": "investigate"},
                                            {"id": "reexplore", "label": "重新探索",
                                             "description": "放弃当前假设方向，重新扫描母题矩阵",
                                             "action": "reexplore"},
                                            {"id": "accept", "label": "接受不确定性",
                                             "description": "在最终报告中保留不确定性标注，进入综合报告",
                                             "action": "accept"},
                                            {"id": "change_hypothesis", "label": "换假设",
                                             "description": "从剩余的备选假设中另选一条进行验证",
                                             "action": "change_hypothesis"},
                                        ],
                                        "phase": "E7_complete",
                                    })
                )

            return {
                "claims": claims,
                "claim_index": total,
                "test_iteration": test_iter,
                "current_phase": "E7_complete",
                "pending_events": events
            }

        claim = claims[claim_index]
        statement = claim.get("statement", "")
        t0 = time.time()

        await self._judge_claim(claim)
        elapsed = time.time() - t0
        logger.info(f"[E7] claim {claim_index+1}/{total} judged in {elapsed:.1f}s: "
                    f"verdict={claim.get('verdict', '?')} statement={statement[:60]}...")

        next_index = claim_index + 1
        done = next_index >= total

        so_far_supported = sum(1 for c in claims[:next_index] if c.get("verdict") == "supported")
        so_far_insufficient = sum(1 for c in claims[:next_index] if c.get("verdict") == "insufficient")
        so_far_refuted = sum(1 for c in claims[:next_index] if c.get("verdict") == "refuted")

        if done:
            test_iter = state.get("test_iteration", 0) + 1
            all_supported = sum(1 for c in claims if c.get("verdict") == "supported")
            all_insufficient = sum(1 for c in claims if c.get("verdict") == "insufficient")
            all_refuted = sum(1 for c in claims if c.get("verdict") == "refuted")

            events = [
                build_sse_event(SSEEventType.EVIDENCE_DATA,
                                claims=claims,
                                message=f"判定完成 ({total}/{total}): "
                                        f"{all_supported} 支持, {all_insufficient} 不足, {all_refuted} 否定")
            ]

            if (all_refuted > 0 or all_insufficient > 0) and test_iter < 3:
                problematic = [c for c in claims if c.get("verdict") in ("insufficient", "refuted")]
                events.append(
                    build_sse_event(SSEEventType.EVOLUTION_INTERRUPT,
                                    interruptId="decide_next_action",
                                    interruptType="decide_next_action",
                                    threadId=state.get("thread_id", ""),
                                    data={
                                        "summary": f"已验证{total}条主张: {all_supported}条支持, {all_insufficient}条证据不足, {all_refuted}条被否定",
                                        "status": {"total": total, "supported": all_supported,
                                                   "insufficient": all_insufficient, "refuted": all_refuted},
                                        "problematicClaims": problematic[:5],
                                        "options": [
                                            {"id": "investigate", "label": "深入调查",
                                             "description": "对证据不足的主张扩大检索范围，尝试补充证据",
                                             "action": "investigate"},
                                            {"id": "reexplore", "label": "重新探索",
                                             "description": "放弃当前假设方向，重新扫描母题矩阵",
                                             "action": "reexplore"},
                                            {"id": "accept", "label": "接受不确定性",
                                             "description": "在最终报告中保留不确定性标注，进入综合报告",
                                             "action": "accept"},
                                            {"id": "change_hypothesis", "label": "换假设",
                                             "description": "从剩余的备选假设中另选一条进行验证",
                                             "action": "change_hypothesis"},
                                        ],
                                        "phase": "E7_complete",
                                    })
                )

            return {
                "claims": claims,
                "claim_index": next_index,
                "test_iteration": test_iter,
                "current_phase": "E7_complete",
                "pending_events": events
            }
        else:
            # 本条刚判定的结论，让用户看到具体每条的判定结果
            verdict_label = {"supported": "支持", "insufficient": "证据不足",
                              "refuted": "否定", "pending": "待定"}.get(claim.get("verdict", ""), claim.get("verdict", ""))
            return {
                "claims": claims,
                "claim_index": next_index,
                "current_phase": "E7_testing",
                "pending_events": [
                    build_sse_event(SSEEventType.THINKING_STEP,
                                    step="test_claims",
                                    message=f"第 {claim_index+1}/{total} 条判定：{verdict_label} —— {statement[:30]}"),
                    build_sse_event(SSEEventType.STATUS,
                                    message=f"主张判定进度: {next_index}/{total} "
                                            f"({so_far_supported}支持 {so_far_insufficient}不足 {so_far_refuted}否定)")
                ]
            }

    async def _generate_opposing_query(self, statement: str, target: str) -> str:
        """P1-3: 用快模生成"反方查询"——用于检索与主张相悖的证据。

        核心问题: 正向检索 query=主张本身，只会召回语义相似的段（偏正例）。
        如果语料中有"白蛇保持妖性"段落，它不会匹配"白蛇趋于人性化"的正向 query。
        解决: 让 LLM 生成一个"如果主张不成立，原文会怎么写"的查询。
        fast fail: LLM 异常 → 返回空串（跳过反例检索，不阻塞主流程）。
        """
        from ..config import settings as app_settings
        fast_model = app_settings.effective_fast_model
        system = (
            "你的任务是为学术主张生成一个'反方检索查询'——即如果该主张不成立或存在反例，"
            "原文中可能出现的描述。只返回检索关键词组（不要完整句子，10-20字），用于向量检索。\n"
            "示例: 主张'白蛇在明代趋于人性化' → 反方查询'白蛇妖性保持 妖怪本性 非人化'\n"
            "示例: 主张'清代白蛇形象更加复杂' → 反方查询'白蛇形象单一 平面化 简单'\n"
            "只返回关键词组，不要解释。"
        )
        user = f"主张: {statement}\n追踪目标: {target}"
        try:
            with llm_call_context("test_claims_retrieve", "generate_opposing_query",
                                  upstream="retrieve_for_claim",
                                  downstream="filtered_retrieve_con"):
                resp = await self.llm.generate(system, user, model=fast_model,
                                               max_tokens=64, temperature=0.3,
                                               thinking_disabled=True)
            query = resp.content.strip().strip('"').strip()
            # 简单校验: 不能是原主张本身、不能太短
            if query and len(query) >= 4 and query != statement:
                return query
            return ""
        except Exception as e:
            logger.warning(f"_generate_opposing_query 异常（降级跳过）: {e}")
            return ""

    async def _generate_claims(self, hypothesis: str, target: str) -> list[str]:
        """为假设生成 2 条可检验的主张。

        fast fail: LLM JSON 异常 -> 直接用原始假设作为主张。
        """
        from ..config import settings as app_settings
        fast_model = app_settings.effective_fast_model
        system = "你是学术假设检验专家。为一个假设生成 2 条具体的、可检验的子主张。只返回 JSON: {\"claims\": [\"主张1\", \"主张2\"]}"
        user = f"假设: {hypothesis}\n追踪目标: {target}"
        with llm_call_context("prepare_claims", "generate_claims",
                              upstream="select_hypothesis",
                              downstream="test_claims_retrieve"):
            resp = await self.llm.generate(system, user, model=fast_model, max_tokens=256,
                                           temperature=0.3, thinking_disabled=True)
        content = resp.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            data = parse_llm_json(content)
            return data.get("claims", [hypothesis])
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"_generate_claims JSON 异常（降级）: {e}")
            return [hypothesis]


    async def _judge_claim(self, claim: dict):
        """LLM 分类证据正反立场 + 判定主张 verdict。

        先对每条证据标注 support/oppose/neutral，再给出综合 verdict。
        首次判定使用 evidence_pool，重测时从 evidence_pro+evidence_con 重建证据池。
        fast fail: LLM JSON 异常 -> 所有证据归入 pro，verdict=insufficient。
        """
        statement = claim.get("statement", "")

        # 重建证据池: 优先用 evidence_pool（首次），否则从 pro+con 合并（重测）
        evidence_pool = claim.get("evidence_pool")
        if not evidence_pool:
            evidence_pool = list(claim.get("evidence_pro", [])) + list(claim.get("evidence_con", []))
            # 去重（按 text 前 100 字符）
            seen = set()
            deduped = []
            for ev in evidence_pool:
                key = ev.get("text", ev.get("excerpt", ""))[:100]
                if key and key not in seen:
                    seen.add(key)
                    deduped.append(ev)
            evidence_pool = deduped

        if not evidence_pool:
            claim["verdict"] = "insufficient"
            claim["confidence"] = "low"
            claim["gap_detected"] = True
            claim["explanation"] = "未检索到相关证据"
            claim["evidence_pro"] = []
            claim["evidence_con"] = []
            claim.pop("evidence_pool", None)
            return

        # 构建带编号的证据列表
        evidence_items = []
        for i, ev in enumerate(evidence_pool):
            evidence_items.append(f"[{i}] {ev['text'][:500]}")
        evidence_text = "\n".join(evidence_items)

        from ..config import settings as app_settings
        fast_model = app_settings.effective_fast_model
        last_verdict = claim.get("_last_verdict", "")
        last_explanation = claim.get("_last_explanation", "")
        if last_verdict:
            system = (
                "这是对该主张的第二次深入评估。上一次判定为「{0}」，理由：{1}。\n"
                "现在补充了一批扩大检索范围后找到的新证据（可能包含之前未检索到的文献）。\n"
                "请仔细审查新证据中是否有支持或反对的信号——即使是间接相关也值得关注。\n"
                "对每条证据判断立场（support=支持主张/oppose=反对主张/neutral=无关），然后给出综合 verdict。\n"
                "只返回 JSON: {{\"verdict\":\"supported|insufficient|refuted\",\"confidence\":\"high|medium|low\","
                "\"gap_detected\":true/false,\"pro_indices\":[0,2],\"con_indices\":[1],"
                "\"explanation\":\"综合说明(中文,一句话)\"}}"
            ).format(last_verdict, last_explanation[:200] if last_explanation else "无")
        else:
            system = (
                "评估学术主张。对每条证据判断立场（support=支持主张/oppose=反对主张/neutral=无关），然后给出综合 verdict。\n"
                "只返回 JSON: {\"verdict\":\"supported|insufficient|refuted\",\"confidence\":\"high|medium|low\","
                "\"gap_detected\":true/false,\"pro_indices\":[0,2],\"con_indices\":[1],\"explanation\":\"综合说明(中文,一句话)\"}"
            )
        user = f"主张: {statement}\n\n证据列表:\n{evidence_text}"

        with llm_call_context("test_one_claim", "judge_claim",
                              upstream="test_claims_retrieve",
                              downstream="_route_after_one_claim"):
            resp = await self.llm.generate(system, user, model=fast_model, max_tokens=512,
                                           temperature=0.2, thinking_disabled=True)
        content = resp.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]

        try:
            data = parse_llm_json(content)
            claim["verdict"] = data.get("verdict", "insufficient")
            claim["confidence"] = data.get("confidence", "low")
            claim["gap_detected"] = data.get("gap_detected", False)
            claim["explanation"] = data.get("explanation", "")
            pro_indices = data.get("pro_indices", [])
            con_indices = data.get("con_indices", [])
            # 按 LLM 分类拆分证据
            claim["evidence_pro"] = [
                evidence_pool[i] for i in pro_indices
                if isinstance(i, int) and 0 <= i < len(evidence_pool)
            ]
            claim["evidence_con"] = [
                evidence_pool[i] for i in con_indices
                if isinstance(i, int) and 0 <= i < len(evidence_pool)
            ]
            # 未分类的证据默认归入 pro（保守处理）
            classified = set(pro_indices) | set(con_indices)
            for i, ev in enumerate(evidence_pool):
                if i not in classified:
                    claim["evidence_pro"].append(ev)
            # 清理临时字段
            claim.pop("evidence_pool", None)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"judge_claim LLM JSON 异常: {e}")
            claim["verdict"] = "insufficient"
            claim["confidence"] = "low"
            claim["gap_detected"] = False
            claim["explanation"] = f"证据分类失败: {str(e)[:80]}"
            claim["evidence_pro"] = evidence_pool
            claim["evidence_con"] = []
            claim.pop("evidence_pool", None)
        # 清理深入调查的临时上下文字段
        claim.pop("_last_verdict", None)
        claim.pop("_last_explanation", None)

    # ====================================================================
    #  B-041: decide_next_action (Interrupt #3 handler)
    # ====================================================================

    async def _decide_next_action(self, state: EvolutionWorkbenchState) -> dict:
        """Interrupt #3 恢复后的处理：读取用户在决策面板中的选择。

        中断事件已由 _test_one_claim 在进入 interrupt_before 之前发送。
        当用户选择 change_hypothesis 时，额外发送 select_hypothesis 中断事件
        （因为 select_hypothesis 是 interrupt_before，事件必须在到达前发出）。
        """
        raw_decision = state.get("decision")
        if not raw_decision:
            raw_decision = state.get("user_decision")
        if not raw_decision:
            logger.warning("E7 decision: decision 为空，回退到 'accept'")
            raw_decision = "accept"

        decision = raw_decision.strip() if isinstance(raw_decision, str) else str(raw_decision)
        valid_decisions = {"investigate", "reexplore", "accept", "change_hypothesis"}
        if decision not in valid_decisions:
            logger.warning(f"E7 decision: 未知 decision '{decision}'，回退到 'accept'")
            decision = "accept"

        logger.info(f"E7 decision: user chose '{decision}', proceeding to route")

        # B-156: 针对用户选择推送差异化进度文案，让前端明确知道后续要做什么
        _decision_msgs = {
            "accept": "用户选择接受不确定性，正在综合各主张与证据，生成最终分析报告...",
            "investigate": "用户选择深入调查，对证据不足的主张扩大检索范围，补充证据...",
            "reexplore": "用户选择重新探索，回到矩阵扫描以发现新的证据模式...",
        }
        _msg = _decision_msgs.get(decision)
        if _msg:
            self._emit_live(build_sse_event(
                SSEEventType.THINKING_STEP,
                step="decide_next_action",
                message=_msg
            ))

        result = {
            "needs_decision": False,
            "decision": decision,
            "current_phase": "E7_decided",
            "_from_deep_investigation": False,
        }

        # 换假设或重新探索时重置 test_iteration（新分析路径）
        if decision in ("change_hypothesis", "reexplore"):
            result["test_iteration"] = 0
            logger.info("E7 decision: reset test_iteration=0 for new analysis path")

        # 重新探索时保存上一轮假设并递增计数，以便 E6 生成不同假设
        if decision == "reexplore":
            result["previous_hypotheses"] = state.get("candidate_hypotheses", [])
            result["reexplore_count"] = state.get("reexplore_count", 0) + 1
            logger.info(
                "E7 decision: reexplore — saved %d previous hypotheses, reexplore_count=%d",
                len(result["previous_hypotheses"]), result["reexplore_count"]
            )

        # 换假设时需要发送 select_hypothesis 中断事件
        # 因为 select_hypothesis 是 interrupt_before，事件必须在到达该节点前发出
        if decision == "change_hypothesis":
            candidates = state.get("candidate_hypotheses", [])
            result["pending_events"] = [
                build_sse_event(SSEEventType.EVOLUTION_INTERRUPT,
                                interruptId="select_hypothesis",
                                interruptType="select_hypothesis",
                                threadId=state.get("thread_id", ""),
                                data={
                                    "candidateHypotheses": candidates,
                                    "allowMultiSelect": True,
                                    "phase": "E7_decided",
                                    "source": "change_hypothesis",
                                })
            ]

        return result

    # ====================================================================
    #  B-043: reflect_and_re_search
    # ====================================================================

    async def _reflect_and_re_search(self, state: EvolutionWorkbenchState) -> dict:
        """深入调查：对 verdict=="insufficient" 的 claim 进行多角度扩大检索。

        策略（与首次检索不同）：
        1. LLM 将 claim statement 改写为 2-3 个适合古籍检索的关键词查询
        2. 每个查询以 top_k=30、无 source_type/dynasty 过滤进行检索
        3. 严格去重：不追加与已有 evidence 重复的文档
        4. 存储上一轮判定上下文供 _judge_claim 做 gap-aware 重判
        5. 所有 insufficient claims 都处理（不限制数量）

        fast fail: LLM 查询生成失败 -> 回退到原始 statement 查询。
        单次检索失败 -> 跳过该查询，继续下一个。
        """
        claims = state.get("claims", [])
        target = state.get("tracking_target", "")
        insufficient = [(i, c) for i, c in enumerate(claims)
                        if c.get("verdict") == "insufficient"]

        if not insufficient:
            logger.info("reflect: 无 insufficient claims，跳过补充检索，强制进入 synthesize")
            return {"current_phase": "E7_reflected", "test_iteration": 3}

        from ..config import settings as _cfg
        fast_model = _cfg.effective_fast_model

        # B-152d: 并行化 — 此前 6 claim × 3 查询双层串行，18 次检索耗时 ~3m39s（占全程 45%）。
        # 现在 claim 间并行、claim 内查询并行，全局 Semaphore(4) 限流保护 ChromaDB/嵌入器
        _sem = asyncio.Semaphore(4)

        # 进度事件累积（节点返回时一次性 flush，让用户看到 reflect 内部发生了什么）
        reflect_events: list = [build_sse_event(
            SSEEventType.THINKING_STEP,
            step="reflect",
            message=f"深入调查: 对 {len(insufficient)} 条证据不足的主张扩大检索范围...",
        )]

        async def _retrieve_limited(query: str) -> list:
            async with _sem:
                # B-155/硬约束: reflect 是程序化批量检索，必须跳过 LLM 打标
                # （旧实现直接 self.retriever.retrieve() 不传 skip_tagging，18 次检索
                #  每次触发 ~16s 的 query_tagging LLM 调用，是"扩大搜索"慢的根因）
                # 同时加 12s 检索超时，慢检索快速失败跳过
                try:
                    return await asyncio.wait_for(
                        self._filtered_retrieve(query, filters=None, top_k=30, skip_tagging=True),
                        timeout=12,
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"reflect: 检索超时(12s) 跳过 [{query[:40]}]")
                    return []

        async def _reflect_one_claim(idx: int, claim: dict) -> bool:
            """处理单个 insufficient claim，返回是否标记为 pending_retest。"""
            statement = claim.get("statement", "")

            # Step 1: LLM 生成多样化古籍检索查询
            alt_queries = []
            try:
                query_system = (
                    "你是古籍文献检索专家。给定一条学术主张，生成 2-3 个能在古籍原文（文言文/古白话）中"
                    "实际匹配到的搜索关键词组合。用空格分隔关键词。关键词必须具体、可能在古文中出现，"
                    "不要现代学术术语。每个查询占一行，只输出查询。"
                )
                query_user = f"主张: {statement}\n追踪目标: {target}\n\n古籍搜索关键词（每行一个查询）:"
                with llm_call_context("reflect_and_re_search", "reflect_research",
                                      upstream="decide_next_action",
                                      downstream="test_one_claim"):
                    resp = await self.llm.generate(query_system, query_user,
                                                   model=fast_model, max_tokens=150, temperature=0.6,
                                                   thinking_disabled=True)
                raw_queries = resp.content.strip().split("\n")
                alt_queries = [q.strip() for q in raw_queries if q.strip()]
                logger.info(
                    f"reflect: LLM generated {len(alt_queries)} queries for claim[{idx}] "
                    f"statement={statement[:60]}..."
                )
                _ev = build_sse_event(
                    SSEEventType.THINKING_STEP,
                    step="reflect",
                    message=f"深入调查 主张{idx + 1}/{len(insufficient)}: LLM 生成 {len(alt_queries)} 个古籍检索词，开始扩大检索...",
                )
                reflect_events.append(_ev)
                self._emit_live(_ev)  # B-156: 实时推送，避免3.5分钟无反馈
            except Exception as e:
                logger.warning(f"reflect: query generation failed for claim[{idx}]: {e}")

            # 回退：至少用原始 statement + target 查询
            if not alt_queries:
                alt_queries = [f"{target} {statement}"]

            # Step 2: 多轮扩大检索 + 严格去重（并行执行查询）
            seen_keys: set[str] = set()
            for ev_list_key in ("evidence_pro", "evidence_con"):
                for ev in claim.get(ev_list_key, []):
                    key = (ev.get("text", "") or ev.get("excerpt", ""))[:100]
                    if key:
                        seen_keys.add(key)

            new_docs: list[dict] = []
            batches = await asyncio.gather(
                *[_retrieve_limited(q) for q in alt_queries[:3]],
                return_exceptions=True,
            )
            for q, batch in zip(alt_queries[:3], batches):
                if isinstance(batch, BaseException):
                    logger.warning(
                        f"reflect: retrieval failed for claim[{idx}] query[{q[:40]}]: "
                        f"{type(batch).__name__}"
                    )
                    continue
                for d in batch:
                    key = (d.get("content", "") or "")[:100]
                    if key and key not in seen_keys:
                        seen_keys.add(key)
                        new_docs.append(d)

            # Step 3: 追加新证据或标记为确认不充分
            if new_docs:
                logger.info(
                    f"reflect: found {len(new_docs)} new docs for claim[{idx}] "
                    f"({len(alt_queries)} queries, statement={statement[:50]}...)"
                )
                _ev = build_sse_event(
                    SSEEventType.THINKING_STEP,
                    step="reflect",
                    message=f"深入调查 主张{idx + 1}/{len(insufficient)}: 扩大检索找到 {len(new_docs)} 条新证据，标记重测",
                )
                reflect_events.append(_ev)
                self._emit_live(_ev)
                for d in new_docs:
                    _meta = d.get("metadata", {}) or {}
                    _p = _meta.get("paragraph_index")
                    _content_full = (d.get("content", "") or "")
                    _content = _content_full[:2000]
                    claim["evidence_pro"].append({
                        "text": _content,
                        "chapterNumber": _cn_to_arabic(_meta.get("chapter_number"), _meta.get("source_type") or "primary_literature"),
                        "chapterTitle": _meta.get("chapter_title") or _meta.get("source_title") or _meta.get("title") or (_meta.get("dynasty") or "") or "证据片段",
                        "paragraphIndex": _p if isinstance(_p, int) else -1,
                        "startPos": _meta.get("char_start", -1),
                        "chunkId": (d.get("chunk_id", "") or _meta.get("chunk_id", "") or ""),
                        "excerpt": _content,
                        "excerptFull": _content_full,
                        "dynasty": _meta.get("dynasty") or "",
                        "sourceType": _meta.get("source_type") or "primary_literature",
                    })
                claim["_last_verdict"] = claim.get("verdict", "insufficient")
                claim["_last_explanation"] = claim.get("explanation", "")
                claim["verdict"] = "pending_retest"
                return True

            logger.warning(
                f"reflect: no new evidence after deep search for claim[{idx}] "
                f"(statement={statement[:50]}...)"
            )
            reflect_events.append(build_sse_event(
                SSEEventType.THINKING_STEP,
                step="reflect",
                message=f"深入调查 主张{idx + 1}: 扩大检索后仍未找到新证据，确认证据不足",
            ))
            claim["gap_detected"] = True
            existing_exp = claim.get("explanation", "")
            if "深入调查后仍未找到新证据" not in (existing_exp or ""):
                claim["explanation"] = (existing_exp or "") + " [深入调查后仍未找到新证据]"
            return False

        results = await asyncio.gather(
            *[_reflect_one_claim(i, c) for i, c in insufficient],
            return_exceptions=True,
        )
        retest_indices = [i for (i, _c), r in zip(insufficient, results) if r is True]
        _failed = sum(1 for r in results if isinstance(r, BaseException))
        if _failed:
            logger.warning(f"reflect: {_failed} 个 claim 的补充检索任务异常中断")

        if not retest_indices:
            logger.info("reflect: 扩大检索后无可重测 claim，强制进入 synthesize")
            reflect_events.append(build_sse_event(
                SSEEventType.THINKING_STEP,
                step="reflect",
                message="深入调查完成: 各主张均无新证据，进入综合报告",
            ))
            return {"claims": claims, "current_phase": "E7_reflected",
                    "test_iteration": 3, "pending_events": reflect_events}

        logger.info(f"reflect: {len(retest_indices)} 条 claim 标记为 pending_retest: {retest_indices}")
        reflect_events.append(build_sse_event(
            SSEEventType.THINKING_STEP,
            step="reflect",
            message=f"深入调查完成: {len(retest_indices)} 条主张找到新证据，重新检验",
        ))
        return {"claims": claims, "claim_index": retest_indices[0],
                "current_phase": "E7_reflected", "_from_deep_investigation": True,
                "pending_events": reflect_events}

    # ====================================================================
    #  B-044: synthesize (E8)
    # ====================================================================

    async def _synthesize(self, state: EvolutionWorkbenchState) -> dict:
        """E8: 综合所有分析 -> 演化叙述 + 4 个 SSE 事件。

        B-150: 锚点ID+前端取数据模式。LLM只用锚点ID [朝代-chN:M] 引用证据，
        前端按锚点精确O(1)查找 evidence_refs 显示原文。移除所有后处理正则/模糊匹配。

        fast fail: LLM 不可用 -> 向上传播 _LLM_FATAL。
        LLM JSON 异常 -> 生成 fallback 叙述。
        """
        target = state.get("tracking_target", "")
        matrix = state.get("motif_matrix") or []
        phylogeny = state.get("phylogeny") or {}
        claims = state.get("claims") or []

        # === Phase 1: 构建 evidence_refs（直接使用 chunk excerpt，不通过 API 重新解析）===
        # B-141: 移除 API 段落重解析。chunk 的 excerpt 就是 LLM 看到的原文，
        # 直接使用它作为弹窗展示文本，消除数据源不一致导致的段落错配。
        _MAX_EXCERPT_FOR_FRONTEND = 500  # 弹窗展示的最大字符数
        evidence_refs: dict[str, dict] = {}
        src_idx = 0
        _all_evidence_meta: list[dict] = []

        for c in claims:
            for ev_list, tag in [(c.get("evidence_pro", []), "支持"), (c.get("evidence_con", []), "反对")]:
                for ev in ev_list:
                    src_idx += 1
                    src_id = f"s{src_idx}"
                    ch_num_raw = ev.get("chapterNumber", ev.get("chapter_number", ""))
                    ch_num = _cn_to_arabic(ch_num_raw, source_type) if isinstance(ch_num_raw, str) else ch_num_raw
                    ch_title = ev.get("chapterTitle", ev.get("chapter_title", ""))
                    # B-152j: 弹窗不截断 — 优先取 excerptFull(完整 chunk 文本)供前端完整显示；
                    # LLM prompt 侧另行截断（第 1886 行 _MAX_EXCERPT_FOR_LLM）
                    chunk_text = ev.get("excerptFull") or ev.get("excerpt", ev.get("text", "")) or ""

                    # U06: 段落号优先使用 chunker 直接记录的 paragraph_index
                    p_idx = ev.get("paragraphIndex", ev.get("paragraph_index", -1))
                    if not isinstance(p_idx, int) or p_idx < 0:
                        p_idx = -1  # 标记为未知段落

                    dynasty = ev.get("dynasty", "")
                    if not dynasty:
                        dynasty = _infer_dynasty(ch_num)
                    source_type = ev.get("sourceType", ev.get("source_type", "primary_literature"))

                    # 弹窗展示用截断版本，LLM 用完整版本
                    excerpt_short = chunk_text[:_MAX_EXCERPT_FOR_FRONTEND]
                    evidence_refs[src_id] = {
                        "chunkId": ev.get("chunkId", ev.get("chunk_id", "")),
                        "charStart": ev.get("charStart", ev.get("char_start", -1)),
                        "chapterNumber": ch_num,
                        "chapterTitle": ch_title,
                        "paragraphIndex": p_idx,
                        "excerpt": excerpt_short,
                        "excerptFull": chunk_text,
                        "dynasty": dynasty,
                        "sourceType": source_type,
                    }
                    _all_evidence_meta.append({
                        "src_idx": src_idx,
                        "ch_num": ch_num,
                        "ch_title": ch_title,
                        "p_idx": p_idx,
                        "dynasty": dynasty,
                        "source_type": source_type,
                        "tag": tag,
                        "excerpt_len": len(chunk_text),
                        "chunk_id": (ev.get("chunkId", ev.get("chunk_id", "")) or ""),
                        # B-152g: 去重键改为 chunkId —— 语料 paragraph_index 大面积缺失
                        # （ChromaDB 实测 345/365 块为 -1），(章节,段落) 键会把上百个
                        # 不同段落塌缩成一条（实测 53 条证据去重后只剩 4 条）。
                        # chunkId 是 chunk 真实身份；缺失时回退旧键
                        "para_key": (ev.get("chunkId", ev.get("chunk_id", "")) or "")
                                    if (ev.get("chunkId", ev.get("chunk_id", "")) or "")
                                    else ((ch_num, p_idx) if source_type == "primary_literature"
                                          else ("r", ch_title, ch_num, p_idx)),
                        "priority": 0 if source_type == "primary_literature" else 1,
                    })

        # === Phase 2: 智能筛选 + 描述性标签 ===
        # B-143: 弃用数字编号 [ref:N]，改用描述性标签 [朝代-简称:段落]。
        # LLM 天然知道"清代雷峰塔传奇"是哪个朝代，不会像数字ID那样张冠李戴。
        # 每个证据用其朝代+章节简称+段落号作为唯一标识，前端用标签查找 evidence_refs。

        # Step 2a: 段落级去重
        _deduped_by_para: dict[tuple, dict] = {}
        for meta in _all_evidence_meta:
            key = meta["para_key"]
            if key not in _deduped_by_para or meta["excerpt_len"] > _deduped_by_para[key]["excerpt_len"]:
                _deduped_by_para[key] = meta

        # Step 2b: 精选 + 朝代均衡
        _selected_src_ids: set[int] = set()
        _sorted_deduped = sorted(
            _deduped_by_para.values(),
            key=lambda m: (m["priority"], -m["excerpt_len"])
        )
        _primary_items = [m for m in _sorted_deduped if m["priority"] == 0]
        _research_items = [m for m in _sorted_deduped if m["priority"] == 1]

        # 每个朝代最多选 MAX_PER_DYNASTY 条主文献，防止某一朝代证据过度集中
        # B-150: 证据量翻倍 (5->10)，让 LLM 有更全面的原文依据
        _MAX_PER_DYNASTY = 10
        _dynasty_counts: dict[str, int] = {}
        for m in _primary_items:
            dynasty = m.get("dynasty", "")
            if _dynasty_counts.get(dynasty, 0) < _MAX_PER_DYNASTY:
                _selected_src_ids.add(m["src_idx"])
                _dynasty_counts[dynasty] = _dynasty_counts.get(dynasty, 0) + 1

        _max_research = min(len(_research_items), max(len(_primary_items), 8))
        for m in _research_items[:_max_research]:
            _selected_src_ids.add(m["src_idx"])

        # Step 2c: 朝代多样性 — 确保每个朝代至少有1条
        for dynasty in ("唐", "宋", "明", "清", "现当代"):
            dynasty_evidence = [m for m in _deduped_by_para.values() if m["dynasty"] == dynasty]
            if dynasty_evidence and not any(m["src_idx"] in _selected_src_ids for m in dynasty_evidence):
                best = max(dynasty_evidence, key=lambda m: -m["priority"])
                _selected_src_ids.add(best["src_idx"])

        # Step 2d: 生成固定格式标签 [朝代-ch章节号:段落]
        # 使用章节号而非作品简称 — LLM 无法自行缩短固定格式，保证标签唯一且可复制
        # B-152b: 章节号/段落号钳位为非负整数 — 前端正则只认 \d+，
        # 负号标签（如 src-ch-1:119、清-ch4:-1）会解析失败导致 94% 引用无法点击
        def _make_label(meta: dict) -> str:
            """为证据生成固定格式引用标签（保证非负分量）。"""
            dynasty = meta.get("dynasty", "")
            ch_num = meta.get("ch_num", 0)
            p_idx = meta.get("p_idx", 0)
            ch = ch_num if isinstance(ch_num, int) and ch_num > 0 else 0
            p = p_idx if isinstance(p_idx, int) and p_idx >= 0 else 0
            if dynasty and ch_num > 0 and meta.get("source_type") == "primary_literature":
                return f"{dynasty}-ch{ch}:{p}"
            else:
                return f"src-ch{ch}:{p}"

        # B-152b: 唯一槽位分配 — 钳位后可能撞键（如两条清-ch4:0），
        # 后到者顺延到空闲段落号，保证 1 个标签唯一对应 1 条证据
        _used_labels: set[str] = set()

        def _allocate_label(base: str) -> str:
            m = re.match(r"^(.*-ch\d+:)(\d+)$", base)
            if not m:
                return base
            prefix, p = m.group(1), int(m.group(2))
            while base in _used_labels:
                p += 1
                base = f"{prefix}{p}"
            return base

        _ordered_meta = sorted(
            [m for m in _all_evidence_meta if m["src_idx"] in _selected_src_ids],
            key=lambda m: m["src_idx"]
        )
        # Step 2d-2e: 生成锚点ID标签 + 构建证据行（单次循环）
        # B-150: 移除短ID系统。LLM只用完整锚点ID [朝代-chN:M]，前端按标签精确O(1)查找。
        _selected_evidence_refs: dict[str, dict] = {}
        _evidence_labels: list[str] = []
        evidence_lines: list[str] = []
        _MAX_EXCERPT_FOR_LLM = 2000  # B-152j: prompt 侧截断省 token；前端弹窗走 excerptFull 不截断

        # B-151: 构建 shortId → 完整标签映射，用于兜底修正 LLM 偶尔写的旧格式短ID
        _dynasty_counters: dict[str, int] = {}
        _short_to_label: dict[str, str] = {}

        for meta in _ordered_meta:
            label = _allocate_label(_make_label(meta))
            _used_labels.add(label)
            orig_ref = evidence_refs[f"s{meta['src_idx']}"]
            dynasty = orig_ref.get("dynasty", "") or "研究"

            # 生成旧格式短ID作为兜底: 唐1, 唐2, 宋1, ...
            _dynasty_counters[dynasty] = _dynasty_counters.get(dynasty, 0) + 1
            short_id = f"{dynasty}{_dynasty_counters[dynasty]}"
            _short_to_label[short_id] = label

            _selected_evidence_refs[label] = {
                "chapterNumber": orig_ref["chapterNumber"],
                "chapterTitle": orig_ref["chapterTitle"],
                "paragraphIndex": orig_ref["paragraphIndex"],
                "excerpt": orig_ref["excerpt"],
                "excerptFull": orig_ref.get("excerptFull", orig_ref["excerpt"]),
                "dynasty": orig_ref["dynasty"],
                "sourceType": orig_ref["sourceType"],
                "label": label,
            }
            _evidence_labels.append(label)

            # 证据行: [唐-ch1:3] [主文献] [支持] 李黄 段落3: 原文...
            excerpt_llm = orig_ref.get("excerptFull", orig_ref["excerpt"])[:_MAX_EXCERPT_FOR_LLM]
            src_label = "[主文献]" if orig_ref.get("sourceType") == "primary_literature" else "[研究文献]"
            anchor_tag = f"[{label}]"
            evidence_lines.append(
                f"{anchor_tag} {src_label} [{meta['tag']}] "
                f"{orig_ref['chapterTitle']} 段落{orig_ref['paragraphIndex']}: {excerpt_llm}"
            )

        _evidence_text = "\n".join(evidence_lines)
        logger.info(
            f"_synthesize evidence selection: {len(_all_evidence_meta)} total -> "
            f"{len(_deduped_by_para)} deduped -> {len(_selected_src_ids)} selected for LLM "
            f"({len(_evidence_text)} chars)"
        )

        system = (
            "你是白蛇传演化分析专家。基于以下分析结果撰写综合叙述。\n\n"
            "## 报告结构\n"
            "1. 总体趋势\n"
            "2. 按母题逐条分析（每个母题独立一节）\n"
            "3. 谱系关系解读\n"
            "4. 待确认问题\n\n"
            "## 引用方法\n"
            "证据列表里每行以标签开头，格式 [朝代-chN:M]。这个格式绝对不能自行修改，必须原样精确复制。\n\n"
            "标签含义（以本项目文献为例）：\n"
            "  朝代 — 唐/宋/明/清/现当代\n"
            "  chN  — 章节。ch1=唐·李黄，ch2=宋·西湖三塔记，ch3=明·白娘子永镇雷峰塔，\n"
            "          ch4=清·方成培雷峰塔传奇，ch5=清·玉山堂主人雷峰塔传奇\n"
            "  :M   — 该章节内的段落号\n\n"
            "例如 [唐-ch1:3] = 唐代·李黄·第3段。\n"
            "另一类标签形如 [src-chN:M] —— src 表示该证据为研究文献或元数据不全"
            "（无朝代/章节号，N/M 均为非负编号）。这类标签同样原样引用，不要试图改写成朝代格式。\n"
            "母题矩阵里的方括号（如 [M1@唐]）不是引用标签，禁止把矩阵片段当作证据引用。\n"
            "引用格式（系统会逐字校验，校验失败的引用连同标签一起删除，宁缺毋滥）：\n"
            "  先放一句该证据的原文原句（用中文弯引号\u201c\u201d包裹，15-60字，必须逐字摘自该标签"
            "对应证据的原文，禁止改写、增删字、拼接两处原文），紧跟标签本身。\n"
            "  正确示例：…\u201c吾当命法海下凡，委曲收服妖邪\u201d[清-ch4:5]\n"
            "  错误示例（引文不是证据原文，会被删除）：\u201c法海奉旨收妖\u201d[清-ch4:5]\n"
            "[主文献] 优先级高于 [研究文献]。引用分散：不同论点引用不同证据，同一标签全篇不超过3次。\n\n"
            "## 输出要求\n"
            "直接输出报告正文，第一行必须是 `# ` 开头的主标题。\n"
            "禁止输出任何写作前的思考、计划、证据整理、自我确认或对约束的复述文字。\n\n"
            "## 关键约束\n"
            "你只能使用上述证据列表中实际出现的标签。禁止编造、猜测或修改任何标签。\n"
            "编造不存在的标签是严重错误，会导致引用无法验证。\n\n"
            "## 信息来源规则\n"
            "  a) 引文 + 标签 的组合中，引文必须逐字出自该标签对应证据 —— 系统会逐字校验。\n"
            "  b) 凭记忆写出的原文（如《警世通言》句子）若不在证据列表中，句末标注（模型训练数据），\n"
            "     且禁止配任何标签 —— 配了标签属于双重编造，是最严重的错误。\n"
            "  c) 一般性的背景陈述（如时代背景、文学史常识）无需标注，可直接写。\n"
        )

        matrix_text = ""
        for cell in matrix:
            summary = (cell.get("summary", "") or "")[:120]
            line = f"- {cell.get('motif_name', '') or cell.get('motif_id', '')} @ {cell.get('dynasty', '')}: {cell.get('occurrence_count', 0)}次"
            if summary:
                line += f" — {summary}"
            matrix_text += line + "\n"

        claims_text = ""
        verdict_labels = {"supported": "已证实", "insufficient": "证据不足", "refuted": "已否定", "pending": "待验证", "pending_retest": "待重测"}
        for c in claims:
            v = c.get('verdict', '')
            label = verdict_labels.get(v, v)
            claims_text += f"- 【{label}】{c.get('statement', '')}\n"

        user = (
            f"追踪目标: {target}\n\n"
            f"母题矩阵:\n{matrix_text}\n"
            f"主张验证:\n{claims_text}\n"
            f"证据原文（每行以引用标签开头，引用时按 system 规定的格式：弯引号原句+标签）:\n"
            + _evidence_text
        )

        sc = state.get("system_context", "")
        if sc:
            system += f"\n\n{sc}"

        from ..config import settings as app_settings
        model = app_settings.effective_model
        # B-156: 综合报告是单次长生成（4096 token），可能耗时较久——
        # 开始即 live 推送，让用户实时知道在生成报告（不再延迟到节点 return）
        self._emit_live(build_sse_event(
            SSEEventType.THINKING_STEP,
            step="synthesize",
            message=f"E8 综合呈现: 正在综合 {len(claims)} 条主张与母题矩阵证据，生成演化分析报告（LLM 长文本生成中，可能耗时较久）..."
        ))
        _synthesize_start_events: list[dict] = []
        with llm_call_context("synthesize", "synthesize",
                              upstream="test_one_claim",
                              downstream="present"):
            resp = await self.llm.generate(system, user, model=model, max_tokens=4096,
                                           temperature=0.3, thinking_disabled=True)
        narrative = resp.content
        # B-156: LLM 输出预览 — 让用户看到报告已开始生成、大致内容
        _e8_preview = (narrative or "")[:60].replace("\n", " ").strip()
        if _e8_preview:
            self._emit_live(build_sse_event(
                SSEEventType.THINKING_STEP,
                step="synthesize",
                message=f"E8 LLM 输出: 报告已生成约 {len(narrative)} 字，开头: {_e8_preview}..."
            ))
        # B-152: 剥离写在正文前的思考/规划前奏（trace 实锤：模型先输出"先整理…/注意…"
        # 等大段自言自语再写报告）。以第一个 Markdown 标题行作为正文起点。
        _h = re.search(r"^#{1,3} .+$", narrative, re.M)
        if _h and _h.start() > 0:
            logger.info(f"_synthesize 剥离思考前奏: {_h.start()} chars")
            narrative = narrative[_h.start():]
        # B-151: 将 LLM 偶尔写的旧格式短ID [明3][宋2] 转换为完整锚点 [明-ch3:1]
        # 这是兜底修正，不是主要逻辑——正确情况下 LLM 应按 prompt 直接写完整锚点
        if _short_to_label:
            _short_fixed = 0
            _short_unknown: set[str] = set()
            def _replace_short_id(m: re.Match) -> str:
                nonlocal _short_fixed
                sid = m.group(1)
                full = _short_to_label.get(sid)
                if full:
                    _short_fixed += 1
                    return f"[{full}]"
                _short_unknown.add(sid)
                return m.group(0)
            narrative = re.sub(
                r'\[(\w{1,4}\d+)\]', _replace_short_id, narrative
            )
            if _short_fixed > 0:
                logger.info(
                    f"_synthesize short-id fix: 修正 {_short_fixed} 个旧格式短ID → 完整锚点"
                )
            # B-151 fix: 移除 LLM 编造的旧格式短ID（不在 _short_to_label 映射中的）
            if _short_unknown:
                for sid in sorted(_short_unknown, key=len, reverse=True):
                    narrative = narrative.replace(f"[{sid}]", "")
                logger.info(
                    f"_synthesize short-id strip: removed {len(_short_unknown)} hallucinated short IDs: "
                    f"{sorted(_short_unknown)[:10]}"
                )
        # B-150: 锚点验证 — 检查 LLM 引用的锚点是否都在 evidence_refs 中存在
        # B-152: ch-?\d+ 兼容 src-ch-1（研究文献/元数据不全，章节号为 -1）——
        # 旧正则匹配不了负号，导致合法引用被判为"未引用"（12/13 误报）
        _anchor_pattern = re.compile(r'\[([\w一-鿿]{1,4}-ch-?\d+:\d+)\]')
        narrative_anchors = set(_anchor_pattern.findall(narrative))
        ref_labels = set(_evidence_labels)
        missing = narrative_anchors - ref_labels
        if missing:
            logger.warning(
                f"_synthesize anchor validation: {len(missing)} anchors not found in evidence_refs: "
                f"{sorted(missing)[:10]}"
            )
            # B-150 fix: 从 narrative 中移除 LLM 编造的锚点标签，前端不再看到假标签
            # B-152m: 连同编造标签前面的引号引文一起移除 —— 只删标签会残留
            # "引文"。 悬空引文，看起来像没出处的引用。整对删除。
            fake_total = len(_anchor_pattern.findall(narrative))
            for fake_label in sorted(missing, key=len, reverse=True):
                # 先尝试整对删除（引文 + 间隔 + 标签）
                pair_re = re.compile(
                    r'[\u201c"]([^"\u201c\u201d]{0,300})[\u201d"][^[]{0,60}'
                    + re.escape(f"[{fake_label}]")
                )
                narrative, n_pair = pair_re.subn("", narrative)
                if n_pair == 0:
                    narrative = narrative.replace(f"[{fake_label}]", "")
            _rate = len(missing) / max(fake_total, 1) * 100
            logger.warning(
                f"_synthesize anchor fix: removed {len(missing)}/{fake_total} hallucinated anchors "
                f"(编造率 {_rate:.0f}%) — 若持续 >15% 需检查 prompt 约束"
            )
        extra = ref_labels - narrative_anchors
        if extra:
            logger.info(
                f"_synthesize anchor validation: {len(extra)} evidence refs not cited in narrative"
            )

        # B-152p/B-158: 引文全面校验（治本版，确定性后处理，无额外 LLM 调用）
        # 旧版只校验"弯引号+60字符内跟标签"的引文，三类编造引文漏网：
        #   (a) 无标签引文 —— 完全绕过校验，用户看到"引文在原文中找不到"的主因；
        #   (b) 「」直角引号 —— 旧正则不识别，引文整体漏检；
        #   (c) 标签内空格变体 [清-ch4: 5] —— 标签匹配失败连坐引文。
        # 校验规则：
        #   带标签 → 与该标签证据 excerptFull 逐字比对；失败 → 引文连同标签整对删除
        #   无标签 → 与全部选中证据合并语料比对；命中 → 保留（真实原文，仅缺标签）；
        #            未命中 → 追加（模型训练数据）标记（兑现 system prompt 规则 b：
        #            非证据原文必须标注训练数据来源，与证据标签严格区分）
        import unicodedata as _ud
        def _norm_content(s: str) -> str:
            """NFC 归一化 + 只保留字母数字（与前端 isContentChar 一致）。"""
            s = _ud.normalize("NFC", s)
            return "".join(c for c in s if c.isalnum())

        # (c) 标签空格归一化：[清-ch4: 5] / [清 - ch4:5] → [清-ch4:5]
        narrative = re.sub(
            r'\[([\w一-鿿]{1,4})\s*-\s*ch\s*(-?\d+)\s*:\s*(\d+)\]',
            lambda m: f"[{m.group(1)}-ch{m.group(2)}:{m.group(3)}]",
            narrative,
        )

        def _quote_in(full: str, quote: str) -> bool:
            """4 级逐字匹配：精确 / 归一化整串 / 12字前缀 / 15字滑窗。"""
            exc, nq = _norm_content(full), _norm_content(quote)
            return (
                (quote in full)
                or (nq in exc)
                or (len(nq) >= 12 and nq[:12] in exc)
                or (len(nq) >= 15 and any(nq[i:i + 15] in exc for i in range(0, len(nq) - 14, 5)))
            )

        # 全部选中证据的合并语料（无标签引文的比对基准）
        _corpus_norm = "".join(
            _norm_content(r.get("excerptFull") or r.get("excerpt", "") or "")
            for r in _selected_evidence_refs.values()
        )

        # 引文跨度：弯引号/直引号/直角引号，内容 4-300 字
        _quote_span_re = re.compile(r'([\u201c\u300c"])([^\u201c\u201d\u300d"]{4,300})([\u201d\u300d"])')
        # 引文闭号之后 120 字符内的可选标签（允许中间隔少量叙述文字）
        _label_after_re = re.compile(r'[^[]{0,120}\[([\w一-鿿]{1,4}-ch-?\d+:\d+)\]')

        _removed_quotes = 0
        _marked_memory_quotes = 0
        _out: list[str] = []
        _last = 0
        for _m in _quote_span_re.finditer(narrative):
            _out.append(narrative[_last:_m.start()])
            quote = _m.group(2)
            tail = narrative[_m.end():_m.end() + 120]
            lm = _label_after_re.match(tail)
            if lm:
                label = lm.group(1)
                ref = _selected_evidence_refs.get(label)
                full = (ref or {}).get("excerptFull") or (ref or {}).get("excerpt", "") or ""
                if full and _quote_in(full, quote):
                    _out.append(_m.group(0))       # 合法带标签引文 → 原样保留
                    _last = _m.end()
                else:
                    _removed_quotes += 1           # 编造引文 → 引文+间隔+标签整对删除
                    _last = _m.end() + lm.end()
            else:
                if _quote_in(_corpus_norm, quote):
                    _out.append(_m.group(0))       # 真实原文但缺标签 → 保留
                else:
                    _marked_memory_quotes += 1     # 非证据原文 → 标注训练数据来源
                    _out.append(f"{_m.group(1)}{quote}{_m.group(3)}（模型训练数据）")
                _last = _m.end()
        _out.append(narrative[_last:])
        narrative = "".join(_out)
        if _removed_quotes:
            logger.warning(
                f"_synthesize quote validation: removed {_removed_quotes} fabricated tagged quotes "
                f"(引文与标签证据不符，整对删除)"
            )
        if _marked_memory_quotes:
            logger.warning(
                f"_synthesize quote validation: marked {_marked_memory_quotes} untagged quotes "
                f"as （模型训练数据）(引文不在任何证据原文中)"
            )

        # Build structured cells for final emission
        motifs_from_state = state.get("motifs", [])
        if motifs_from_state:
            # 统一用 "ID: 名称" 格式，前端 findCell() 用 split(':') 提取 ID
            motif_names_final = [
                f"{m.get('id', '')}: {m.get('name', '')}"
                for m in motifs_from_state
            ]
        else:
            # Fallback: 从矩阵数据中重建列名（防止 state 中 motifs 意外丢失）
            seen_ids = list(dict.fromkeys(c.get("motif_id", "") for c in matrix))
            motif_names_final = [f"{mid}: (fallback)" for mid in seen_ids if mid]
            logger.warning(f"E8: motifs 为空，从矩阵重建 {len(motif_names_final)} 列")

        dynasties_final = state.get("dynasties") or ["唐", "宋", "明", "清"]
        cells_final = []
        for cell in matrix:
            occ = cell.get("occurrence_count", 0)
            docs = cell.get("documents", [])
            summary = cell.get("summary", "")
            cells_final.append({
                "motifId": cell.get("motif_id", ""),
                "era": cell.get("dynasty", ""),
                "status": "absent" if occ == 0 else "stable",
                "value": _fmt_cell_value(occ, summary),
                "tooltip": summary or None,
                "occurrenceCount": occ,
                "evidence": [
                    {
                        "chapterNumber": _cn_to_arabic(d.get("metadata", {}).get("chapter_number", ""), d.get("metadata", {}).get("source_type", "primary_literature")),
                        "chapterTitle": d.get("metadata", {}).get("chapter_title", ""),
                        "paragraphIndex": d.get("metadata", {}).get("paragraph_index", 0),
                        "excerpt": (d.get("content", "") or "")[:2000],
                        "dynasty": d.get("metadata", {}).get("dynasty", ""),
                    }
                    for d in docs
                ],
            })
        # 构建 pending_events：母题矩阵仅在数据存在时发送，避免空数据覆盖前端已有矩阵
        # 合并生成开始事件 + 完成摘要，让用户看到报告生成的时间线
        _section_count = len(re.findall(r"^#{1,3} ", narrative, re.M))
        _narr_len = len(narrative)
        pending_events: list[dict] = list(_synthesize_start_events) + [
            build_sse_event(SSEEventType.THINKING_STEP,
                            step="synthesize",
                            message=f"综合报告生成完成：约 {_narr_len} 字，{_section_count} 个章节"),
        ]
        if motifs_from_state and matrix:
            pending_events.append(
                build_sse_event(SSEEventType.MOTIF_MATRIX,
                                rows=dynasties_final, columns=motif_names_final, cells=cells_final,
                                isFinal=True)
            )
        else:
            logger.warning(
                f"E8: 跳过空母题矩阵 (motifs_in_state={len(motifs_from_state)}, "
                f"matrix_cells={len(matrix)}, dynasties={dynasties_final})"
            )
        pending_events.extend([
            build_sse_event(SSEEventType.PHYLOGENY_DATA,
                            nodes=(phylogeny or {}).get("nodes", []),
                            edges=(phylogeny or {}).get("edges", []),
                            is_final=True),
            build_sse_event(SSEEventType.EVIDENCE_DATA, claims=claims, is_final=True),
            build_sse_event(SSEEventType.EVOLUTION_NARRATIVE, narrative=narrative, evidence_refs=_selected_evidence_refs),
        ])
        return {
            "evolution_narrative": narrative, "current_phase": "E8_synthesized",
            "evidence_refs": _selected_evidence_refs,
            "pending_events": pending_events,
        }

    # ====================================================================
    #  B-045: present (E8 输出)
    # ====================================================================

    async def _present(self, state: EvolutionWorkbenchState) -> dict:
        target = state.get("tracking_target", "")
        narrative = state.get("evolution_narrative", "")
        # 生成摘要（取前200字，在句号处截断）
        summary = narrative[:200]
        last_period = max(summary.rfind("。"), summary.rfind("？"), summary.rfind("！"))
        if last_period > 50:
            summary = summary[:last_period + 1]
        else:
            summary = summary[:180] + "…"

        return {
            "current_phase": "complete",
            "pending_events": [
                build_sse_event(SSEEventType.WORKFLOW_DONE,
                                workflow="evolution_workbench",
                                summary=f"已完成'{target}'的演化分析",
                                phase="complete"),
                build_sse_event(SSEEventType.EVOLUTION_CHAT_CARD,
                                trackingTarget=target,
                                summary=summary,
                                narrative=narrative,
                                threadId=state.get("thread_id", ""),
                                workflow="evolution"),
            ]
        }

    # ====================================================================
    #  B-047a/b/c: run_stream
    # ====================================================================

    async def run_stream(self, query: str, tracking_target: str = "",
                         target_type: str = "location", thread_id: str = "stream",
                         system_context: str = "") -> AsyncGenerator[dict, None]:
        """v8 流式执行。

        B-047a: 基础事件循环
        B-047b: interrupt 检测
        B-047c: pending_events 刷新 + 错误处理
        """
        initial: EvolutionWorkbenchState = {
            "messages": [],
            "thread_id": thread_id,
            "original_query": query,
            "tracking_target": tracking_target or query,
            "target_type": target_type,
            "system_context": system_context,
            "motifs": [],
            "suggested_additions": [],
            "motif_matrix": [],
            "dynasties": [],
            "changes": [],
            "phylogeny": None,
            "correlations": [],
            "candidate_hypotheses": [],
            "selected_hypothesis_ids": [],
            "claims": [],
            "test_iteration": 0,
            "claim_index": 0,
            "needs_decision": False,
            "decision_options": [],
            "user_decision": None,
            "evolution_narrative": "",
            "current_phase": "",
            "errors": [],
            "pending_events": [],
        }

        # U-02: Yield thinking_plan with 8-step skeleton before execution
        yield build_sse_event(SSEEventType.THINKING_PLAN,
                              total_steps=8,
                              steps=[
                                  {"name": "decompose_motifs", "label": "母题分解"},
                                  {"name": "scan_motif_matrix", "label": "矩阵扫描"},
                                  {"name": "classify_changes", "label": "变化分类"},
                                  {"name": "infer_phylogeny", "label": "谱系推断"},
                                  {"name": "correlate_external", "label": "外部关联"},
                                  {"name": "formulate_hypothesis", "label": "假设生成"},
                                  {"name": "test_claims", "label": "主张验证"},
                                  {"name": "synthesize", "label": "综合报告"},
                              ])

        yield build_sse_event(SSEEventType.STATUS,
                              message=f"启动文脉演化分析: {tracking_target or query}")

        config = {"configurable": {"thread_id": thread_id}} if self._checkpointer else None

        # 启动 LLM trace（全量记录每次 LLM 调用的输入输出）
        trace = Tracer.start_trace(thread_id, query)
        logger.info(f"[LLM_TRACE] trace started: trace_id={trace.trace_id} thread_id={thread_id}")

        logger.info(f"EvolutionWorkbench.run_stream: checkpointer={self._checkpointer is not None} config={config is not None} thread_id={thread_id}")
        logger.info(f"EvolutionWorkbench graph nodes: {[n for n in self._graph.nodes]}")
        # 使用 astream() 执行图。interrupt_before 节点会由 LangGraph 内部处理:
        # state 在中断前自动保存到 checkpointer，astream 生成器自然耗尽。
        # 不能提前 return —— 否则 LangGraph 来不及保存中断状态，resume 会报 404。
        logger.info(f"EvolutionWorkbench: astream mode, config={bool(config)}")
        # B-156: 队列 + drainer 模式 — astream 在后台 task 中消费，事件经队列实时 yield。
        # 节点内可通过 _emit_live() 立即推送进度（如 E2 矩阵扫描每格完成时），
        # 无需等到节点 return。队列无界（put_nowait 不阻塞节点），drainer 消费即 yield。
        self._live_queue = asyncio.Queue()

        async def _drain_astream():
            try:
                async for node_output in self._graph.astream(initial, config):
                    if self._cancel_event.is_set():
                        await self._live_queue.put(
                            {"type": "status", "message": "演化分析已被取消", "phase": "cancelled"})
                        return
                    for node_name, output in node_output.items():
                        if isinstance(output, dict):
                            pending = output.get("pending_events", [])
                            for pe in pending:
                                await self._live_queue.put(pe)
            except asyncio.CancelledError:
                await self._live_queue.put(
                    {"type": "status", "message": "演化分析已被取消", "phase": "cancelled"})
            except _LLM_FATAL as e:
                logger.error(f"EvolutionWorkbench LLM 致命错误: {e}")
                await self._live_queue.put({"type": "error",
                                            "data": {"code": "LLM_FATAL",
                                                     "message": f"AI 服务暂时不可用: {str(e)[:200]}"}})
            except Exception as e:
                # 兜底：捕获 httpx 连接失效(ValueError('no active connection'))等非 _LLM_FATAL 异常，
                # 避免被 Task 吞没导致前端无限卡住（与 resume 同根因防御）。
                logger.error(f"EvolutionWorkbench astream 未预期异常: {type(e).__name__}: {e}", exc_info=True)
                await self._live_queue.put({"type": "error",
                                            "data": {"code": "INTERNAL_ERROR",
                                                     "message": f"演化分析出现未预期错误: {type(e).__name__}: {str(e)[:200]}"}})
            finally:
                summary = trace.llm_summary()
                logger.info(f"[LLM_TRACE] trace summary: {json.dumps(summary, ensure_ascii=False)}")
                await self._live_queue.put(_STREAM_END)

        drainer = asyncio.create_task(_drain_astream())
        try:
            while True:
                item = await self._live_queue.get()
                if item is _STREAM_END:
                    break
                yield item
                if item.get("phase") == "cancelled":
                    break
        finally:
            if not drainer.done():
                drainer.cancel()
            self._live_queue = None
        logger.info(f"EvolutionWorkbench: astream exhausted (interrupt or complete)")

    async def run(self, query: str, tracking_target: str = "",
                  target_type: str = "location", thread_id: str = "default",
                  system_context: str = "") -> dict:
        """非流式执行。"""
        initial: EvolutionWorkbenchState = {
            "messages": [],
            "thread_id": thread_id,
            "original_query": query,
            "tracking_target": tracking_target or query,
            "target_type": target_type,
            "system_context": system_context,
            "motifs": [],
            "suggested_additions": [],
            "motif_matrix": [],
            "dynasties": [],
            "changes": [],
            "phylogeny": None,
            "correlations": [],
            "candidate_hypotheses": [],
            "selected_hypothesis_ids": [],
            "claims": [],
            "test_iteration": 0,
            "claim_index": 0,
            "needs_decision": False,
            "decision_options": [],
            "user_decision": None,
            "evolution_narrative": "",
            "current_phase": "",
            "errors": [],
            "pending_events": [],
        }
        config = {"configurable": {"thread_id": thread_id}} if self._checkpointer and thread_id else None
        return await self._graph.ainvoke(initial, config) if config else await self._graph.ainvoke(initial)

    async def resume_stream(self, resume_data: dict, config: dict,
                            thread_id: str) -> AsyncGenerator[dict, None]:
        """B-093: 从中断点恢复流式执行 — 使用与 run_stream 一致的 astream() API。

        Fast fail: astream() 与 run_stream 使用完全相同的输出格式 ({node_name: output_dict})，
        节点 output 中的 pending_events 被逐条 yield，包括中断事件和工作区驱动事件。
        """
        from langgraph.types import Command
        # LangGraph 1.x: resume 只作为 interrupt() 返回值, update 用于注入状态
        command = Command(resume=resume_data, update=resume_data)

        # 启动新的 LLM trace（resume 段）
        trace = Tracer.start_trace(thread_id, f"resume: {resume_data.get('action', '?')}")
        logger.info(
            f"[LLM_TRACE] resume trace started: trace_id={trace.trace_id} "
            f"thread_id={thread_id} action={resume_data.get('action','?')} "
            f"has_checkpointer={self._checkpointer is not None}"
        )
        node_count = 0
        # B-156: 与 run_stream 一致的队列+drainer 模式，支持节点内 _emit_live() 实时推送
        self._live_queue = asyncio.Queue()

        async def _drain_resume():
            nonlocal node_count
            try:
                async for node_output in self._graph.astream(command, config):
                    if self._cancel_event.is_set():
                        await self._live_queue.put(
                            {"type": "status", "message": "演化分析已被取消", "phase": "cancelled"})
                        return
                    for node_name, output in node_output.items():
                        node_count += 1
                        if isinstance(output, dict):
                            pending = output.get("pending_events", [])
                            logger.info(
                                f"EvolutionWorkbench: resume node #{node_count} {node_name} "
                                f"pending_events={len(pending)} phase={output.get('current_phase','?')}"
                            )
                            for pe in pending:
                                await self._live_queue.put(pe)
                        else:
                            logger.warning(
                                f"EvolutionWorkbench: resume node #{node_count} {node_name} "
                                f"output is not dict: type={type(output).__name__}"
                            )
            except asyncio.CancelledError:
                await self._live_queue.put(
                    {"type": "status", "message": "演化分析已被取消", "phase": "cancelled"})
            except _LLM_FATAL as e:
                logger.error(f"EvolutionWorkbench resume LLM 致命错误: {e}")
                await self._live_queue.put({"type": "error",
                                            "data": {"code": "LLM_FATAL",
                                                     "message": f"AI 服务暂时不可用: {str(e)[:200]}"}})
            except Exception as e:
                # 兜底：捕获 httpx 连接失效(ValueError('no active connection'))等非 _LLM_FATAL 异常，
                # 避免被 Task 吞没导致前端无限卡住（B-096 resume 卡死根因）。
                logger.error(f"EvolutionWorkbench resume 未预期异常: {type(e).__name__}: {e}", exc_info=True)
                await self._live_queue.put({"type": "error",
                                            "data": {"code": "INTERNAL_ERROR",
                                                     "message": f"演化分析出现未预期错误: {type(e).__name__}: {str(e)[:200]}"}})
            finally:
                summary = trace.llm_summary()
                logger.info(f"[LLM_TRACE] resume trace summary: {json.dumps(summary, ensure_ascii=False)}")
                await self._live_queue.put(_STREAM_END)

        drainer = asyncio.create_task(_drain_resume())
        try:
            while True:
                item = await self._live_queue.get()
                if item is _STREAM_END:
                    break
                yield item
                if item.get("phase") == "cancelled":
                    break
        finally:
            if not drainer.done():
                drainer.cancel()
            self._live_queue = None
        logger.info(
            f"EvolutionWorkbench: resume astream exhausted after {node_count} nodes "
            f"(interrupt or complete)"
        )


# ====================================================================
#  E1 fallback: 当 LLM 无法生成母题时使用预置模板
# ====================================================================

_LOCATION_FALLBACK_MOTIFS = {
    "雷峰塔": [
        {"id": "M1", "name": "镇压与囚禁", "type": "invariant",
         "core": "作为镇压妖物的宗教建筑，是叙事中权力与惩罚的空间符号"},
        {"id": "M2", "name": "法海的法力权威", "type": "invariant",
         "core": "法海以雷峰塔为法器，其降妖法力与塔的建筑存在绑定关系"},
        {"id": "M3", "name": "塔的倒塌与解放", "type": "variable",
         "core": "雷峰塔的物理倒塌成为旧秩序瓦解、人性解放的叙事隐喻"},
        {"id": "M4", "name": "爱情与禁锢的对立", "type": "invariant",
         "core": "塔作为分离爱人的物理屏障，象征世俗礼教对自由爱情的压制"},
        {"id": "M5", "name": "雷峰塔的景观意象", "type": "variable",
         "core": "塔的视觉形象（赭色、八角、依山临湖）在不同朝代文本中承载不同的美学意涵"},
    ],
    "西湖": [
        {"id": "M1", "name": "邂逅空间", "type": "invariant",
         "core": "西湖作为男女主人公相遇的地理空间，烟雨朦胧的自然环境烘托浪漫氛围"},
        {"id": "M2", "name": "湖山意象与情感映射", "type": "variable",
         "core": "西湖的晴雨晨昏被文本调用为情节推进的氛围装置，承载人物情感变化"},
        {"id": "M3", "name": "三塔镇妖", "type": "variable",
         "core": "西湖三塔的宗教功能随朝代演变，从镇压妖物的法器变为浪漫传说的背景"},
        {"id": "M4", "name": "断桥借伞", "type": "invariant",
         "core": "断桥作为西湖的子空间，借伞定情成为白蛇传最经典的叙事母题"},
        {"id": "M5", "name": "从风景到情感符号", "type": "variable",
         "core": "西湖从唐代的地理背景逐渐升格为承载回忆与宿命的情感符号"},
    ],
    "白素贞": [
        {"id": "M1", "name": "妖异身份与人性挣扎", "type": "invariant",
         "core": "白素贞的蛇妖身份与她追求人间爱情之间的永恒冲突"},
        {"id": "M2", "name": "修炼与仙缘", "type": "variable",
         "core": "白素贞在青城山/峨眉山修炼千年的身世设定，赋予其近仙的身份合法性"},
        {"id": "M3", "name": "反抗权威", "type": "variable",
         "core": "从温顺服从到水漫金山的激烈反抗，展现角色主体性的觉醒"},
        {"id": "M4", "name": "母性与妻性", "type": "variable",
         "core": "白素贞作为妻子和母亲的世俗角色与妖异本质之间的张力"},
        {"id": "M5", "name": "命运与自由意志", "type": "invariant",
         "core": "白素贞试图以个人意志对抗天命和宗教秩序的悲剧性母题"},
    ],
    "法海": [
        {"id": "M1", "name": "宗教权威的合法性", "type": "invariant",
         "core": "法海作为金山寺禅师，其降妖行为的宗教正当性来源与演变"},
        {"id": "M2", "name": "正义与偏执的界限", "type": "variable",
         "core": "法海的形象从明代的正统守护者转变为清代的偏执反派"},
        {"id": "M3", "name": "佛道合流", "type": "variable",
         "core": "法海兼具佛教禅师身份与龙虎山道教法术来源的复合特征"},
    ],
    "许仙": [
        {"id": "M1", "name": "凡人视角与读者代入", "type": "invariant",
         "core": "许仙作为普通人代表，其恐惧与犹豫是读者理解超自然事件的桥梁"},
        {"id": "M2", "name": "懦弱与成长", "type": "variable",
         "core": "从动摇背叛到最终追随白素贞，许仙的角色弧线体现人性成长"},
    ],
}

_DEFAULT_FALLBACK_MOTIFS = [
    {"id": "M1", "name": "叙事功能的朝代演变", "type": "variable",
     "core": "追踪目标在不同朝代文本中承担的叙事功能和象征意义的变化"},
    {"id": "M2", "name": "身份与合法性", "type": "invariant",
     "core": "追踪目标的身份属性如何在叙事中被定义、质疑或确认"},
    {"id": "M3", "name": "与其他元素的关联网络", "type": "variable",
     "core": "追踪目标与人物/地点/事件之间的共现关系在不同朝代的演变"},
    {"id": "M4", "name": "文化象征的层累", "type": "variable",
     "core": "追踪目标所承载的文化象征意义如何在历史进程中叠加和转变"},
    {"id": "M5", "name": "文本表现的形式特征", "type": "variable",
     "core": "描述追踪目标的语言风格、修辞策略和叙事技巧的朝代差异"},
]


_CN_NUM_MAP = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15,
    "研究": 0,  # 研究文献无章节号，映射到 0
}



def _cn_to_arabic(raw, source_type: str = "primary_literature") -> int:
    """将中文数字章节号转换为阿拉伯数字，用于前端文本跳转。

    U06: 研究文献的 chapter_number 为 "研究"（中文词非数字），不做转换，
    返回 -1 标记为非数字章节。调用方应检查 source_type 决定展示方式。
    """
    if raw is None:
        return 0
    if isinstance(raw, (int, float)):
        return int(raw)
    raw_str = str(raw).strip()
    if raw_str.isdigit():
        return int(raw_str)
    # 研究文献的章节号是 "研究"（中文词），不是数字
    if source_type == "research_literature" and raw_str == "研究":
        return -1  # 标记为非数字章节，调用方应特殊处理
    # 尝试匹配 "一、xxx" 格式
    match = re.match(r'^([一二三四五六七八九十]+)', raw_str)
    key = match.group(1) if match else raw_str
    return _CN_NUM_MAP.get(key, 0)


# 章节号 → 朝代映射（白蛇传文献结构）
_CHAPTER_DYNASTY_MAP: dict[int, str] = {
    1: "唐",
    2: "宋", 3: "宋",
    4: "明", 5: "明", 6: "明",
    7: "清", 8: "现当代",
}


def _infer_dynasty(chapter_number: int) -> str:
    """根据章节号推断所属朝代。未知章节返回空字符串。"""
    return _CHAPTER_DYNASTY_MAP.get(chapter_number, "")


def _resolve_paragraph_index(chunk_text: str, chapter_num: int,
                              full_paragraphs: dict) -> int | None:
    """在真实段落中查找 chunk 文本所属的段落号。

    用 chunk 中间一段文本作为搜索键，在章节的所有段落中做子串匹配。
    避免依赖 chunk_index（滑动窗口编号），直接定位到自然段。
    """
    if not chunk_text or not full_paragraphs:
        return None

    # 规范化空白以匹配 chunker 和段落分割的不同空白处理
    normalized = re.sub(r'\s+', '', chunk_text)
    if len(normalized) < 15:
        return None

    # 取中间一段（~60字），避免 chunk 开头/结尾的不完整词
    mid_start = max(0, len(normalized) // 3)
    key = normalized[mid_start:mid_start + 60]

    if len(key) < 15:
        key = normalized[:80]

    matches = []
    for (ch, pi), para_text in full_paragraphs.items():
        if ch != chapter_num:
            continue
        para_normalized = re.sub(r'\s+', '', para_text)
        if key in para_normalized:
            matches.append(pi)

    if len(matches) == 0:
        return None
    if len(matches) > 1:
        logger.warning(
            f"_resolve_paragraph_index: chunk 匹配到 {len(matches)} 个段落 "
            f"(ch={chapter_num}, paras={matches})，使用第一个"
        )
    return matches[0]


def _resolve_para_by_start_pos(start_pos: int, chapter_text: str) -> int:
    """用 chunk 的 start_pos 元数据计算真实段落号。

    段落分隔符为双换行（与 chunker._clean_text 和 _synthesize 保持一致）。
    start_pos 是 chunk 在 cleaned_text 中的起始字符偏移，由 chunker 在分块时记录。

    算法：累积各段落的字符长度 + 分隔符(\\n\\n = 2 chars)，找到 start_pos 落在
    哪个段落的范围内。若 start_pos 恰好落在分隔符内则归属下一段。
    """
    if not chapter_text or start_pos < 0:
        return 0
    paragraphs = chapter_text.split('\n\n')
    current = 0
    for i, para in enumerate(paragraphs):
        para_len = len(para)
        if current <= start_pos < current + para_len:
            return i
        current += para_len + 2  # +2 为 \\n\\n 分隔符
    # start_pos 超出最后一段末尾 → 归入最后一段
    return max(0, len(paragraphs) - 1)


def _clean_chapter_text(text: str) -> str:
    """与 chunker._clean_text 保持一致的文本清洗。

    统一换行符、合并多余空行、去首尾空白。
    确保通过 node_client 获取的原始文本和 chunker 分块时的 cleaned_text
    在段落边界上对齐，使 start_pos 可正确定位段落。
    """
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


_CELL_VALUE_MAX_LEN = 120  # 母题矩阵单元格摘要最大展示字数 (不含次数前缀)


def _fmt_cell_value(occurrence_count: int, summary: str) -> str | None:
    """格式化母题矩阵单元格展示值。

    B-130: 增大展示长度到 120 字，确保不同朝代的内容差异可见。

    fast fail: 参数类型不匹配直接抛出，不静默兼容。
    始终包含 occurrence_count 前缀，使不同朝代的同母题单元格可区分。
    """
    if occurrence_count == 0:
        return None
    if not summary:
        return f"{occurrence_count}次"
    short = summary[:_CELL_VALUE_MAX_LEN] + "…" if len(summary) > _CELL_VALUE_MAX_LEN else summary
    return f"({occurrence_count}) {short}"


def _build_fallback_motifs(tracking_target: str, target_type: str) -> list[dict]:
    """当 LLM 无法生成母题时，从预置模板中选择。

    优先按 tracking_target 精确匹配，其次按 target_type 匹配，
    最后使用通用默认模板。
    """
    target_lower = tracking_target.strip().lower()
    for key, motifs in _LOCATION_FALLBACK_MOTIFS.items():
        if key in tracking_target or tracking_target in key:
            logger.info(f"E1 fallback: 使用预置模板 '{key}' (匹配 tracking_target={tracking_target})")
            return motifs
    logger.info(f"E1 fallback: 使用通用默认模板 (tracking_target={tracking_target})")
    return _DEFAULT_FALLBACK_MOTIFS


def _hypothesis_summary(target: str, hypotheses: list[dict], fallback: bool) -> str:
    """构建 E6 假设生成进度的摘要消息（含假设预览）。"""
    tag = "（降级到预置假设）" if fallback else ""
    parts = [h.get("statement", "")[:36] for h in hypotheses if h.get("statement")]
    if parts:
        preview = "；".join(parts[:3])
        if len(parts) > 3:
            preview += " 等"
        return f"假设生成完成{tag}：生成 {len(hypotheses)} 条候选假设 —— {preview}"
    return f"假设生成完成{tag}：生成 {len(hypotheses)} 条候选假设"


def _build_e1_result(tracking_target: str, motifs: list[dict],
                     suggested: list[dict], thread_id: str = "", degraded: bool = False) -> dict:
    """构建 E1 阶段的返回 dict（含 interrupt 事件）。

    Args:
        thread_id: 从 state 传入的会话 ID，用于前端 resume 时定位 checkpoint。
                   不得为空字符串，否则 resume 将因查不到 checkpoint 而失败。
    """
    msg = f"已将'{tracking_target}'分解为 {len(motifs)} 个叙事母题"
    if degraded:
        msg += "（使用预置模板）"

    # 列出母题名称，让用户看到具体分解出了什么
    motif_names = [m.get("name", "") for m in motifs if m.get("name")]
    if motif_names:
        preview = "、".join(motif_names[:6])
        if len(motif_names) > 6:
            preview += " 等"
        msg = f"已将'{tracking_target}'分解为 {len(motifs)} 个叙事母题：{preview}"
        if degraded:
            msg += "（使用预置模板）"

    return {
        "motifs": motifs,
        "suggested_additions": suggested,
        "current_phase": "E1_complete",
        "pending_events": [
            build_sse_event(SSEEventType.THINKING_STEP,
                            step="decompose_motifs",
                            message=msg),
            build_sse_event(SSEEventType.EVOLUTION_INTERRUPT,
                            interruptId="confirm_motifs",
                            interruptType="confirm_motifs",
                            threadId=thread_id,
                            data={
                                "trackingTarget": tracking_target,
                                "motifs": motifs,
                                "suggestedAdditions": suggested,
                                "phase": "E1_complete",
                            })
        ]
    }
