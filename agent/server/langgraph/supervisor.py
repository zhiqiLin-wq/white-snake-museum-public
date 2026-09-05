"""Supervisor Graph（总控图）— 意图分类 → 条件路由到专业子图 → 汇总结果。

核心设计（agent-deep-design §3/§5）：
- classify_intent: LLM 分类用户意图为 5 种之一
- conditional_edges: 根据意图路由到对应子图
- 所有子图共享 RAG + LLM + MCP + Skills 基础设施
- checkpointer 传入所有子图，实现跨子图状态持久化
- astream_events() 提供真正的 LangGraph 流式输出

5 个子图：
1. literature_qa  (✅ 已有，在运行) — 通用文献问答
2. location_story (✅ 已激活) — 地标叙事深度导游
3. comparison     (✅ 已激活) — 跨朝代/角色对比分析
4. character      (🆕 占位) — 角色演变分析
5. itinerary      (🆕 占位) — 文旅路线规划
"""
import json
import logging
import time
import uuid
from typing import Optional, AsyncGenerator, Literal

from ..llm.json_utils import parse_llm_json
from ..llm.provider import LLMFatalError, RateLimitError, APITimeoutError
from ..observability.tracer import llm_call_context
from ..dispatch.tool_dispatcher import ToolDispatcher
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List, Dict, Any
from ..api.events import SSEEventType

logger = logging.getLogger(__name__)

# 意图类型 (v6: 5 种, 已移除 character, v11: 新增 annotation, v12: 新增 open_chapter + map_control)
IntentType = Literal["literature_qa", "location_story", "comparison", "itinerary", "evolution", "dynasty_compare", "annotation", "open_chapter", "map_control"]

# 意图中文标签 (v6: 精简, 仅保留活跃意图)
INTENT_LABELS = {
    "literature_qa": "文献问答",
    "location_story": "景观叙事",
    "comparison": "对比分析",
    "itinerary": "路线规划",
    "evolution": "文脉演化",
    "dynasty_compare": "跨朝代对读",
    "annotation": "文本标注",
    "open_chapter": "打开章节",
    "map_control": "地图控制",
}

# B-117: 旧版意图映射 (标记为 deprecated，v8 保留向后兼容)
LEGACY_INTENT_MAP = {
    "planning":   "itinerary",    # deprecated in v8
    "location":   "location_story",  # deprecated in v8
    "story":      "literature_qa",   # deprecated in v8
    "factual":    "literature_qa",   # deprecated in v8
    "analysis":   "literature_qa",   # deprecated in v8
    "character":  "evolution",       # deprecated in v8
}


class SupervisorState(TypedDict, total=False):
    """Supervisor 全局状态 (v6)。"""
    messages: Annotated[list, add_messages]
    intent: str
    intent_confidence: float
    route_reason: str                 # v12: LLM 路由选择理由（审计+前端展示）
    keywords: List[str]
    evolution_target: str             # evolution 追踪目标
    evolution_target_type: str        # location|character|theme|object|motif
    subgraph_result: dict
    conversation_id: str
    thread_id: str
    error: Optional[str]


class SupervisorGraph:
    """总控图：分类 → 路由 → 子图执行 → 输出格式化。"""

    def __init__(
        self,
        llm,
        retriever,
        prompt_registry,
        skill_registry=None,
        node_client=None,
        checkpointer=None,
        mcp_registry=None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.prompts = prompt_registry
        self.skill_registry = skill_registry
        self.node_client = node_client
        self._checkpointer = checkpointer
        self.mcp = mcp_registry
        # 统一工具调度层：收敛 user_id 注入 / tool_call_context / 调用
        self.dispatcher = ToolDispatcher(mcp_registry=self.mcp, retriever=self.retriever)
        self._subgraphs = {}  # 延迟初始化
        self._graph = self._build_graph()
        self._active_workbenches: dict[str, object] = {}

    def cancel_evolution(self, thread_id: str) -> bool:
        """取消指定 thread 的演化分析。返回 True 表示找到并取消了。"""
        from .evolution_workbench import EvolutionWorkbench
        wb = self._active_workbenches.get(thread_id)
        if wb is not None:
            wb.cancel()
            return True
        return False

    # ===== 图结构 =====

    def _build_graph(self):
        workflow = StateGraph(SupervisorState)

        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("literature_qa", self._exec_literature_qa)
        workflow.add_node("location_story", self._exec_location_story)
        workflow.add_node("comparison", self._exec_comparison)
        workflow.add_node("itinerary", self._exec_itinerary)
        workflow.add_node("evolution", self._exec_evolution)
        workflow.add_node("dynasty_compare", self._exec_dynasty_compare)
        workflow.add_node("format_output", self._format_output)

        workflow.set_entry_point("classify_intent")

        # ★ 条件路由：根据意图分发到对应子图 (v6: 已移除 character, v10: 新增 dynasty_compare)
        workflow.add_conditional_edges(
            "classify_intent",
            self._route_by_intent,
            {
                "literature_qa": "literature_qa",
                "location_story": "location_story",
                "comparison": "comparison",
                "itinerary": "itinerary",
                "evolution": "evolution",
                "dynasty_compare": "dynasty_compare",
            }
        )

        # 所有子图 → format_output → END
        for node in ["literature_qa", "location_story", "comparison", "itinerary", "evolution", "dynasty_compare"]:
            workflow.add_edge(node, "format_output")
        workflow.add_edge("format_output", END)

        # v6: Supervisor 自身不使用 checkpointer (避免 AsyncSqliteSaver 兼容性问题)
        # 子图各自通过 run()/run_stream() 的参数独立传入 checkpointer
        return workflow.compile()

    def _route_by_intent(self, state: SupervisorState) -> str:
        """根据意图路由到对应子图 (v6: 5 种意图)。"""
        intent = state.get("intent", "literature_qa")
        valid_intents = {"literature_qa", "location_story", "comparison", "itinerary", "evolution", "dynasty_compare", "annotation", "open_chapter", "map_control"}
        if intent == "annotation":
            return "literature_qa"  # 标注委托给 AgentLoop 的 ReAct 循环（有 annotate_user_request 工具）
        return intent if intent in valid_intents else "literature_qa"

    # ===== 意图分类节点 =====

    async def _classify_intent(self, state: SupervisorState) -> dict:
        """LLM 意图分类 (v6 统一分类器) — 一次 LLM 调用同时输出 intent + evolution_scenario。

        改进点 (v4 → v6):
        - 合并 classifier + evolution_scenario_classify，消除二次 LLM 调用
        - 使用 LEGACY_INTENT_MAP 做旧版意图名→新版意图名映射
        - skill_registry 匹配的 workflow 作为强先验
        - 失败时回退到 literature_qa
        """
        from ..config import settings as app_settings

        messages = state.get("messages", [])
        query = ""
        history_parts = []
        for msg in messages:
            role = ""
            content = ""
            if hasattr(msg, 'type'):
                if msg.type == "human":
                    role = "用户"
                    content = msg.content
                elif msg.type == "ai":
                    role = "助手"
                    content = msg.content[:200]
            elif isinstance(msg, dict):
                r = msg.get("role", "")
                if r in ("user", "human"):
                    role = "用户"
                    content = msg.get("content", "")
                elif r in ("assistant", "ai"):
                    role = "助手"
                    content = msg.get("content", "")[:200]
            if role and content:
                history_parts.append(f"[{role}]: {content}")
                if role == "用户":
                    query = content

        if not query:
            query = state.get("conversation_id", "")

        conversation_history = ""
        if len(history_parts) > 1:
            # v13: token 感知修剪 — 分类器 prompt <~2000 tokens，历史预算 ~1200 tokens
            CLASSIFIER_HISTORY_BUDGET = 1200
            kept_parts = []
            token_used = 0
            for part in reversed(history_parts[:-1]):
                part_tokens = len(part) // 2
                if token_used + part_tokens > CLASSIFIER_HISTORY_BUDGET:
                    break
                kept_parts.insert(0, part)
                token_used += part_tokens
            if kept_parts:
                conversation_history = "\n".join(kept_parts)
                logger.debug(
                    f"[classify_intent] history: total={len(history_parts)-1} "
                    f"kept={len(kept_parts)} est_tokens={token_used}"
                )
            # v13: 溢出监控日志
            if not kept_parts:
                logger.warning(
                    f"[classify_intent] all history pruned: "
                    f"total={len(history_parts)-1} query={query[:80]}"
                )
            elif len(kept_parts) < len(history_parts) - 1:
                logger.info(
                    f"[classify_intent] pruned {len(history_parts)-1 - len(kept_parts)} "
                    f"old msgs, kept {len(kept_parts)}"
                )

        # ★ 检查 skill 匹配结果（强先验：用户显式配置的技能是确定性映射，
        #   不属于规则堆叠，v12 保留）
        skill_hint = None
        if self.skill_registry:
            matched = self.skill_registry.match(query)
            if matched:
                skill_def = self.skill_registry.get(matched[0].get("name", ""))
                if skill_def:
                    skill_hint = skill_def.get("workflow", None)

        try:
            fast_model = app_settings.effective_fast_model
            if self.prompts is None:
                # v12: fallback prompt 与 classifier.yaml 同构的能力卡精简版。
                # 不再堆区分规则（v11 的规则补丁已全部移除），防御由后端护栏承担。
                system = (
                    "你是白蛇传数字博物馆的智能路由助手。根据用户问题，从下方路径能力卡中"
                    "选择最合适的一条工作流路径，并给出选择理由。只返回 JSON。\n"
                    "总原则：1) 先判断答案形态（事实汇总/清单枚举/视图操作/演变解释/旅程规划/界面标注）；"
                    "2) literature_qa 是通用兜底，能力覆盖最广，只有答案形态明确匹配专用路径时才离开；"
                    "3) 重路径有代价，不确定时绝不选 evolution/dynasty_compare。\n"
                    "路径能力卡：\n"
                    "- literature_qa（默认，低-中）：ReAct Agent 可调全部检索/统计/共现/韵文枚举/"
                    "互构分析/地图导航工具，凡查、数、找、列、比、解释都能答，包括跨朝代事实汇总。\n"
                    "- evolution（重路径，高，数分钟+确认中断）：仅当答案必须含'演变的解释'"
                    "（演变模式/成因/谱系/母题结构/假设检验）且明确是跨朝代深度研究诉求。\n"
                    "- location_story（中）：具体地点的故事/历史/文化讲解（纯'定位到/飞到'不算）。\n"
                    "- dynasty_compare（中）：明确指定两个朝代版本或两个章节编号的系统性对读。\n"
                    "- comparison（低）：并排/分栏/左右对照打开视图，不自动分析。\n"
                    "- itinerary（中）：行程规划、攻略、门票等旅行事务。\n"
                    "- open_chapter（极低）：打开/查看/读第X章到工作台，无对比/分析/标注意图。\n"
                    "- map_control（极低）：打开/切换地图朝代图层（'定位到某地点'不算）。\n"
                    "- annotation（低）：对文献创建或删除标注/划线/高亮/标记。\n"
                    "只返回JSON:{\"intent\":\"...\",\"keywords\":[...],\"route_reason\":\"一句话理由\","
                    "\"evolution_target\":\"...\",\"evolution_target_type\":\"location|character|theme|object|motif\"}"
                )
                user = f"查询：{query}\n历史：{conversation_history}"
            else:
                system, user = self.prompts.render("classifier", {
                    "query": query,
                    "conversation_history": conversation_history,
                })
            # thinking_disabled=True: DeepSeek V4 默认思考模式会把小 max_tokens 预算
            # 全部吃在 reasoning 上导致 content 为空（曾致意图分类 100% 失败）。
            # 分类是短输出任务，关思考更快更稳。
            with llm_call_context("supervisor", "classify_intent",
                                  upstream="user_query", downstream="_route_by_intent"):
                resp = await self.llm.generate(system, user, model=fast_model, max_tokens=256, temperature=0.3, thinking_disabled=True)
            data = parse_llm_json(resp.content)
            intent_raw = data.get("intent", "literature_qa")
            route_reason = str(data.get("route_reason", "") or "").strip()

            # ★ v6 意图映射: 旧版名 → 新版名
            mapped_intent = LEGACY_INTENT_MAP.get(intent_raw, intent_raw)

            # 验证映射后的 intent 是否合法（v12: 白名单补全 open_chapter/map_control，
            # v11 新增意图后此处曾滞后，导致这两个意图永远被白名单打死再靠 regex retry 捞回）
            valid = {"literature_qa", "location_story", "comparison", "itinerary", "evolution",
                     "dynasty_compare", "annotation", "open_chapter", "map_control"}
            if mapped_intent not in valid:
                logger.warning(f"意图分类: LLM 输出非法意图 '{intent_raw}' → 回退 literature_qa")
                mapped_intent = "literature_qa"
                route_reason = route_reason or f"非法意图'{intent_raw}'回退"

            # ★ v12 harness 护栏（唯一保留的确定性路由防线）：
            # evolution 是重路径（数分钟+确认中断），LLM 误选代价高。
            # 单点校验：查询与对话历史中无**强演化语义指示** → 降级 literature_qa。
            # 词表只收强信号（演变/演化/变迁/成因等）；"变化/怎么变"是弱信号——
            # "各版本韵文有哪些？前后发生了什么变化"这类枚举对比也含它，
            # 放进来会让护栏形同虚设（测试回归实锤）。第二道防线是 E1 母题确认。
            if mapped_intent == "evolution":
                _EVOLUTION_SEMANTICS = (
                    "演变", "演化", "变迁", "演进", "源流",
                    "谱系", "成因", "发展脉络",
                )
                check_text = query + " " + conversation_history
                if not any(kw in check_text for kw in _EVOLUTION_SEMANTICS):
                    logger.warning(
                        f"意图分类: LLM=evolution → 护栏降级 literature_qa "
                        f"(query 无演化语义指示, query='{query[:60]}')"
                    )
                    mapped_intent = "literature_qa"
                    route_reason = f"护栏拦截：查询无演化语义指示，降级通用问答（LLM 原判 evolution）"

            # ★ skill hint 覆盖（强先验，确定性映射）
            if skill_hint and mapped_intent == "literature_qa":
                mapped_intent = skill_hint
                route_reason = route_reason or f"用户自定义技能先验覆盖为 {skill_hint}"
                logger.info(f"意图分类: LLM={intent_raw} → 被 skill 覆盖为 {skill_hint}")

            logger.info(
                f"意图分类: intent={mapped_intent} (LLM raw={intent_raw}) "
                f"reason='{route_reason[:80]}'"
            )
            return {
                "intent": mapped_intent,
                "intent_confidence": 0.85,
                "keywords": data.get("keywords", []),
                "route_reason": route_reason,
                "evolution_target": data.get("evolution_target", "") if mapped_intent == "evolution" else "",
                "evolution_target_type": data.get("evolution_target_type", "location") if mapped_intent == "evolution" else "",
            }
        except (json.JSONDecodeError, LLMFatalError, RateLimitError, APITimeoutError) as e:
            logger.warning(f"Supervisor 意图分类失败 (model={app_settings.effective_fast_model}): {e}")
            fallback_intent = skill_hint or "literature_qa"
            return {
                "intent": fallback_intent,
                "intent_confidence": 0.0 if not skill_hint else 0.5,
                "keywords": [],
                "route_reason": "LLM 分类失败，降级到默认路径" if not skill_hint else "LLM 分类失败，使用技能先验",
            }

    # ===== 子图执行节点 =====

    async def _exec_by_intent(self, intent: str, state: SupervisorState) -> dict:
        """根据意图调度到对应子图 (非流式)。"""
        exec_map = {
            "literature_qa": self._exec_literature_qa,
            "location_story": self._exec_location_story,
            "comparison": self._exec_comparison,
            "itinerary": self._exec_itinerary,
            "evolution": self._exec_evolution,
            "annotation": self._exec_literature_qa,  # v11: 标注委托给 AgentLoop ReAct
        }
        executor = exec_map.get(intent, self._exec_literature_qa)
        return await executor(state)

    async def _exec_literature_qa(self, state: SupervisorState) -> dict:
        """执行 LiteratureQA 子图。"""
        from .literature_qa import LiteratureQAWorkflow
        wf = LiteratureQAWorkflow(
            self.llm, self.retriever, self.prompts,
            checkpointer=None,  # v6: checkpointer 禁用 (LangGraph 版本兼容性)
        )
        query = self._extract_query(state)
        result = await wf.run(query, thread_id=state.get("thread_id", "default"),
                              system_context=getattr(self, '_request_system_context', ''))
        return {"subgraph_result": {"workflow": "literature_qa", **result}}

    async def _exec_location_story(self, state: SupervisorState) -> dict:
        """执行 LocationStory 子图。"""
        from .location_story import LocationStoryWorkflow
        from ..utils.entity_names import resolve_location_from_query
        wf = LocationStoryWorkflow(
            self.llm, self.retriever, self.prompts,
            node_client=self.node_client,
            checkpointer=None,  # v6: checkpointer 禁用 (LangGraph 版本兼容性)
        )
        query = self._extract_query(state)
        location_name, _ = resolve_location_from_query(query)
        result = await wf.run(location_name, query=query, thread_id=state.get("thread_id", "default"),
                              system_context=getattr(self, '_request_system_context', ''))
        return {"subgraph_result": {"workflow": "location_story", **result}}

    async def _exec_comparison(self, state: SupervisorState) -> dict:
        """执行 Comparison 子图。"""
        from .comparison import ComparisonWorkflow
        wf = ComparisonWorkflow(
            self.llm, self.retriever, self.prompts,
            checkpointer=None,  # v6: checkpointer 禁用 (LangGraph 版本兼容性)
        )
        query = self._extract_query(state)
        result = await wf.run(aspect=query, query=query, thread_id=state.get("thread_id", "default"))
        return {"subgraph_result": {"workflow": "comparison", **result}}

    async def _exec_itinerary(self, state: SupervisorState) -> dict:
        """执行 ItineraryPlanner 子图（完整版 — 天气 · 酒店 · 交通 · 路线优化 · 真实数据）。"""
        from .itinerary import ItineraryPlannerWorkflow
        wf = ItineraryPlannerWorkflow(
            self.llm, self.retriever, self.prompts,
            node_client=self.node_client,
            checkpointer=None,  # v6: checkpointer 禁用 (LangGraph 版本兼容性)
            mcp_registry=self.mcp,
        )
        query = self._extract_query(state)
        result = await wf.run(query, thread_id=state.get("thread_id", "default"),
                              system_context=getattr(self, '_request_system_context', ''))
        return {"subgraph_result": {
            "workflow": "itinerary",
            "answer": result.get("itinerary", ""),
            "weather_advice": result.get("weather_advice", ""),
            "hotel_advice": result.get("hotel_advice", ""),
            "transportation_plan": result.get("transportation_plan", ""),
            "optimized_route": result.get("optimized_route", []),
            "sources": result.get("sources", []),
        }}

    # _exec_character 和 _stream_character 已删除 (U01-05)
    # 角色分析功能已并入 EvolutionWorkbench (target_type="character")

    async def _exec_evolution(self, state: SupervisorState) -> dict:
        """执行 EvolutionWorkbench 子图 (v8 — checkpointer 启用)。"""
        from .evolution_workbench import EvolutionWorkbench
        from ..utils.entity_names import resolve_entity_from_query
        cp = getattr(self, '_request_cp', self._checkpointer)
        wf = EvolutionWorkbench(
            self.llm, self.retriever, self.prompts,
            mcp_registry=self.mcp,
            checkpointer=cp,  # B-002: 启用 checkpointer, 支持用户隔离
            node_client=self.node_client,
        )
        query = self._extract_query(state)

        # 优先使用 state 中已分类的参数
        tracking_target = state.get("evolution_target", "")
        target_type = state.get("evolution_target_type", "location")
        if not tracking_target:
            loc, char, etype = resolve_entity_from_query(query)
            tracking_target = loc if loc != "未指定" else (char or query)
            target_type = etype if etype != "unknown" else "location"

        evo_tid = f"{state.get('thread_id', 'default')}_evo_{uuid.uuid4().hex[:6]}"
        result = await wf.run(query, tracking_target=tracking_target,
                             target_type=target_type,
                             thread_id=evo_tid,
                             system_context=getattr(self, '_request_system_context', ''))
        return {"subgraph_result": {
            "workflow": "evolution",
            "answer": result.get("evolution_narrative", ""),
            "evolution_map": {
                "trackingTarget": result.get("tracking_target", ""),
                "motifs": result.get("motifs", []),
                "motifMatrix": result.get("motif_matrix", []),
                "phylogeny": result.get("phylogeny", {}),
                "claims": result.get("claims", []),
                "evolutionNarrative": result.get("evolution_narrative", ""),
            },
        }}

    async def _exec_dynasty_compare(self, state: SupervisorState) -> dict:
        """执行跨朝代对读 (v10 新增) — 通过 MCP 工具 dynasty_compare_elements 实现。

        委托 ToolDispatcher 统一调度：自动注入 user_id（修复鉴权）+ tool_call_context 埋点。
        结果直接给 UI，不截断（truncate=False）。
        正则提取失败时降级走 Agent Loop ReAct，让 LLM 自行推理章节号。
        """
        query = self._extract_query(state)
        # 从用户消息中提取朝代和章节编号
        left_ch, right_ch = _extract_chapter_numbers(query)
        if left_ch is None or right_ch is None:
            # 正则提取失败 → 降级走 Agent Loop ReAct
            from ..agent_loop import AgentLoop
            agent = AgentLoop(
                self.llm, self.retriever, self.prompts,
                mcp_registry=self.mcp, node_client=self.node_client,
            )
            result = await agent.run(
                query, thread_id=state.get("thread_id", ""),
                history=_extract_history_from_state(state),
            )
            return {"subgraph_result": {
                "workflow": "agent_loop_fallback",
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
            }}
        try:
            tool_result = await self.dispatcher.dispatch(
                "dynasty_compare_elements", {
                    "left_chapter_number": left_ch,
                    "right_chapter_number": right_ch,
                    "query": query,
                },
                caller="langgraph_supervisor", purpose="dynasty_compare",
                user_id=getattr(self, "_request_user_id", ""),
                upstream="_exec_dynasty_compare",
                truncate=False,
            )
            if not tool_result.success:
                return {"subgraph_result": {
                    "workflow": "dynasty_compare",
                    "error": tool_result.error or "跨朝代对比工具执行失败",
                }}
            raw = tool_result.result
        except (RuntimeError, ValueError, ConnectionError) as e:
            return {"subgraph_result": {
                "workflow": "dynasty_compare",
                "error": f"跨朝代对比工具执行失败: {e}",
            }}
        return {"subgraph_result": {"workflow": "dynasty_compare", **raw}}

    # ===== 输出格式化 =====

    async def _format_output(self, state: SupervisorState) -> dict:
        """统一输出格式。"""
        subgraph_result = state.get("subgraph_result", {})
        intent = state.get("intent", "literature_qa")
        return {
            "subgraph_result": {
                **subgraph_result,
                "intent": intent,
                "intent_label": INTENT_LABELS.get(intent, "未知"),
            }
        }

    # ===== 公共接口 =====

    async def run(self, query: str, conversation_id: Optional[str] = None,
                  history: Optional[list] = None,
                  paragraphs: Optional[list] = None,
                  user_id: str = "",
                  user_memory=None,
                  system_context: str = "") -> dict:
        """非流式执行 (v6 — 绕过 LangGraph ainvoke, 直接调用子图)。

        不使用 _graph.ainvoke() 以避免 AsyncSqliteSaver 兼容性问题。
        逻辑与 astream_events 对齐: 意图分类 → 路由 → 子图.run()。
        """
        self._request_cp = user_memory.get_checkpointer() if user_memory else self._checkpointer
        self._request_user_id = user_id
        self._request_system_context = system_context
        from langchain_core.messages import HumanMessage, AIMessage
        thread_id = conversation_id or str(uuid.uuid4())

        # 构建 messages（前端已按 token 预算裁剪，此处不再截断）
        initial_messages = []
        if history:
            for h in history:
                role = h.get("role", "")
                content = h.get("content", "")
                if role == "user":
                    initial_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    initial_messages.append(AIMessage(content=content[:500]))
        initial_messages.append(HumanMessage(content=query))

        state: SupervisorState = {
            "messages": initial_messages,
            "intent": "",
            "intent_confidence": 0.0,
            "keywords": [],
            "evolution_target": "",
            "evolution_target_type": "",
            "subgraph_result": {},
            "conversation_id": conversation_id or thread_id,
            "thread_id": thread_id,
            "error": None,
            "_paragraphs": paragraphs,  # 阶段 0 — API-01c
        }

        # Step 1: 意图分类
        updates = await self._classify_intent(state)
        state.update(updates)
        intent = state.get("intent", "literature_qa")

        # Step 2: 路由到子图并执行 (非流式)
        result = await self._exec_by_intent(intent, state)
        sr = result.get("subgraph_result", {})
        return {
            "answer": sr.get("answer", ""),
            "sources": sr.get("sources", []),
            "workflow_used": sr.get("workflow", intent),
            "intent": intent,
            "intent_label": INTENT_LABELS.get(intent, ""),
            "conversation_id": conversation_id,
            "degraded": sr.get("degraded", False),
        }

    async def astream_events(self, query: str, conversation_id: Optional[str] = None,
                             history: Optional[list] = None,
                             paragraphs: Optional[list] = None,
                             user_id: str = "",
                             user_memory=None,
                             system_context: str = "") -> AsyncGenerator[dict, None]:
        """★ Supervisor 流式执行入口 (v6 统一架构)。

        流程: 意图分类 → 路由到子图流式执行 → done
        不再包含快路径 (_try_fast_tool_path 已移入 AgentLoop 的 _try_fast_answer)
        """
        self._request_cp = user_memory.get_checkpointer() if user_memory else self._checkpointer
        self._request_user_id = user_id
        self._request_system_context = system_context
        from langchain_core.messages import HumanMessage, AIMessage
        from ..utils.entity_names import resolve_location_from_query, resolve_character_from_query
        thread_id = conversation_id or str(uuid.uuid4())

        # 构建消息历史（前端已按 token 预算裁剪，此处不再截断）
        initial_messages = []
        if history:
            for h in history:
                role = h.get("role", "")
                content = h.get("content", "")
                if role == "user":
                    initial_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    initial_messages.append(AIMessage(content=content[:300]))
        initial_messages.append(HumanMessage(content=query))

        # ---- v10: BU-05b 直通路由 — 前端程序化请求绕过 LLM 规划 ----
        passthrough_intent = _detect_passthrough_intent(query)
        if passthrough_intent:
            async for event in self._stream_passthrough(passthrough_intent, query, thread_id):
                yield event
            yield {"type": "done", "data": {
                "workflow": passthrough_intent["intent"],
                "thread_id": thread_id,
            }}
            return

        # Step 1: 意图分类 (v12 统一分类器, ~200ms)
        yield {"type": "status", "data": "正在分析问题意图..."}
        initial: SupervisorState = {
            "messages": initial_messages,
            "intent": "",
            "intent_confidence": 0.0,
            "keywords": [],

"evolution_target": "",
            "evolution_target_type": "",
            "subgraph_result": {},
            "conversation_id": conversation_id or thread_id,
            "thread_id": thread_id,
            "error": None,
            "_paragraphs": paragraphs,  # 阶段 0 — API-01c
        }

        updates = await self._classify_intent(initial)
        initial.update(updates)
        intent = initial.get("intent", "literature_qa")
        keywords = initial.get("keywords", [])
        confidence = initial.get("intent_confidence", 0)

        # v12: LLM 误分类 retry 兜底 — 检测明显 UI 命令被误判为 literature_qa
        if intent == "literature_qa":
            direct_cmd = _detect_direct_command(query)
            if direct_cmd and (
                (direct_cmd["intent"] == "open_chapter" and direct_cmd.get("chapter_number", 0) > 0)
                or direct_cmd["intent"] == "map_control"
            ):
                logger.warning(
                    f"[Supervisor] LLM misclassified as literature_qa, "
                    f"retry with hint: {direct_cmd['intent']}"
                )
                retry_query = (
                    f"[系统指令: 这是一条 UI 操作命令（{direct_cmd['intent']}），"
                    f"不是文献问答。你必须返回 intent={direct_cmd['intent']}。] "
                )
                retry_messages = list(initial.get("messages", []))
                from langchain_core.messages import HumanMessage as _HM
                retry_messages.append(_HM(content=retry_query + query))
                retry_state = dict(initial)
                retry_state["messages"] = retry_messages
                retry_updates = await self._classify_intent(retry_state)
                retry_intent = retry_updates.get("intent", "")
                if retry_intent in ("open_chapter", "map_control"):
                    intent = retry_intent
                    initial.update(retry_updates)
                    keywords = initial.get("keywords", keywords)
                    confidence = initial.get("intent_confidence", confidence)
                else:
                    logger.warning(
                        f"[Supervisor] retry failed, falling back to regex: {direct_cmd['intent']}"
                    )
                    intent = direct_cmd["intent"]
                    # v12: regex fallback — 将 regex 提取的参数注入 initial
                    if direct_cmd["intent"] == "open_chapter":
                        initial["_direct_cmd"] = direct_cmd
                    elif direct_cmd["intent"] == "map_control":
                        initial["_direct_cmd"] = direct_cmd

        yield {
            "type": "intent",
            "data": {
                "intent": intent,
                "intent_label": INTENT_LABELS.get(intent, "未知"),
                "keywords": keywords[:10],
                "confidence": round(confidence, 2),
                # v12: 路由选择理由（LLM 生成或护栏/降级说明），前端展示+审计
                "route_reason": initial.get("route_reason", ""),
            }
        }
        yield {"type": "status", "data": f"识别为: {INTENT_LABELS.get(intent, intent)} · 关键词=[{' '.join(keywords[:5])}]"}

        # Step 2: 根据意图路由到对应子图 (v6: 5 种意图, 无快路径)
        intent_stream_map = {
            "literature_qa": self._stream_literature_qa,
            "location_story": self._stream_location_story,
            "comparison": self._stream_comparison,
            "itinerary": self._stream_itinerary,
            "evolution": self._stream_evolution,
            "dynasty_compare": self._stream_dynasty_compare,
            "annotation": self._stream_annotation,
            "open_chapter": self._stream_open_chapter,
            "map_control": self._stream_map_control,
        }
        stream_func = intent_stream_map.get(intent, self._stream_literature_qa)

        # B-115: 结构化日志
        target_str = initial.get("evolution_target", "") or query[:40]
        logger.info(f"[Supervisor] intent={intent} confidence={round(confidence, 2)} target={target_str}")
        logger.info(f"[Supervisor] routing to {INTENT_LABELS.get(intent, intent)} thread_id={thread_id}")

        t0 = time.monotonic()

        # B-114: 错误边界 — 子图异常时优雅降级
        try:
            async for event in stream_func(query, thread_id, initial):
                if event.get("type") == "done":
                    continue
                yield event
        except Exception as e:
            logger.error(f"[Supervisor] 子图执行异常: {e}")
            yield {"type": SSEEventType.ERROR, "data": {
                "code": "SUBGRAPH_ERROR",
                "message": f"分析引擎遇到问题: {str(e)[:200]}",
                "workflow": intent,
            }}

        duration_ms = int((time.monotonic() - t0) * 1000)
        logger.info(f"[Supervisor] subgraph completed workflow={intent} duration_ms={duration_ms}")

        # B-113: 统一的 done 事件（仅此处发出）
        yield {"type": "done", "data": {
            "workflow": intent,
            "intent_label": INTENT_LABELS.get(intent, ""),
            "thread_id": thread_id,
            "duration_ms": duration_ms,
        }}

    # ===== 子图流式执行器 =====

    async def _stream_literature_qa(self, query: str, thread_id: str,
                                     state: SupervisorState) -> AsyncGenerator[dict, None]:
        from ..agent_loop import AgentLoop
        agent = AgentLoop(
            self.llm, self.retriever, self.prompts,
            mcp_registry=self.mcp,
            node_client=self.node_client,
            user_id=getattr(self, '_request_user_id', ''),
            system_context=getattr(self, '_request_system_context', ''),
        )
        # Extract conversation history from state (exclude last message = current query)
        history = _extract_history_from_state(state)
        try:
            async for event in agent.run_stream(query, history=history, thread_id=thread_id):
                if event.get("type") == "done":
                    continue
                yield event
        except Exception as e:
            logger.exception(f"[Supervisor] literature_qa 流异常: {e}")
            yield {"type": SSEEventType.ERROR,
                   "data": {"message": f"回答过程中断（{str(e)[:120]}），请重试一次。"}}

    async def _load_existing_agent_annotations(self, ch_num: int) -> list[dict]:
        """拉取该章已保存的 agent 标注（转为 save 格式），失败时返回空表。

        用于保存前与本次结果合并——同段重跑不同意图（人物/专有名词）
        时保留此前类目的标注，避免"新标注出现、旧标注消失"。
        """
        try:
            data = await self.node_client.get_annotations(
                ch_num, user_id=getattr(self, '_request_user_id', '')
            )
        except Exception as e:
            logger.warning(f"[Supervisor] 读取已有标注失败 ch={ch_num}: {e}")
            return []
        out: list[dict] = []
        for pkey, anns in (data or {}).get("annotations", {}).items():
            try:
                para = int(str(pkey).split(":")[1])
            except (IndexError, ValueError):
                para = 0
            for a in anns:
                if a.get("source") != "agent":
                    continue
                span = a.get("span") or {}
                out.append({
                    "id": a.get("id") or "",
                    "paragraph_index": para,
                    "category": a.get("category", "custom"),
                    "entity": a.get("label", ""),
                    "start_char": span.get("startChar", 0),
                    "end_char": span.get("endChar", 0),
                    "explanation": a.get("note", ""),
                    "color": a.get("color", ""),
                })
        return out

    async def _load_existing_agent_marginalia(self, ch_num: int) -> list[dict]:
        """拉取该章已保存的 agent 旁批（转为 save 格式），失败时返回空表。"""
        try:
            data = await self.node_client.get_annotations(
                ch_num, user_id=getattr(self, '_request_user_id', '')
            )
        except Exception as e:
            logger.warning(f"[Supervisor] 读取已有旁批失败 ch={ch_num}: {e}")
            return []
        out: list[dict] = []
        for pkey, margs in (data or {}).get("marginalia", {}).items():
            try:
                para = int(str(pkey).split(":")[1])
            except (IndexError, ValueError):
                para = 0
            for m in margs:
                if m.get("source") != "agent":
                    continue
                out.append({
                    "id": m.get("id") or "",
                    "chapter_number": m.get("chapterNumber", ch_num),
                    "paragraph_index": m.get("paragraphIndex", para),
                    "anchor_char_offset": m.get("anchorCharOffset"),
                    "content": m.get("content", ""),
                    "source": "agent",
                })
        return out

    @staticmethod
    def _merge_annotations(existing: list[dict], incoming: list[dict]) -> list[dict]:
        """按 (段落, 实体, 位置) 去重合并标注：同键新结果覆盖旧结果，其余保留。"""
        def _key(a: dict):
            return (a.get("paragraph_index"), a.get("entity"),
                    a.get("start_char"), a.get("end_char"))
        merged: dict = {_key(a): a for a in existing if a.get("id")}
        for a in incoming:
            merged[_key(a)] = a
        return list(merged.values())

    @staticmethod
    def _merge_marginalia(existing: list[dict], incoming: list[dict]) -> list[dict]:
        """按 id 合并旁批：同 id（同实体）新内容覆盖旧内容，其余保留。"""
        merged: dict = {m["id"]: m for m in existing if m.get("id")}
        for m in incoming:
            merged[m["id"]] = m
        return list(merged.values())

    async def _stream_annotation(self, query: str, thread_id: str,
                                  state: SupervisorState) -> AsyncGenerator[dict, None]:
        """标注意图: 直接调用 annotate_user_request MCP 工具，绕过 LLM ReAct 决策。
        从 query 中的系统上下文提取 chapter_number，确定性执行标注流程。
        """
        import re as _re
        from ..api.events import SSEEventType as _ET
        from ..agent_loop import AgentLoop

        # === BLOCK A: 提取章节号 (KEEP) ===
        ch_num = 0
        ch_match = _re.search(r'chapter_number\s*=\s*(\d+)', query)
        if ch_match:
            ch_num = int(ch_match.group(1))

        if ch_num <= 0:
            ch_match2 = _re.search(r'chapterNumber[\"\']?\s*:\s*(\d+)', query)
            if ch_match2:
                ch_num = int(ch_match2.group(1))

        # === BLOCK B: 章节号校验 (KEEP) ===
        if ch_num <= 0:
            yield {"type": _ET.CHUNK, "data": "请先在左侧文献树中打开一个章节，然后再标注。"}
            return

        # === BLOCK C: 用户指令清洗 (KEEP) ===
        user_instruction = _re.sub(r'\s*\[系统上下文:.*?\]', '', query).strip()
        if not user_instruction or len(user_instruction) < 2:
            user_instruction = "找出文中实体并标注"

        # === BLOCK D: 删除意图检测与处理 (KEEP) ===
        DELETE_KEYWORDS = ["消除", "删除", "清除", "去掉", "移除", "取消", "删掉", "清空", "抹掉"]
        is_delete_intent = any(kw in user_instruction for kw in DELETE_KEYWORDS)

        if is_delete_intent:
            delete_categories = None
            delete_all = False
            if any(kw in user_instruction for kw in ["地点", "位置", "地名"]):
                delete_categories = ["location"]
            elif any(kw in user_instruction for kw in ["人物", "角色", "人名"]):
                delete_categories = ["person"]
            elif any(kw in user_instruction for kw in ["事件"]):
                delete_categories = ["event"]
            elif any(kw in user_instruction for kw in ["术语", "专有名词", "名词"]):
                delete_categories = ["term"]
            elif any(kw in user_instruction for kw in ["母题", "主题"]):
                delete_categories = ["motif"]
            elif any(kw in user_instruction for kw in ["全部", "所有", "整个", "本章", "清空"]):
                delete_all = True

            # 安全兜底: 未匹配到类别关键词时，不 fallback 为"删全部"，
            # 而是提示用户明确指定类别（避免"专有名词"等未覆盖词导致误删全部）
            if not delete_categories and not delete_all:
                yield {"type": _ET.CHUNK, "data": (
                    "未能识别要删除的标注类别。请明确指定，例如：\n"
                    "- 删除第1章的**人物**标注\n"
                    "- 删除第1章的**术语**标注\n"
                    "- 删除第1章的**全部**标注"
                )}
                return

            cat_desc = "全部类别" if delete_all else "、".join(delete_categories)
            yield {"type": _ET.STATUS, "data": f"正在删除第{ch_num}章{cat_desc}标注..."}

            try:
                raw_result = await self.mcp.call_tool("delete_annotations", {
                    "chapter_number": ch_num,
                    "categories": delete_categories,
                    "delete_all": delete_all,
                    "user_id": getattr(self, '_request_user_id', ''),
                })
                result = raw_result.get("result", raw_result) if isinstance(raw_result, dict) else raw_result
            except (RuntimeError, ValueError, ConnectionError) as e:
                yield {"type": _ET.ERROR, "data": {"code": "TOOL_ERROR", "message": f"删除标注失败: {e}"}}
                return

            if isinstance(result, dict) and "error" in result:
                yield {"type": _ET.ERROR, "data": {"code": "DELETE_ERROR", "message": result["error"]}}
                return

            is_category_delete = isinstance(result, dict) and result.get("categories") is not None

            # Get chapter info for building the SSE event
            if self.node_client is None:
                yield {"type": _ET.ERROR, "data": {"code": "NO_NODE_CLIENT", "message": "Node API 客户端未初始化"}}
                return

            ch_data = await self.node_client.get_literature_by_chapter(ch_num)
            if ch_data is None:
                yield {"type": _ET.ERROR, "data": {"code": "CHAPTER_NOT_FOUND", "message": f"未找到第{ch_num}章"}}
                return

            ch_title = ch_data.get("title", f"第{ch_num}章")
            ch_content = ch_data.get("content", "")
            ch_content = _re.sub(r'\r\n|\r', '\n', ch_content)
            all_paragraphs = [p.strip() for p in ch_content.split('\n\n') if p.strip()]
            p_end = max(0, len(all_paragraphs) - 1)

            if delete_all or not is_category_delete:
                # Full delete: send empty annotation_data for all paragraphs
                clear_groups = [
                    {"passageKey": f"{ch_num}:{pi}", "paragraphIndex": pi,
                     "annotations": [], "insights": []}
                    for pi in range(p_end + 1)
                ]
                yield {"type": _ET.ANNOTATION_DATA, "data": {
                    "chapterNumber": ch_num,
                    "chapterTitle": ch_title,
                    "paragraphRange": [0, p_end],
                    "passageAnnotations": clear_groups,
                    "deleteAll": True,
                }}
            else:
                # Category-specific delete: send empty annotation_data with deleteCategories
                # so the frontend can filter which categories to keep
                clear_groups = [
                    {"passageKey": f"{ch_num}:{pi}", "paragraphIndex": pi,
                     "annotations": [], "insights": []}
                    for pi in range(p_end + 1)
                ]
                yield {"type": _ET.ANNOTATION_DATA, "data": {
                    "chapterNumber": ch_num,
                    "chapterTitle": ch_title,
                    "paragraphRange": [0, p_end],
                    "passageAnnotations": clear_groups,
                    "deleteAll": True,
                    "deleteCategories": delete_categories,
                }}

            yield {"type": _ET.CHUNK, "data": f"已删除第{ch_num}章{cat_desc}标注。"}
            return

        # === BLOCK E: 自定义类别检测 (KEEP) ===
        # 检测用户指定的自定义标注类别（如"颜色""动作""器物"等非标准类别）
        custom_categories = None
        custom_match = _re.search(r'(?:标注|标出|找出|标记).*?(?:的|文中)?(.{1,4})(?:词|色|类|别)', user_instruction)
        if not custom_match:
            custom_match = _re.search(r'(?:把|将|给).*?(.{1,4})(?:标|标注|标记|高亮)', user_instruction)
        if custom_match:
            custom_word = custom_match.group(1).strip()
            if custom_word and custom_word not in ("人物", "地点", "事件", "术语", "母题", "实体", "所有", "全部"):
                custom_categories = ["custom"]
                user_instruction = f"标注类别: {custom_word}。{user_instruction}"

        yield {"type": _ET.STATUS, "data": f"正在标注第{ch_num}章..."}
        if custom_categories:
            yield {"type": _ET.CHUNK, "data": f"使用自定义类别标注。"}

        # === BLOCK F: 段落索引解析 (KEEP) ===
        # 第12段 → [11], 第12-15段 → [11,12,13,14], 未指定 → None (全章)
        para_indices = None
        para_match = _re.search(r'第\s*(\d+)\s*(?:[-~到至]\s*(\d+)\s*)?段', user_instruction)
        if para_match:
            start_p = int(para_match.group(1)) - 1
            end_p_str = para_match.group(2)
            if end_p_str:
                end_p = int(end_p_str) - 1
                para_indices = list(range(start_p, end_p + 1))
            else:
                para_indices = [start_p]

        # === BLOCK G: 旁注意图检测 (KEEP) ===
        _MARGINALIA_KW = ["旁注", "批注", "旁批", "评注", "批语", "写注", "加注"]
        has_marginalia_intent = any(kw in user_instruction for kw in _MARGINALIA_KW)

        # === BLOCK H: 标注执行入口 (P2-02) ===
        # 检测是否走新路径
        _paragraphs_input = state.get("_paragraphs") if hasattr(state, 'get') else None

        # 方案 A: 当 _paragraphs_input 为空时，主动从 node_client 拉取章节段落，
        # 让"全章 + 标准类别"标注也走流式两轮 pipeline，避免旧路径在
        # mcp.call_tool 一次性同步阻塞期间前端工作台长时间无任何逐段反馈。
        # 保留旧路径的情况: 指定段落标注(para_indices) 或 自定义类别(custom_categories)，
        # 因为新路径暂不支持段落子集过滤和 custom 类别（段落少时旧路径阻塞短，可接受）。
        if (not _paragraphs_input and self.node_client
                and not para_indices and not custom_categories):
            try:
                _ch_data_h = await self.node_client.get_literature_by_chapter(ch_num)
                if _ch_data_h is not None:
                    _content_h = _re.sub(r'\r\n|\r', '\n', _ch_data_h.get("content", ""))
                    _paragraphs_input = [p.strip() for p in _content_h.split('\n\n') if p.strip()]
            except Exception as _e_h:
                logger.warning("[P2-02] 主动拉取章节段落失败，回退旧路径: %s", _e_h)

        if _paragraphs_input and len(_paragraphs_input) > 0:
            # ── 新路径 ──
            async for event in self._stream_annotation_new_path(
                ch_num, ch_title="", user_instruction=user_instruction,
                paragraphs=_paragraphs_input,
            ):
                yield event
            return  # 新路径完成，不执行任何旧代码

        # ── 旧路径（_paragraphs_input 为空时到达此处，代码完全不变）──
        yield {"type": _ET.TOOL_CALL, "data": {
            "tool": "annotate_user_request",
            "arguments": {
                "chapter_number": ch_num,
                "user_instruction": user_instruction,
                "paragraph_indices": para_indices,
                "has_marginalia": has_marginalia_intent,
            },
        }}

        call_params: dict = {
            "chapter_number": ch_num,
            "user_instruction": user_instruction,
        }
        if para_indices is not None:
            call_params["paragraph_indices"] = para_indices
        if has_marginalia_intent:
            call_params["marginalia_notes"] = []

        # C: 预期管理 — 开工前告知阶段与耗时预期，消除"卡住"感
        yield {"type": _ET.STATUS, "data": (
            f"开始标注第{ch_num}章：实体发现 → 精确定位 → 旁注生成，"
            "全程约 1 分钟，标注结果将逐段出现在正文中..."
        )}

        try:
            result = await self.mcp.call_tool("annotate_user_request", call_params)
            raw = result.get("result", result) if isinstance(result, dict) else result
        except (RuntimeError, ValueError, ConnectionError) as e:
            yield {"type": _ET.ERROR, "data": {"code": "TOOL_ERROR", "message": f"工具调用异常: {e}"}}
            return

        is_error = isinstance(raw, dict) and "error" in raw
        yield {"type": _ET.TOOL_RESULT, "data": {
            "tool": "annotate_user_request",
            "summary": raw.get("error", f"标注{len(raw.get('annotations', []))}个实体") if isinstance(raw, dict) else "",
            "success": not is_error,
        }}

        if is_error:
            yield {"type": _ET.CHUNK, "data": f"标注未成功: {raw.get('error', '')}"}
            return

        # Fast fail: node_client must be available for paragraph boundary calculation
        if self.node_client is None:
            yield {"type": _ET.ERROR, "data": {"code": "NO_NODE_CLIENT", "message": "Node API 客户端未初始化，无法计算段落边界"}}
            return

        # 直接构建 annotation_data 事件，不通过 _format_annotation_data
        # 避免单行 JSON 过大导致前端 SSE 解析失败
        target = raw.get("target", {}) if isinstance(raw, dict) else {}
        ch_title = target.get("chapter_title", "")
        para_range = target.get("paragraph_range", [0, 0])
        annotations_raw = raw.get("annotations", []) if isinstance(raw, dict) else []
        ann_count = len(annotations_raw)

        # 逐段分组发送 annotation_data，每段一行
        import re as _re2
        ch_data = await self.node_client.get_literature_by_chapter(ch_num)
        if ch_data is None:
            yield {"type": _ET.ERROR, "data": {"code": "CHAPTER_NOT_FOUND", "message": f"未找到第{ch_num}章"}}
            return
        ch_content = ch_data.get("content", "")
        ch_content = _re2.sub(r'\r\n|\r', '\n', ch_content)
        all_paragraphs = [p.strip() for p in ch_content.split('\n\n') if p.strip()]
        p_start = max(0, para_range[0]) if isinstance(para_range, list) and len(para_range) >= 2 else 0
        p_end = min(len(all_paragraphs), (para_range[1] if isinstance(para_range, list) and len(para_range) >= 2 else 0) + 1)

        # 构建边界表
        boundaries = []
        offset = 0
        for p_idx in range(p_start, p_end):
            p_text = all_paragraphs[p_idx]
            boundaries.append({"idx": p_idx, "start": offset, "end": offset + len(p_text)})
            offset += len(p_text) + 2  # +2 for paragraph separator (\n\n)

        # 按段落分组标注
        grouped: dict[int, list[dict]] = {}
        for i, ann in enumerate(annotations_raw):
            sc = ann.get("start_char", 0)
            ec = ann.get("end_char", sc + len(ann.get("entity", "")))
            para_idx = p_start
            rel_start = sc
            for b in boundaries:
                if b["start"] <= sc < b["end"]:
                    para_idx = b["idx"]
                    rel_start = sc - b["start"]
                    break
            rel_end = rel_start + (ec - sc)

            if para_idx not in grouped:
                grouped[para_idx] = []
            grouped[para_idx].append({
                "id": f"agent_ann_{ch_num}_{para_idx}_{i}",
                "category": ann.get("category", "custom"),
                "label": ann.get("entity", ""),
                "span": {"startChar": rel_start, "endChar": rel_end},
                "explanation": ann.get("explanation", "")[:80],
                "confidence": ann.get("confidence", "medium"),
                "color": ann.get("color", "#F1C40F"),
                "source": "agent",
                "chapterNumber": ch_num,
                "paragraphIndex": para_idx,
                "text": ann.get("entity", ""),
                "createdAt": int(__import__('time').time() * 1000),
                "updatedAt": int(__import__('time').time() * 1000),
            })

        # 逐段发送小 annotation_data 事件
        for p_idx in sorted(grouped.keys()):
            passage_annotations = [{
                "passageKey": f"{ch_num}:{p_idx}",
                "paragraphIndex": p_idx,
                "annotations": grouped[p_idx],
                "insights": [],
            }]
            yield {
                "type": "annotation_data",
                "data": {
                    "chapterNumber": ch_num,
                    "chapterTitle": ch_title,
                    "paragraphRange": [p_start, p_end - 1],
                    "passageAnnotations": passage_annotations,
                    "marginalia": [],
                },
            }

        # 旁注/批注支持：用户要求写旁注时，LLM 为实体生成批注文本
        _MARGINALIA_KEYWORDS = ["旁注", "批注", "旁批", "评注", "批语", "写注", "加注"]
        marginalia_list: list = []
        if any(kw in user_instruction for kw in _MARGINALIA_KEYWORDS) and ann_count > 0:
            yield {"type": _ET.STATUS, "data": f"正在为 {ann_count} 个实体生成旁注..."}
            from ..config import settings as _cfg

            # 取优先级最高的类别（人物 > 地点 > 事件 > 术语 > 母题），去重，最多15个
            cat_priority = {"person": 0, "location": 1, "event": 2, "term": 3, "motif": 4}
            seen_entities: set[str] = set()
            candidates = []
            for p_idx in sorted(grouped.keys()):
                for ann in grouped[p_idx]:
                    entity = ann.get("label", "")
                    if entity and entity not in seen_entities and len(entity) >= 2:
                        seen_entities.add(entity)
                        prio = cat_priority.get(ann.get("category", "custom"), 5)
                        candidates.append((prio, p_idx, ann))

            candidates.sort(key=lambda x: (x[0], x[2].get("label", "")))
            candidates = candidates[:15]

            marginalia_list = []
            _marg_total = len(candidates)
            _marg_done = 0
            for prio, p_idx, ann in candidates:
                entity = ann.get("label", "")
                category = ann.get("category", "custom")
                cat_name = {"person": "人物", "location": "地点", "event": "事件",
                            "term": "术语", "motif": "母题"}.get(category, category)

                m_system = (
                    f"你是白蛇传学术研究助手。为原文中标注的{cat_name}「{entity}」撰写一句旁注。\n"
                    "旁注应是一句30-60字的学术性点评，说明该实体在文中的叙事功能或文化含义。\n"
                    "只返回旁注文字，不要其他内容，不要引号包裹。"
                )
                para_text = all_paragraphs[p_idx] if p_idx < len(all_paragraphs) else ""
                context = para_text[max(0, ann["span"]["startChar"] - 30):ann["span"]["endChar"] + 30]
                m_user = f"原文片段: ...{context}...\n实体: {entity}\n旁注:"

                try:
                    with llm_call_context("supervisor", "generate_annotation_note",
                                          upstream="annotation_scan", downstream="note_output"):
                        m_resp = await self.llm.generate(m_system, m_user,
                            model=_cfg.effective_fast_model, max_tokens=100, temperature=0.5,
                            thinking_disabled=True)
                    note = m_resp.content.strip().strip('"').strip("'").strip("「").strip("」")
                    if note:
                        marginalia_list.append({
                            "id": f"agent_marg_{ch_num}_{p_idx}_{entity}",
                            "chapterNumber": ch_num,
                            "paragraphIndex": p_idx,
                            "anchorCharOffset": ann["span"]["startChar"],
                            "content": f"{entity}: {note}",
                            "color": ann.get("color", "#F5A623"),
                            "source": "agent",
                        })
                    # A: 旁注逐条实时播报 — 消除 20~30s 静默期
                    _marg_done += 1
                    yield {"type": _ET.THINKING_STEP, "data": {
                        "step": "generate_marginalia",
                        "message": f"旁注 {_marg_done}/{_marg_total}: 「{entity}」{'✓' if note else '× 跳过'}",
                    }}
                except Exception:
                    pass

            if marginalia_list:
                yield {
                    "type": "marginalia_data",
                    "data": {"marginalia": marginalia_list},
                }
                yield {"type": _ET.CHUNK, "data": f"已为 {len(marginalia_list)} 个实体生成旁注。"}

        # v12: 保存标注到后端文件，确保后续删除操作能命中
        # MCP 工具输出的 annotations_raw 缺少 id 和 paragraph_index，
        # 需要补齐后再保存，否则 INSERT 会因主键缺失而失败
        if self.node_client and annotations_raw:
            annotations_for_save = []
            for i, ann in enumerate(annotations_raw):
                sc = ann.get("start_char", 0)
                ec = ann.get("end_char", sc + len(ann.get("entity", "")))
                para_idx = p_start
                for b in boundaries:
                    if b["start"] <= sc < b["end"]:
                        para_idx = b["idx"]
                        break
                annotations_for_save.append({
                    **ann,
                    "id": f"agent_ann_{ch_num}_{para_idx}_{i}",
                    "paragraph_index": para_idx,
                })
            try:
                # 增量合并保存: 拉取已有 agent 标注/旁批 → 去重合并 → 全量保存，
                # 同段重跑不同意图时保留此前类目的标注
                existing_anns = await self._load_existing_agent_annotations(ch_num)
                existing_margs = await self._load_existing_agent_marginalia(ch_num)
                merged_anns = self._merge_annotations(existing_anns, annotations_for_save)
                merged_margs = self._merge_marginalia(existing_margs, marginalia_list)
                await self.node_client.save_annotations(
                    ch_num,
                    annotations=merged_anns,
                    marginalia=merged_margs if merged_margs else None,
                    user_id=getattr(self, '_request_user_id', ''),
                )
            except Exception as e:
                logger.error(f"[Supervisor] save_annotations (old path) 失败 ch={ch_num}: {e}")
                yield {"type": _ET.ERROR, "data": {
                    "code": "SAVE_FAILED",
                    "message": f"第{ch_num}章标注保存失败: {str(e)[:200]}。请手动刷新页面重试。",
                }}

        yield {"type": _ET.CHUNK, "data": (
            f"已完成第{ch_num}章标注（{ann_count} 个实体）。"
            f"正文已高亮 {len(grouped)} 段，点击高亮词可查看释义。"
        )}

    # ===== P2-02: 新标注路径（两轮流水线 + 批注生成）=====

    async def _stream_annotation_new_path(
        self, ch_num: int, ch_title: str,
        user_instruction: str, paragraphs: list,
    ) -> AsyncGenerator[dict, None]:
        """新标注路径：两轮流水线 + 分批流式 Marginalia 生成。

        阶段 2 — P2-02.
        由 _stream_annotation 在检测到 state["_paragraphs"] 非空时调用。
        """
        import time as _time
        from ..mcp.tools.annotation_pipeline import (
            discovery_pass, precise_resolution_pass
        )
        from ..mcp.tools.coverage_tracker import CoverageTracker
        from ..config import settings as _cfg
        from ..api.events import SSEEventType as _ET

        # ── 类别检测（内联简单版本，阶段 3 完成后替换为 parse_annotation_intent）──
        # TODO: 阶段 3 完成后替换为:
        #   from ..mcp.tools.annotate_user_request import parse_annotation_intent
        #   parsed = parse_annotation_intent(user_instruction)
        #   _categories = parsed["categories"] if parsed else [...]
        _categories = ["person", "location", "event", "term", "motif"]
        _cat_hints = {"人物": "person", "地点": "location", "事件": "event",
                       "术语": "term", "母题": "motif"}
        for cn_label, en_key in _cat_hints.items():
            if cn_label in user_instruction:
                _categories = [en_key]
                break

        # ── 获取章节标题（如未传入则从 node_client 获取）──
        if not ch_title and self.node_client:
            try:
                ch_data = await self.node_client.get_literature_by_chapter(ch_num)
                if ch_data:
                    ch_title = ch_data.get("title", f"第{ch_num}章")
            except Exception:
                ch_title = f"第{ch_num}章"

        # ── Pass 1: Discovery ──
        # C: 预期管理 — 开工前告知阶段与规模
        yield {"type": _ET.STATUS, "data": (
            f"开始标注第{ch_num}章（共 {len(paragraphs)} 段）："
            "实体发现 → 精确定位 → 旁注生成，全程约 1 分钟，标注结果将逐段出现在正文中..."
        )}
        yield {"type": _ET.STATUS, "data": "Pass 1/3: 正在扫描全文发现实体..."}

        try:
            discoveries = await discovery_pass(
                paragraphs, _categories, self.llm,
                prompt_registry=self.prompts,
                concurrency=_cfg.annotation_discovery_concurrency,
                window_size=_cfg.annotation_discovery_window_size,
                overlap=_cfg.annotation_discovery_overlap,
            )
        except Exception as e:
            logger.error("[P2-02] discovery_pass failed: %s", e)
            yield {"type": _ET.ERROR, "data": {
                "code": "DISCOVERY_ERROR",
                "message": f"实体发现阶段失败: {str(e)[:200]}",
            }}
            return

        yield {"type": _ET.STATUS, "data": f"发现 {len(discoveries)} 个候选实体"}

        if not discoveries:
            yield {"type": _ET.CHUNK, "data": "未发现匹配的实体。请尝试调整标注类别。"}
            return

        # ── Pass 2: Precise Resolution ──
        yield {"type": _ET.STATUS, "data": "Pass 2/3: 正在精确定位..."}

        tracker = CoverageTracker(len(paragraphs), _categories)
        _collected = {}  # Dict[int, List[dict]] — paragraph_index → annotations

        try:
            async for result in precise_resolution_pass(
                paragraphs, discoveries, self.llm,
                prompt_registry=self.prompts,
                concurrency=_cfg.annotation_precise_concurrency,
                coverage_tracker=tracker,
            ):
                p_idx = result["paragraph_index"]
                p_anns = result["annotations"]
                p_cov = result["coverage"]

                # 收集标注供 Pass 3 使用
                _collected[p_idx] = p_anns

                frontend_anns = []
                for i, ann in enumerate(p_anns):
                    frontend_anns.append({
                        "id": f"agent_ann_{ch_num}_{p_idx}_{i}",
                        "category": ann["category"],
                        "label": ann["entity"],
                        "span": {
                            "startChar": ann["start_char"],
                            "endChar": ann["end_char"],
                        },
                        "explanation": ann.get("explanation", "")[:80],
                        "confidence": ann.get("confidence", "medium"),
                        "color": ann.get("color", "#F1C40F"),
                        "source": "agent",
                        "chapterNumber": ch_num,
                        "paragraphIndex": p_idx,
                        "text": ann["entity"],
                        "createdAt": int(_time.time() * 1000),
                        "updatedAt": int(_time.time() * 1000),
                    })

                yield {
                    "type": "annotation_data",
                    "data": {
                        "chapterNumber": ch_num,
                        "chapterTitle": ch_title,
                        "passageKey": f"{ch_num}:{p_idx}",
                        "paragraphIndex": p_idx,
                        "annotations": frontend_anns,
                        "insights": [],
                        "marginalia": [],
                        "coverage": p_cov,
                        "isComplete": False,
                    },
                }

            # ── Pass 3: Marginalia 生成 ──
            _MARGINALIA_KEYWORDS = ["旁注", "批注", "旁批", "评注", "批语", "写注", "加注"]
            has_marginalia_intent = any(
                kw in user_instruction for kw in _MARGINALIA_KEYWORDS
            )

            # 收集所有标注结果
            all_annotations = []
            for p_idx, anns in _collected.items():
                for ann in anns:
                    all_annotations.append((p_idx, ann))

            yield {"type": _ET.STATUS, "data": f"定位到 {len(all_annotations)} 个实体，生成标注数据..."}

            marginalia_list: list = []  # 初始化，确保后续 save_annotations 可安全引用

            if has_marginalia_intent and all_annotations:
                yield {"type": _ET.STATUS, "data": "Pass 3/3: 正在生成批注..."}
                cat_priority = {"person": 0, "location": 1, "event": 2,
                                "term": 3, "motif": 4}
                seen_entities = set()
                candidates = []
                for p_idx, ann in all_annotations:
                    entity = ann.get("entity", "")
                    if entity and entity not in seen_entities and len(entity) >= 2:
                        seen_entities.add(entity)
                        prio = cat_priority.get(ann.get("category", "custom"), 5)
                        candidates.append((prio, p_idx, ann))
                candidates.sort(key=lambda x: (x[0], x[2].get("entity", "")))

                max_count = _cfg.annotation_marginalia_max_count
                batch_size = _cfg.annotation_marginalia_batch_size

                if len(candidates) > max_count:
                    yield {"type": _ET.CHUNK, "data": (
                        f"共 {len(candidates)} 个实体，分批生成前 {max_count} 个的批注..."
                    )}
                    candidates = candidates[:max_count]

                _all_paras_map = {
                    p["index"]: p["text"] for p in paragraphs
                }

                _marg_total = len(candidates)
                _marg_done = 0
                for batch_start in range(0, len(candidates), batch_size):
                    batch = candidates[batch_start:batch_start + batch_size]
                    batch_margs = []

                    for _prio, p_idx, ann in batch:
                        entity = ann.get("entity", "")
                        category = ann.get("category", "custom")
                        cat_name = {
                            "person": "人物", "location": "地点",
                            "event": "事件", "term": "术语", "motif": "母题"
                        }.get(category, category)

                        m_system = (
                            f"你是白蛇传学术研究助手。为原文中标注的{cat_name}"
                            f"「{entity}」撰写一句旁注。\n"
                            "旁注应是一句30-60字的学术性点评，"
                            "说明该实体在文中的叙事功能或文化含义。\n"
                            "只返回旁注文字，不要其他内容，不要引号包裹。"
                        )
                        para_text = _all_paras_map.get(p_idx, "")
                        s_char = ann.get("start_char", 0)
                        e_char = ann.get("end_char", s_char)
                        context = para_text[
                            max(0, s_char - 30):min(len(para_text), e_char + 30)
                        ]
                        m_user = f"原文片段: ...{context}...\n实体: {entity}\n旁注:"

                        _note_ok = False
                        try:
                            with llm_call_context("supervisor", "generate_annotation_note",
                                                  upstream="annotation_scan", downstream="note_output"):
                                m_resp = await self.llm.generate(
                                    m_system, m_user,
                                    model=_cfg.effective_model,
                                    max_tokens=100,
                                    temperature=0.5,
                                    thinking_disabled=True,
                                )
                            note = (
                                m_resp.content.strip()
                                .strip('"').strip("'")
                                .strip("「").strip("」")
                            )
                            if note:
                                _note_ok = True
                                batch_margs.append({
                                    "id": f"agent_marg_{ch_num}_{p_idx}_{entity}",
                                    "chapterNumber": ch_num,
                                    "paragraphIndex": p_idx,
                                    "anchorCharOffset": s_char,
                                    "content": f"{entity}: {note}",
                                    "color": ann.get("color", "#F5A623"),
                                    "source": "agent",
                                })
                        except Exception:
                            pass

                        # A: 旁注逐条实时播报 — 消除批量生成静默期（本地计数，无跨请求状态）
                        _marg_done += 1
                        yield {"type": _ET.THINKING_STEP, "data": {
                            "step": "generate_marginalia",
                            "message": f"旁注 {_marg_done}/{_marg_total}: 「{entity}」{'✓' if _note_ok else '× 跳过'}",
                        }}

                    if batch_margs:
                        marginalia_list.extend(batch_margs)
                        yield {
                            "type": "marginalia_data",
                            "data": {"marginalia": batch_margs},
                        }

                if marginalia_list:
                    yield {"type": _ET.CHUNK, "data": (
                        f"已为 {len(marginalia_list)} 个实体生成批注。"
                    )}

            # v12: 保存标注到后端文件，确保后续删除操作能命中
            # 新路径也需保存，否则标注仅在 SSE 流中存在，刷新后丢失
            if self.node_client and _collected:
                annotations_for_save = []
                for p_idx, anns in _collected.items():
                    for i, ann in enumerate(anns):
                        annotations_for_save.append({
                            "id": f"agent_ann_{ch_num}_{p_idx}_{i}",
                            "paragraph_index": p_idx,
                            "entity": ann["entity"],
                            "category": ann["category"],
                            "start_char": ann["start_char"],
                            "end_char": ann["end_char"],
                            "explanation": ann.get("explanation", ""),
                            "confidence": ann.get("confidence", "medium"),
                            "color": ann.get("color", ""),
                            "source": ann.get("source", "agent"),
                        })
                if annotations_for_save:
                    try:
                        # 增量合并保存: 拉取已有 agent 标注/旁批 → 去重合并 → 全量保存
                        existing_anns = await self._load_existing_agent_annotations(ch_num)
                        existing_margs = await self._load_existing_agent_marginalia(ch_num)
                        merged_anns = self._merge_annotations(existing_anns, annotations_for_save)
                        merged_margs = self._merge_marginalia(existing_margs, marginalia_list)
                        await self.node_client.save_annotations(
                            ch_num,
                            annotations=merged_anns,
                            marginalia=merged_margs if merged_margs else None,
                            user_id=getattr(self, '_request_user_id', ''),
                        )
                    except Exception as e:
                        logger.error(f"[Supervisor] save_annotations (new path) 失败 ch={ch_num}: {e}")
                        yield {"type": _ET.ERROR, "data": {
                            "code": "SAVE_FAILED",
                            "message": f"第{ch_num}章标注保存失败: {str(e)[:200]}。请手动刷新页面重试。",
                        }}

            # ── 最终完成事件 ──
            final_snap = tracker.snapshot()
            last_p_idx = sorted(_collected.keys())[-1] if _collected else 0
            yield {
                "type": "annotation_data",
                "data": {
                    "chapterNumber": ch_num,
                    "chapterTitle": ch_title,
                    "passageKey": f"{ch_num}:{last_p_idx}",
                    "paragraphIndex": last_p_idx,
                    "annotations": [],
                    "insights": [],
                    "marginalia": [],
                    "coverage": final_snap,
                    "isComplete": True,
                },
            }

            yield {"type": _ET.CHUNK, "data": (
                f"已完成第{ch_num}章标注（{final_snap['totalEntities']} 个实体）。"
                f"正文已高亮 {final_snap['paragraphsAnnotated']} 段，点击高亮词可查看释义。"
            )}
            # 覆盖率/质量口径仅记日志（开发者视角），不再推给用户
            logger.info(
                "[P2-02] annotation quality: coverage=%.0f%% (%d/%d), entities=%d, quality=%s",
                final_snap['coverageRate'] * 100,
                final_snap['paragraphsAnnotated'], final_snap['paragraphsTotal'],
                final_snap['totalEntities'], final_snap['quality'],
            )

        except Exception as e:
            logger.error("[P2-02] precise_resolution_pass failed: %s", e)
            yield {"type": _ET.ERROR, "data": {
                "code": "RESOLUTION_ERROR",
                "message": f"精确定位阶段失败: {str(e)[:200]}",
            }}

    # ===== v12: UI 操作 handler (open_chapter, map_control) =====

    async def _stream_open_chapter(self, query: str, thread_id: str,
                                   state: SupervisorState) -> AsyncGenerator[dict, None]:
        """LLM 分类为 open_chapter 时 — 驱动前端工作台打开指定章节。

        参数提取优先级：系统上下文 chapter_number=N > 用户原文第X章/回。
        """
        import re as _re
        from ..api.events import SSEEventType as _ET

        ch_num = 0

        ctx_match = _re.search(r'chapter_number\s*=\s*(\d+)', query)
        if ctx_match:
            ch_num = int(ctx_match.group(1))

        if ch_num <= 0:
            ch_match = _re.search(
                r'第\s*([\d一二三四五六七八九十]+)\s*(?:章|回)', query
            )
            if ch_match:
                ch_num = _parse_single_chapter_number(ch_match.group(1)) or 0

        if ch_num <= 0:
            yield {"type": _ET.CHUNK,
                   "data": "请指定要打开的章节（例如'打开第六章'），或在左侧文献树中先打开一个章节。"}
            return

        ch_title = ""
        if self.node_client:
            ch_info = await self.node_client.get_literature_by_chapter(ch_num)
            if ch_info:
                ch_title = ch_info.get("title", "") or ch_info.get("name", "") or ""
        if not ch_title:
            ch_title = f"第{ch_num}章"

        yield {"type": _ET.STATUS, "data": f"正在打开{ch_title}..."}
        yield {"type": _ET.TEXT_NAVIGATE, "data": {
            "chapterNumber": ch_num, "chapterTitle": ch_title,
        }}
        yield {"type": _ET.CHUNK, "data": f"已打开{ch_title}。"}

    async def _stream_map_control(self, query: str, thread_id: str,
                                  state: SupervisorState) -> AsyncGenerator[dict, None]:
        """LLM 分类为 map_control 时 — 驱动前端切换地图朝代图层。

        仅处理 set_map_layer。focus_location 走 P5 literature_qa → AgentLoop → focus_map_location MCP 工具。
        """
        import re as _re
        from ..api.events import SSEEventType as _ET

        dynasty = "all"
        d_match = _re.search(r'(唐|宋|元|明|清)', query)
        if d_match:
            dynasty = d_match.group(1)

        dyn_label = dynasty if dynasty != "all" else "全部朝代"
        yield {"type": _ET.STATUS, "data": f"正在显示{dyn_label}地图..."}
        yield {"type": _ET.MAP_LAYER_UPDATE, "data": {
            "dynasty": dynasty, "action": "set",
        }}
        yield {"type": _ET.CHUNK, "data": f"已切换到{dyn_label}地图图层。"}

    async def _stream_location_story(self, query: str, thread_id: str,
                                      state: SupervisorState) -> AsyncGenerator[dict, None]:
        from .location_story import LocationStoryWorkflow
        from ..utils.entity_names import resolve_location_from_query
        location_name, _ = resolve_location_from_query(query)
        wf = LocationStoryWorkflow(
            self.llm, self.retriever, self.prompts,
            node_client=self.node_client,
            checkpointer=None,  # v6: checkpointer 禁用 (LangGraph 版本兼容性)
        )
        async for event in wf.run_stream(location_name, query=query, thread_id=thread_id,
                                         system_context=getattr(self, '_request_system_context', '')):
            if event.get("type") == "done":
                continue
            yield event

    async def _stream_comparison(self, query: str, thread_id: str,
                                  state: SupervisorState) -> AsyncGenerator[dict, None]:
        """仅打开对照视图，不做文本分析 (v11)。

        与 dynasty_compare 的区别:
        - comparison: 只发送 open_split_view -> 前端分栏 -> 用户自行阅读标注
        - dynasty_compare: 调用 MCP 工具 -> 全量 AI 分析 -> 展示演变元素
        """
        from ..api.events import SSEEventType

        left_ch, right_ch = _extract_chapter_numbers(query)
        if left_ch is None or right_ch is None:
            # B-159: 治"零耗时踢皮球"——先分流，不直接反问。
            # 含 UI 词 = 用户确实想打开对照视图但缺章节号 → 反问是合理交互；
            # 无 UI 词 = 分析型 query 被误路由进来 → 转交 literature_qa (AgentLoop)
            # 真正作答（共现检索/关系归纳等工具直接可答），不再要求用户"指定章节"。
            import re as _re
            _ui_word_re = _re.compile(r'打开|并排|分栏|左右|对照视图|拖入')
            if _ui_word_re.search(query):
                yield {"type": SSEEventType.CHUNK, "data": (
                    "我无法确定您想对照哪两个章节。您可以：\n"
                    "1. 指定具体章节，如「对比第4章和第7章」或「对比明代和清代」\n"
                    "2. 或者让我用 AI 分析对比，如「帮我分析清代和现当代的许仙描写差异」"
                )}
                return
            logger.info(f"[comparison] 无章节号且非UI指令，转交 literature_qa 兜底作答: {query[:50]}")
            async for event in self._stream_literature_qa(query, thread_id, state):
                yield event
            return
        if left_ch == right_ch:
            yield {"type": SSEEventType.CHUNK, "data": (
                "请选择两个不同的章节进行对照。"
            )}
            return

        yield {"type": SSEEventType.OPEN_SPLIT_VIEW, "data": {
            "left_chapter_number": left_ch, "right_chapter_number": right_ch,
        }}

        yield {"type": SSEEventType.CHUNK, "data": (
            f"已为您打开第{left_ch}章和第{right_ch}章的左右对照视图，"
            f"您可以开始阅读和标注了。"
            f"需要分析差异时，可以在 Agent 面板中说「分析差异」。"
        )}

    async def _stream_itinerary(self, query: str, thread_id: str,
                                 state: SupervisorState) -> AsyncGenerator[dict, None]:
        """旅行规划（完整版）— 天气 · 酒店 · 交通 · 路线优化 · 流式攻略生成。"""
        from .itinerary import ItineraryPlannerWorkflow
        wf = ItineraryPlannerWorkflow(
            self.llm, self.retriever, self.prompts,
            node_client=self.node_client,
            checkpointer=None,  # v6: checkpointer 禁用 (LangGraph 版本兼容性)
        )
        async for event in wf.run_stream(query, thread_id=thread_id,
                                         system_context=getattr(self, '_request_system_context', '')):
            if event.get("type") == "done":
                continue
            yield event

    # _stream_character 已删除 (U01-05)

    async def _stream_evolution(self, query: str, thread_id: str,
                                state: SupervisorState) -> AsyncGenerator[dict, None]:
        """流式执行文脉演化工作流 (v6)

        evolution_scenario 已由 _classify_intent 从统一分类器获取，无需二次 LLM 调用。
        """
        from .evolution_workbench import EvolutionWorkbench
        from ..utils.entity_names import resolve_entity_from_query

        # 从 state 获取已分类的 evolution 参数
        tracking_target = state.get("evolution_target", "")
        target_type = state.get("evolution_target_type", "location")

        # 如果分类器未提供追踪目标，或者目标不在当前 query 中出现
        # （LLM 可能受对话历史影响给出错误 target），从 query 直接提取
        if not tracking_target or tracking_target == query or tracking_target not in query:
            if tracking_target and tracking_target not in query:
                logger.warning(
                    f"[_stream_evolution] 分类器 target='{tracking_target}' 不在 query 中，"
                    f"回退到实体提取 (query='{query[:60]}')"
                )
            loc, char, etype = resolve_entity_from_query(query)
            if loc != "未指定":
                tracking_target = loc
                target_type = "location"
            elif char:
                tracking_target = char
                target_type = "character"
            else:
                tracking_target = query
                target_type = "location"

        # 每次演化分析使用唯一 thread_id，防止 LangGraph checkpointer 恢复旧会话状态
        evo_thread_id = f"{thread_id}_evo_{uuid.uuid4().hex[:6]}"
        logger.info(f"[_stream_evolution] evo_thread_id={evo_thread_id} (conversation thread={thread_id})")

        wf = EvolutionWorkbench(
            self.llm, self.retriever, self.prompts,
            mcp_registry=self.mcp,
            checkpointer=getattr(self, '_request_cp', self._checkpointer),  # B-002: 启用 checkpointer, 支持用户隔离
            node_client=self.node_client,
        )

        self._active_workbenches[evo_thread_id] = wf
        try:
            count = 0
            async for event in wf.run_stream(
                query, tracking_target=tracking_target,
                target_type=target_type, thread_id=evo_thread_id,
                system_context=getattr(self, '_request_system_context', ''),
            ):
                count += 1
                if event.get("type") == "done":
                    logger.info(f"[_stream_evolution] filtering done event (#{count})")
                    continue
                logger.info(f"[_stream_evolution] yielding event #{count}: type={event.get('type', '?')}")
                yield event
            logger.info(f"[_stream_evolution] total events yielded: {count}")
        finally:
            self._active_workbenches.pop(evo_thread_id, None)

    async def _stream_dynasty_compare(self, query: str, thread_id: str,
                                     state: SupervisorState) -> AsyncGenerator[dict, None]:
        """流式跨朝代对读 (v10) — 执行 AI 对比分析，结果输出到工作台和对话框。
        正则提取失败时降级走 Agent Loop ReAct，让 LLM 自行推理章节号。"""
        from ..api.events import build_sse_event
        left_ch, right_ch = _extract_chapter_numbers(query)
        if left_ch is None or right_ch is None:
            # 正则提取失败 → 降级走 Agent Loop ReAct
            yield {"type": SSEEventType.STATUS, "data": "正在分析您的对比需求..."}
            from ..agent_loop import AgentLoop
            agent = AgentLoop(
                self.llm, self.retriever, self.prompts,
                mcp_registry=self.mcp, node_client=self.node_client,
            )
            async for event in agent.run_stream(
                query, thread_id=thread_id,
                history=_extract_history_from_state(state),
            ):
                yield event
            return

        yield {"type": SSEEventType.STATUS, "data": f"正在对比第{left_ch}章与第{right_ch}章..."}

        if not self.mcp:
            yield {"type": SSEEventType.ERROR, "data": {"code": "NO_MCP", "message": "MCP registry 不可用"}}
            return

        try:
            tool_result = await self.mcp.call_tool("dynasty_compare_elements", {
                "left_chapter_number": left_ch,
                "right_chapter_number": right_ch,
                "query": query,
            })
            result_data = tool_result.get("result", tool_result)
        except (RuntimeError, ValueError, ConnectionError) as e:
            yield {"type": SSEEventType.ERROR, "data": {"code": "TOOL_ERROR", "message": f"跨朝代对比失败: {e}"}}
            return

        if result_data.get("error"):
            yield {"type": SSEEventType.ERROR, "data": {"code": "COMPARE_ERROR", "message": result_data["error"]}}
            return

        # 工作台：创建对比标签页（DynastyCompareView 自带内部分栏）
        yield build_sse_event(SSEEventType.DYNASTY_COMPARE, **result_data)
        yield {"type": SSEEventType.STATUS, "data": "结构化对比面板已生成，正在进行深度分析..."}

        # 对话框：委托 ReAct agent 生成深度分析（复用 literature_qa 范式）。
        # 机械拼接元素列表的回答过于单薄——agent 可自由调用 get_chapter_full_text /
        # entity_appearances / find_shared_passages 等工具交叉取证，给出有原文引用的分析。
        from ..agent_loop import AgentLoop
        agent = AgentLoop(
            self.llm, self.retriever, self.prompts,
            mcp_registry=self.mcp,
            node_client=self.node_client,
            user_id=getattr(self, '_request_user_id', ''),
            system_context=getattr(self, '_request_system_context', ''),
        )
        el_names = "、".join(
            el.get("elementName", "") for el in result_data.get("elements", [])[:12]
        )
        enhanced_query = (
            f"{query}\n\n"
            f"[系统备注] 已为你执行 dynasty_compare_elements（第{left_ch}章 vs 第{right_ch}章），"
            f"结构化对比面板（元素：{el_names}）已在工作台展示。"
            f"请勿重复执行该工具、勿罗列面板数据；请在对话框中针对用户问题的焦点"
            f"（如本例'雷峰塔的描写'）做深度分析：调用 get_chapter_full_text 阅读两章原文、"
            f"用 entity_appearances / search_literature 定位相关描写段落、"
            f"必要时用 find_shared_passages 找文本传承证据，"
            f"引用原文句子（注明第X章第N段）支撑结论。"
        )
        history = _extract_history_from_state(state)
        try:
            async for event in agent.run_stream(enhanced_query, history=history, thread_id=thread_id):
                if event.get("type") == "done":
                    continue
                yield event
        except Exception as e:
            logger.exception(f"[Supervisor] dynasty_compare 深度分析流异常: {e}")
            yield {"type": SSEEventType.ERROR,
                   "data": {"message": f"深度分析过程中断（{str(e)[:120]}）。结构化对比面板已生成，可重试对话。"}}

    # _try_fast_tool_path 已删除 (U01-08)
    # 快路径逻辑已移入 AgentLoop._try_fast_answer() — Phase 2

    # ===== 辅助 =====

    def _extract_query(self, state: SupervisorState) -> str:
        """从 state 中提取用户查询文本。"""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == "human":
                return msg.content
            elif isinstance(msg, dict) and msg.get("role") in ("user", "human"):
                return msg.get("content", "")
        return ""

    # ===== v10 直通路由 (BU-05b) =====

    async def _stream_passthrough(self, passthrough: dict, query: str,
                                  thread_id: str) -> AsyncGenerator[dict, None]:
        """v10: 前端程序化请求直通路由 — 绕过 LLM 规划，直接调用工具并 yield SSE 事件。"""
        from ..api.events import build_sse_event
        intent = passthrough["intent"]

        if intent == "get_location_traceability":
            location_names = passthrough.get("location_names")
            yield {"type": SSEEventType.STATUS, "data": "正在获取地点溯源数据..."}
            if not self.mcp:
                yield {"type": SSEEventType.ERROR, "data": {"code": "NO_MCP", "message": "MCP registry 不可用"}}
                return
            try:
                params = {}
                if location_names:
                    params["location_names"] = location_names
                tool_result = await self.mcp.call_tool("get_location_traceability", params)
                result_data = tool_result.get("result", tool_result)
                yield build_sse_event(SSEEventType.LOCATION_TRACEABILITY, data=result_data)
            except (RuntimeError, ValueError, ConnectionError) as e:
                yield {"type": SSEEventType.ERROR, "data": {"code": "TOOL_ERROR", "message": str(e)}}

        elif intent == "generate_mutual_construction":
            location_name = passthrough.get("location_name", "")
            if not location_name:
                yield {"type": SSEEventType.ERROR, "data": {"code": "INVALID_PARAMS", "message": "缺少 location_name 参数"}}
                return
            yield {"type": SSEEventType.STATUS, "data": f"正在生成{location_name}互构分析..."}
            if not self.mcp:
                yield {"type": SSEEventType.ERROR, "data": {"code": "NO_MCP", "message": "MCP registry 不可用"}}
                return
            try:
                tool_result = await self.mcp.call_tool("generate_mutual_construction", {
                    "location_name": location_name,
                })
                result_data = tool_result.get("result", tool_result)
                yield build_sse_event(SSEEventType.MUTUAL_CONSTRUCTION, **result_data)
            except (RuntimeError, ValueError, ConnectionError) as e:
                yield {"type": SSEEventType.ERROR, "data": {"code": "TOOL_ERROR", "message": str(e)}}

_CHINESE_NUM_MAP = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _parse_single_chapter_number(ch_str: str) -> int | None:
    """将单个章节号字符串（中文数字或阿拉伯数字）转为整数。"""
    stripped = ch_str.strip()
    if not stripped:
        return None
    if stripped.isdigit():
        return int(stripped)
    if stripped in _CHINESE_NUM_MAP:
        return _CHINESE_NUM_MAP[stripped]
    return None


def _detect_direct_command(query: str) -> dict | None:
    """检测自然语言直接命令（仅用于 C-6 retry 兜底，不再作为路由拦截层）。

    当 LLM 将明显 UI 命令误分类为 literature_qa 时，触发重试。
    返回 None 表示不是直接命令。
    """
    import re as _dre

    # 剥离系统上下文，防止 `[系统上下文: 用户正在查看第4章...]` 被误匹配
    clean_query = _dre.sub(r'\[系统上下文[^\]]*\]', '', query).strip()
    if not clean_query:
        return None

    # ---- 地图操作 ----
    # "打开地图" / "显示地图" / "查看地图"
    if _dre.search(r'(打开|显示|查看|切换到).{0,2}地图', clean_query):
        dynasty = "all"
        d_match = _dre.search(r'(唐|宋|元|明|清).{0,4}(?:的|朝)', clean_query)
        if d_match:
            dynasty = d_match.group(1)
        return {"intent": "map_control", "action": "set_map_layer", "dynasty": dynasty}

    # "定位到X" / "飞到X" / "聚焦X"
    loc_match = _dre.search(r'(?:定位到|飞到|聚焦|导航到|跳转到)\s*(.+?)(?:[。，,\.\s]|$)', clean_query)
    if loc_match:
        loc_name = loc_match.group(1).strip()
        if loc_name and len(loc_name) <= 10:
            return {"intent": "map_control", "action": "focus_location", "location_name": loc_name}

    # ---- 分栏对照 ----
    if _dre.search(r'(打开|左右|分栏|并排).{0,2}(对照|对比|并排|分栏)', clean_query):
        left_ch, right_ch = _extract_chapter_numbers(clean_query)
        if left_ch and right_ch:
            return {"intent": "split_view", "left_chapter": left_ch, "right_chapter": right_ch}

    # ---- 打开单个章节 ----
    # "打开第六章" / "帮我打开第6章" / "查看第3回"
    ch_single = _dre.search(r'(?:打开|查看|浏览|读|进入).{0,4}第\s*([\d一二三四五六七八九十]+)\s*(?:章|回)', clean_query)
    if ch_single:
        ch_num = _parse_single_chapter_number(ch_single.group(1))
        if ch_num:
            return {"intent": "open_chapter", "chapter_number": ch_num}

    return None


def _detect_passthrough_intent(query: str) -> dict | None:
    """v10: 检测前端程序化请求的直通意图。

    前端通过 SSE 通道发送的 JSON 格式系统消息:
      {"intent": "get_location_traceability", "location_names": null}
      {"intent": "generate_mutual_construction", "location_name": "雷峰塔"}

    返回 None 表示不是程序化请求，应走正常 LLM 规划流程。
    """
    passthrough_intents = {"get_location_traceability", "generate_mutual_construction"}
    stripped = query.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            data = json.loads(stripped)
            intent = data.get("intent", "")
            if intent in passthrough_intents:
                return data
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _extract_chapter_numbers(query: str) -> tuple[int | None, int | None]:
    """从用户查询中提取两个章节编号。

    支持模式:
    - "对比明代和清代" → 提取朝代 → 映射章节编号
    - "对比第4章和第7章" → 直接提取数字
    - "对比现当代和清代" → 提取"现当代"和"清" → 映射
    - 有歧义时返回 None（由调用方追问用户）
    """
    import re

    # 朝代 → 章节编号映射（基于白蛇传文献选集.txt 实际章节结构）
    dynasty_map = {
        "唐": [1],
        "宋": [2],
        "明": [3],
        "清": [4, 5, 6],
        "现当代": [7],
    }

    # 先尝试直接提取数字形式的章节编号
    numbers = re.findall(r"第\s*(\d+)\s*章", query)
    if len(numbers) >= 2:
        try:
            return int(numbers[0]), int(numbers[1])
        except ValueError:
            pass

    # 尝试从朝代名提取（含别名，按在查询中出现的位置排序确保左右顺序）
    dynasty_aliases = [
        (["现当代", "现代", "当代"], "现当代"),
        (["唐", "唐代", "唐朝"], "唐"),
        (["宋", "宋代", "宋朝"], "宋"),
        (["明", "明代", "明朝"], "明"),
        (["清", "清代", "清朝"], "清"),
    ]
    found_dynasties = []
    for aliases, dynasty_name in dynasty_aliases:
        for alias in aliases:
            pos = query.find(alias)
            if pos >= 0:
                chapters = dynasty_map[dynasty_name]
                found_dynasties.append((dynasty_name, chapters[0], pos))
                break  # 每个朝代只匹配一次

    # 按在查询中出现的先后位置排序（确保左=先提到的，右=后提到的）
    found_dynasties.sort(key=lambda x: x[2])

    if len(found_dynasties) >= 2:
        return found_dynasties[0][1], found_dynasties[1][1]

    # 单数字 + 单朝代
    if len(numbers) == 1 and len(found_dynasties) == 1:
        return found_dynasties[0][1], int(numbers[0])

    return None, None


def _extract_history_from_state(state: dict) -> list[dict] | None:
    """从 SupervisorState 中提取对话历史，转为 AgentLoop 期望的 [{role, content}] 格式。

    排除最后一条消息（当前用户 query），只返回之前的对话历史。
    """
    messages = state.get("messages", [])
    if not messages or len(messages) <= 1:
        return None

    history = []
    # 排除最后一条（当前 query），只取之前的消息
    for msg in messages[:-1]:
        role = ""
        content = ""
        if hasattr(msg, 'type'):
            if msg.type == "human":
                role = "user"
                content = msg.content
            elif msg.type == "ai":
                role = "assistant"
                content = msg.content[:500]
        elif isinstance(msg, dict):
            r = msg.get("role", "")
            if r in ("user", "human"):
                role = "user"
                content = msg.get("content", "")
            elif r in ("assistant", "ai"):
                role = "assistant"
                content = msg.get("content", "")[:500]
        if role and content:
            history.append({"role": role, "content": content})

    return history if history else None
