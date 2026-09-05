"""共享 TypedDict 状态定义 — v8。

B-030: v8 EvolutionWorkbenchState 定义
B-049: 序列化/反序列化辅助函数
"""
from typing import TypedDict, Annotated, Optional
from typing import List, Dict, Any
from langgraph.graph.message import add_messages
import operator


# 阶段 0 — TYP-01: 前端传递的段落信息
class ParagraphInfo(TypedDict):
    """前端传递的段落信息。与前端 ParagraphInfo interface 字段一一对应。"""
    index: int         # 0-based 段落索引
    text: str          # 段落文本（已 trim）


class DocumentRef(TypedDict, total=False):
    content: str
    chunk_id: str       # U06: 新增
    metadata: dict
    score: float


class SourceCitation(TypedDict, total=False):
    chunk_id: str       # U06: 新增
    chapter_number: str
    chapter_title: str
    excerpt: str
    relevance_score: float


# ===== 文献问答 State =====
class LiteratureQAState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    thread_id: str
    query: str
    skill: Optional[str]
    intent: str
    keywords: List[str]
    retrieved_docs: List[dict]
    search_iteration: int
    sufficient: bool
    answer: str
    sources: List[dict]
    tool_calls: List[dict]
    total_tokens_used: int
    degraded: bool
    error: Optional[str]
    system_context: str


# ===== 地标叙事 State =====
class LocationStoryState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    thread_id: str
    query: str
    location_name: str
    location_data: dict
    retrieved_docs: List[dict]
    characters: List[str]
    events: List[dict]
    has_era_shift: bool
    shift_reason: str
    shift_analysis: str
    era_timeline: List[dict]
    cultural_context: str
    narrative: str
    sources: List[dict]
    tool_calls: List[dict]
    total_tokens_used: int
    degraded: bool
    system_context: str


# ===== 旅行规划 State =====
class ItineraryState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    thread_id: str
    query: str
    destination_city: str
    travel_dates: str
    duration_days: int
    interests: List[str]
    budget_level: str
    travel_style: str
    attractions: List[dict]
    location_data: dict
    retrieved_docs: List[dict]
    weather_advice: str
    hotel_advice: str
    transportation_plan: str
    optimized_route: List[dict]
    practical_tips: str
    itinerary: str
    sources: List[dict]
    total_tokens_used: int
    degraded: bool
    system_context: str


# ===== 对比分析 State =====
class ComparisonState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    thread_id: str
    query: str
    topics: List[str]
    chapters: List[str]
    aspect: str
    compare_dimensions: List[str]
    era_range: List[str]
    comparison_type: str
    per_topic_docs: dict
    contrasts: List[dict]
    comparison_result: str
    sources: List[dict]
    tool_calls: List[dict]
    total_tokens_used: int
    degraded: bool


# ========================================================================
#  v8 EvolutionWorkbench 子类型
# ========================================================================

class MotifV8(TypedDict):
    """叙事母题 (v8 E1 输出)。"""
    id: str
    name: str
    type: str
    core: str               # invariant core
    variations: list[str]   # variable surface


class MotifMatrixCellV8(TypedDict):
    """母题矩阵单元格 (v8 E2 输出)。"""
    motif_id: str
    dynasty: str
    occurrence_count: int
    documents: list[dict]
    summary: str


class ChangeRecordV8(TypedDict):
    """变化记录 (v8 E3 输出)。"""
    motif_id: str
    dynasty_from: str
    dynasty_to: str
    change_type: str        # 7 种类型之一
    confidence: str
    explanation: str


class PhylogenyDataV8(TypedDict):
    """谱系网络 (v8 E4 输出)。"""
    nodes: list[dict]
    edges: list[dict]


class CorrelationV8(TypedDict):
    """外部关联 (v8 E5 输出)。"""
    change: str
    external_event: str
    relevance: str
    source_url: str


class HypothesisV8(TypedDict):
    """候选假设 (v8 E6 输出)。"""
    id: str
    statement: str
    confidence: int         # 1-5
    rationale: str


class ClaimV8(TypedDict):
    """可检验的主张 (v8 E7 输出)。"""
    claim_id: str
    statement: str
    evidence_pro: list
    evidence_con: list
    verdict: str            # "supported" | "insufficient" | "refuted"
    confidence: str
    gap_detected: bool


class DecisionOptionV8(TypedDict):
    """中断决策选项 (v8 Interrupt #3)。"""
    id: str
    label: str
    description: str
    action: str             # "investigate" | "reexplore" | "accept" | "change_hypothesis"


# ========================================================================
#  v8 EvolutionWorkbenchState
# ========================================================================

class EvolutionWorkbenchState(TypedDict, total=False):
    """文脉演化工作台 v8 全局状态。

    v6->v8 变更:
    - 删除 scenario, active_dimensions, compare_target
    - 删除 temporal_results, spatial_results, explore_findings, serendipity_findings
    - 删除 scholar_selection, evolution_map, spatial_network, anomalies
    - 删除 needs_pivot, pivot_reason
    - 新增 E1-E8 各阶段字段
    - 新增中断相关字段
    """
    messages: Annotated[list, add_messages]
    thread_id: str

    # === 输入 ===
    tracking_target: str
    target_type: str            # "location" | "character" | "theme" | "object" | "motif"
    original_query: str

    # === E1: Motif Decomposition ===
    motifs: list[dict]          # MotifV8[]
    suggested_additions: list[str]

    # === E2: Motif Matrix ===
    motif_matrix: list[dict]    # MotifMatrixCellV8[]
    dynasties: list[str]

    # === E3: Change Classification ===
    changes: list[dict]         # ChangeRecordV8[]

    # === E4: Phylogeny ===
    phylogeny: dict   # PhylogenyDataV8 (E4 永远返回 dict，初始为 {})

    # === E5: External Correlation ===
    correlations: list[dict]    # CorrelationV8[]

    # === E6: Hypothesis ===
    candidate_hypotheses: list[dict]  # HypothesisV8[]
    selected_hypothesis_ids: list[str]

    # === E7: Claim Testing ===
    claims: list[dict]          # ClaimV8[]
    test_iteration: int
    claim_index: int            # 当前正在验证的主张索引（逐条反馈用）

    # === E7 Interrupt ===
    needs_decision: bool
    decision_options: list[dict]    # DecisionOptionV8[]
    user_decision: Optional[str]
    decision: str                    # E7 resume 注入的用户决策

    # === Resume transient fields (Command(update=...) 注入) ===
    edited_motifs: list[dict]        # E1 resume 注入的编辑后母题
    action: str                      # resume action 类型

    # === E7 Reexplore tracking ===
    reexplore_count: int                # reexplore 次数 (用于变换检索策略)
    previous_hypotheses: list[dict]     # 上一轮生成的假设 (用于让 LLM 生成不同假设)

    # === E8: Synthesis ===
    evolution_narrative: str
    evidence_refs: dict    # 叙事引用的原文证据映射，前端点击引用标签时查找

    # === Control ===
    current_phase: str
    errors: list[str]
    pending_events: list[dict]      # SSE 事件缓冲（每节点替换，不累加）
    system_context: str             # 记忆上下文（摘要 + Core Memory + 被动注入）


# ========================================================================
#  B-049: 序列化/反序列化辅助函数
# ========================================================================

def evolution_state_to_dict(state: EvolutionWorkbenchState) -> dict:
    """将 state 转为可 JSON 序列化的 dict（用于 SSE 推送）。"""
    result = {}
    for k, v in state.items():
        if k == "messages":
            continue
        result[k] = v
    return result


def dict_to_evolution_state(d: dict) -> EvolutionWorkbenchState:
    """从 resume 请求体重建 state（用于 Command(resume=...)）。"""
    state: EvolutionWorkbenchState = {}
    state.update(d)
    return state
