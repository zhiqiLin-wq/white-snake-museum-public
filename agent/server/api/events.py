"""统一 SSE 事件类型定义 — v8 扩展版。

B-060: 新增工作区/演化/人机协同/进度事件类型
B-061: SSE 事件构建辅助函数

使用方式:
    from server.api.events import SSEEventType
    yield {"type": SSEEventType.CHUNK, "data": text}

禁止在子图中使用裸字符串作为事件类型。
"""
from datetime import datetime, timezone
from enum import Enum


class SSEEventType(str, Enum):
    """统一的 Server-Sent Events 事件类型枚举。

    v8 新增:
    - 工作区驱动: annotation_data, stats_data, comparison_data, search_results, text_navigate, map_highlight
    - 演化工作区: motif_matrix, phylogeny_data, evidence_data, evolution_narrative
    - 人机协同: evolution_interrupt, interrupt_resumed
    - 进度: workflow_done, thinking_step
    """

    # ===== 生命周期 (Lifecycle) =====
    THINKING_START = "thinking_start"
    DONE = "done"
    ERROR = "error"

    # ===== 进度与元信息 (Progress & Meta) =====
    STATUS = "status"
    INTENT = "intent"
    SKILL_ACTIVATED = "skill_activated"
    THINKING_STEP = "thinking_step"       # v8 新增: 单步思考进度
    THINKING_PLAN = "thinking_plan"       # U-02: v11 步骤规划骨架
    THINKING_DELTA = "thinking_delta"     # v17: ReAct 决策轮思考内容实时增量透传
    EXPORT_READY = "export_ready"            # U-16: v11 导出就绪
    WORKFLOW_DONE = "workflow_done"       # v8 新增: 子流程完成

    # ===== 内容 (Content) =====
    CHUNK = "chunk"
    SOURCE = "source"

    # ===== 工具调用 (Tool Use) — ReAct 循环 =====
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RETRY_STEP = "retry_step"            # v11: 用户请求重试失败步骤
    SKIP_STEP = "skip_step"              # v11: 用户请求跳过失败步骤

    # ===== 工作区驱动 (Workspace) — v8 新增 =====
    ANNOTATION_DATA = "annotation_data"
    ANNOTATION_PROGRESS = "annotation_progress"   # 智能标注实时进度（阶段/计数/真实统计）
    ANNOTATION_SUGGEST = "annotation_suggest"     # v9: Agent 建议标注
    MARGINALIA_DATA = "marginalia_data"           # v11: Agent 创建的旁批评注
    PARAGRAPH_ALIGNMENT = "paragraph_alignment"   # v9: 段落对齐结果
    STATS_DATA = "stats_data"
    COMPARISON_DATA = "comparison_data"
    SEARCH_RESULTS = "search_results"
    TEXT_NAVIGATE = "text_navigate"
    MAP_HIGHLIGHT = "map_highlight"
    MAP_LAYER_UPDATE = "map_layer_update"            # v11: Agent 切换地图朝代图层

    # ===== v10 新增: 跨朝代对比 + 地图溯源通道 =====
    DYNASTY_COMPARE = "dynasty_compare"
    LOCATION_TRACEABILITY = "location_traceability"      # v10 修复: 地图溯源卡片 literaryRecords 数据通道
    MUTUAL_CONSTRUCTION = "mutual_construction"          # v10 修复: 景观-文本互构分析数据通道
    OPEN_SPLIT_VIEW = "open_split_view"                 # v11: Agent 打开对照视图（仅分栏不分析）

    # ===== v19 新增: 长报告文件输出通道 =====
    REPORT_READY = "report_ready"                       # 完整长报告已写入文件，前端渲染下载卡片

    # ===== 演化工作区 (Evolution) — v8 新增 =====
    MOTIF_MATRIX = "motif_matrix"
    PHYLOGENY_DATA = "phylogeny_data"
    EVIDENCE_DATA = "evidence_data"
    EVOLUTION_NARRATIVE = "evolution_narrative"

    # ===== 人机协同 (Human-in-the-Loop) — v8 新增 =====
    EVOLUTION_INTERRUPT = "evolution_interrupt"
    INTERRUPT_RESUMED = "interrupt_resumed"

    # ===== B-140: 演化分析结果卡片（发送到对话框） =====
    EVOLUTION_CHAT_CARD = "evolution_chat_card"

    # ===== Evolution Workbench 旧版 (v4.0 - v6) — deprecated =====
    EVOLUTION_START = "evolution_start"             # deprecated: use status + intent
    EVOLUTION_SCENARIO = "evolution_scenario"        # deprecated: use motif_matrix
    EVOLUTION_FINDINGS = "evolution_findings"        # deprecated: use motif_matrix
    EVOLUTION_SERENDIPITY = "evolution_serendipity"  # deprecated: removed in v8
    EVOLUTION_CLAIMS = "evolution_claims"            # deprecated: use evidence_data
    EVOLUTION_TEST_PROGRESS = "evolution_test_progress"  # deprecated: use thinking_step
    EVOLUTION_NODE = "evolution_node"               # deprecated: use phylogeny_data.nodes
    EVOLUTION_TRANSITION = "evolution_transition"   # deprecated: use phylogeny_data.edges
    EVOLUTION_ANOMALY = "evolution_anomaly"         # deprecated: removed in v8
    EVOLUTION_SPATIAL_NETWORK = "evolution_spatial_network"  # deprecated: removed in v8
    EVOLUTION_MAP_HIGHLIGHT = "evolution_map_highlight"  # deprecated: use map_highlight
    EVOLUTION_NEXT_ACTION = "evolution_next_action" # deprecated: use evolution_interrupt
    EVOLUTION_COMPLETE = "evolution_complete"       # deprecated: use workflow_done
    EVOLUTION_ERROR = "evolution_error"             # deprecated: use error
    EVOLUTION_PIVOT_PROMPT = "evolution_pivot_prompt"  # deprecated: removed in v8

    # ===== 向后兼容 (v3.1) =====
    EVOLUTION_LENS_PROMPT = "evolution_lens_prompt"


# 前缀分组
EVOLUTION_EVENT_PREFIX = "evolution_"

# 所有 evolution_* 事件的集合
EVOLUTION_EVENTS: frozenset[str] = frozenset(
    v for v in SSEEventType.__members__.values()
    if v.value.startswith(EVOLUTION_EVENT_PREFIX)
)

# B-068: 工作区事件集合（routes.py 透传不包装）
WORKSPACE_EVENTS: frozenset[str] = frozenset([
    SSEEventType.ANNOTATION_DATA,
    SSEEventType.ANNOTATION_PROGRESS,
    SSEEventType.ANNOTATION_SUGGEST,
    SSEEventType.MARGINALIA_DATA,
    SSEEventType.PARAGRAPH_ALIGNMENT,
    SSEEventType.STATS_DATA,
    SSEEventType.COMPARISON_DATA,
    SSEEventType.SEARCH_RESULTS,
    SSEEventType.MOTIF_MATRIX,
    SSEEventType.PHYLOGENY_DATA,
    SSEEventType.EVIDENCE_DATA,
    SSEEventType.EVOLUTION_NARRATIVE,
    SSEEventType.TEXT_NAVIGATE,
    SSEEventType.MAP_HIGHLIGHT,
    SSEEventType.MAP_LAYER_UPDATE,           # v11: Agent 切换地图朝代图层
    SSEEventType.DYNASTY_COMPARE,
    SSEEventType.LOCATION_TRACEABILITY,     # v10 修复
    SSEEventType.MUTUAL_CONSTRUCTION,       # v10 修复
    SSEEventType.EVOLUTION_INTERRUPT,
    SSEEventType.INTERRUPT_RESUMED,
    SSEEventType.THINKING_PLAN,
    SSEEventType.WORKFLOW_DONE,
    SSEEventType.OPEN_SPLIT_VIEW,           # v11: Agent 打开对照视图
    SSEEventType.EXPORT_READY,              # U-16: v11 导出就绪
])


# ================================================================
#  B-061: SSE 事件构建辅助函数
# ================================================================

# 已知的字段名重命名映射（处理 snake_case→camelCase 无法覆盖的 case）
_FIELD_RENAME_MAP: dict[str, str] = {
    "interrupt_id": "interruptId",
    "interrupt_type": "interruptType",
    "thread_id": "threadId",
    "call_id": "callId",
    "tool_name": "toolName",
    "tracking_target": "trackingTarget",
    "target_type": "targetType",
    "chapter_number": "chapterNumber",
    "chapter_title": "chapterTitle",
    "paragraph_index": "paragraphIndex",
    "relevance_score": "relevanceScore",
    "passage_key": "passageKey",
    "is_final": "isFinal",
    "total_matches": "totalMatches",
    "sort_options": "sortOptions",
    "current_sort": "currentSort",
    "highlight_ranges": "highlightRanges",
    "suggested_additions": "suggestedAdditions",
    "selected_hypothesis_ids": "selectedHypothesisIds",
    "allow_multi_select": "allowMultiSelect",
    "rationale": "reasoning",
    "motif_id": "motifId",
    "dynasty_from": "dynastyFrom",
    "dynasty_to": "dynastyTo",
    "change_type": "changeType",
    "occurrence_count": "occurrenceCount",
    "candidate_hypotheses": "candidateHypotheses",
    "evidence_pro": "evidencePro",
    "evidence_con": "evidenceCon",
    "claim_id": "claimId",
    "claim_index": "claimIndex",
    "gap_detected": "gapDetected",
    "test_iteration": "testIteration",
    "decision_options": "decisionOptions",
    "current_phase": "currentPhase",
    "pending_events": "pendingEvents",
    "evolution_narrative": "evolutionNarrative",
    "motif_matrix": "motifMatrix",
    "problematic_claims": "problematicClaims",
    "result_preview": "resultPreview",
    "total_tokens": "totalTokens",
    "error_message": "errorMessage",
    "split_side": "splitSide",                     # v9: 分栏标注目标侧

    # ---- v10 新增: dynasty_compare 字段映射 ----
    "left_occurrences": "leftOccurrences",
    "right_occurrences": "rightOccurrences",
    "context_note": "contextNote",
    "location_changes": "locationChanges",
    "element_id": "elementId",
    "element_name": "elementName",
    "start_char": "startChar",
    "end_char": "endChar",
    "key_findings": "keyFindings",
    "left_chapter_number": "leftChapterNumber",
    "right_chapter_number": "rightChapterNumber",
    "left_chapter_title": "leftChapterTitle",
    "right_chapter_title": "rightChapterTitle",
    "left_status": "leftStatus",
    "right_status": "rightStatus",
    "left_mentions": "leftMentions",
    "right_mentions": "rightMentions",

    # ---- v10 修复: 地图溯源字段映射 ----
    "literary_records": "literaryRecords",
    "total_mentions": "totalMentions",
    "description_style": "descriptionStyle",
    "description_styles": "descriptionStyles",
    "dynasty_order": "dynastyOrder",
    "mutual_construction": "mutualConstruction",
    "landscape_to_text": "landscapeToText",
    "text_to_landscape": "textToLandscape",
    "related_motifs": "relatedMotifs",
    "location_name": "locationName",
}


def _camelize_key(key: str) -> str:
    """Convert snake_case key to camelCase, with explicit mapping for known fields."""
    if key in _FIELD_RENAME_MAP:
        return _FIELD_RENAME_MAP[key]
    if '_' not in key:
        return key
    parts = key.split('_')
    return parts[0] + ''.join(p.capitalize() for p in parts[1:])


def _camelize_value(value):
    """Recursively camelize all dict keys in a value."""
    if isinstance(value, dict):
        return {_camelize_key(k): _camelize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_camelize_value(item) for item in value]
    return value


def build_sse_event(event_type: SSEEventType, **kwargs) -> dict:
    """构建标准 SSE 事件 dict，自动添加 timestamp 并转换字段名为 camelCase。

    trace_id 关联：自动从 current_trace 注入 traceId，让前端事件可回溯后端日志
    （llm_calls.log / tool_calls.log），实现"前端看到哪步 → 后端定位到哪次调用"。
    """
    camelized = {_camelize_key(k): _camelize_value(v) for k, v in kwargs.items()}
    event = {
        "type": event_type.value,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **camelized,
    }
    # 注入 trace_id，便于前端事件与后端日志串联（无活跃 trace 时省略）
    try:
        from ..observability.tracer import current_trace
        trace = current_trace.get()
        if trace:
            event["traceId"] = trace.trace_id
    except Exception:
        pass
    return event
