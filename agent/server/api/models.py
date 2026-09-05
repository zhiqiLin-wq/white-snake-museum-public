"""Pydantic 请求/响应数据模型。"""
from typing import Optional, List
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., max_length=5000, description="用户问题")
    skill: Optional[str] = Field(None, description="指定技能名")
    stream: bool = Field(True, description="是否流式返回")
    conversation_id: Optional[str] = Field(None, description="会话 ID")
    new_session: bool = Field(False, description="强制开启新会话")
    history: Optional[List[dict]] = Field(None, description="历史对话消息列表 [{role, content}, ...]")
    context_tags: Optional[List[dict]] = Field(None, description="前端上下文标签（含章节/段落/标注信息）")
    split_side: Optional[str] = Field(None, description="Split View 分栏侧: left/right")
    existing_user_annotations: Optional[List[str]] = Field(None, description="已有用户标注ID列表")
    paragraphs: Optional[List[dict]] = Field(
        None,
        description="段落列表 [{index: int, text: str}]，Agent 标注流水线使用（阶段 0 — API-01a）"
    )


class SourceCitation(BaseModel):
    chapter_number: str
    chapter_title: str
    excerpt: str
    relevance_score: float


class ToolCallRecord(BaseModel):
    tool: str
    input: dict
    status: str  # "running" | "completed" | "error"
    result_preview: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceCitation] = []
    tool_calls: List[ToolCallRecord] = []
    workflow_used: str = ""
    conversation_id: Optional[str] = None
    degraded: bool = False


class SkillSummary(BaseModel):
    name: str
    display_name: str
    description: str
    category: str
    icon: str = "📖"
    triggers: List[str] = []


class SkillListResponse(BaseModel):
    skills: List[SkillSummary]


class HealthResponse(BaseModel):
    status: str  # "ok" | "starting" | "degraded"
    vector_store: str = "unknown"  # "ready" | "building" | "error" | "not_configured"
    llm: str = "unknown"  # "configured" | "missing_key"
    node_api: str = "unknown"  # "connected" | "unreachable"
    embedding_model: Optional[dict] = None
    uptime_seconds: float = 0
    metrics: Optional[dict] = None
    # B-005: Checkpointer 健康信息
    checkpointer: Optional[dict] = None
    # B-126: 数据层健康信息
    data_layer: Optional[dict] = None
    # B-007: 活跃 thread 数
    active_threads: Optional[int] = None
    # U12: 索引构建进度
    build_progress: Optional[dict] = None


class RebuildResponse(BaseModel):
    status: str
    chunks: int = 0
    message: str = ""


# ================================================================
#  B-090: /chat/resume 请求/响应模型
# ================================================================

class ResumePayload(BaseModel):
    action: Optional[str] = None  # "confirm" | "select" | "decide" — 前端可能不传，后端从 interrupt_id 推导
    edited_motifs: Optional[List[dict]] = None
    selected_hypothesis_ids: Optional[List[str]] = None
    decision: Optional[str] = None  # "investigate" | "reexplore" | "accept" | "change_hypothesis"
    selected_option_id: Optional[str] = None  # 前端可能发送此字段代替 decision


class ChatResumeRequest(BaseModel):
    thread_id: str
    interrupt_id: str  # "confirm_motifs" | "select_hypothesis" | "decide_next_action"
    resume: ResumePayload


class CancelRequest(BaseModel):
    thread_id: str
