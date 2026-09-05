"""统一的 Agent 响应/事件 Pydantic 模型。

包含:
- AgentResponse:     非流式对话响应
- AgentStreamEvent:  单个 SSE 事件 (流式)
- WorkflowResult:    子图执行结果 (内部使用)
- ToolCallRecord:    工具调用记录
- SourceCitation:    文献引用
"""

from typing import Optional, Any
from pydantic import BaseModel, Field


# ============================================================
#  文献引用
# ============================================================
class SourceCitation(BaseModel):
    chapter_number: str = ""
    chapter_title: str = ""
    excerpt: str = ""
    relevance_score: float = 0.0


# ============================================================
#  工具调用记录
# ============================================================
class ToolCallRecord(BaseModel):
    tool: str
    input: dict = Field(default_factory=dict)
    status: str = "running"  # "running" | "completed" | "error"
    result_preview: Optional[str] = None


# ============================================================
#  子图执行结果 (内部使用)
# ============================================================
class WorkflowResult(BaseModel):
    """所有子图 run() 方法的统一返回类型。"""
    answer: str = ""
    sources: list[SourceCitation] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    workflow_used: str = ""
    intent: str = ""
    intent_label: str = ""
    degraded: bool = False
    total_tokens_used: int = 0

    # 扩展字段 (子图特有数据)
    extras: dict[str, Any] = Field(default_factory=dict)


# ============================================================
#  非流式对话响应 (对外 API)
# ============================================================
class AgentResponse(BaseModel):
    """POST /chat (stream=false) 的响应。"""
    answer: str
    sources: list[SourceCitation] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    workflow_used: str = ""
    conversation_id: Optional[str] = None
    degraded: bool = False

    # Evolution 专用
    evolution_map: Optional[dict] = None

    # Itinerary 专用
    weather_advice: Optional[str] = None
    hotel_advice: Optional[str] = None
    transportation_plan: Optional[str] = None
    optimized_route: Optional[list[dict]] = None


# ============================================================
#  SSE 流式单事件
# ============================================================
class AgentStreamEvent(BaseModel):
    """单个 SSE 事件的结构化表示。"""
    type: str
    data: Any = None
