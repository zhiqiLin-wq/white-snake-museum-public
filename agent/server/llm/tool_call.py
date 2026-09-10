"""工具调用相关的数据结构 — LLM ↔ Tool 之间的类型安全接口。

使用方式:
    from server.llm.tool_call import ToolDefinition, ToolCallRequest, ToolCallResult
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolDefinition:
    """工具定义 — 传递给 LLM 的 tool schema。"""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema 格式的参数定义

    def to_openai_format(self) -> dict:
        """转为 OpenAI 兼容的 function definition 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def to_anthropic_format(self) -> dict:
        """转为 Anthropic 兼容的 tool definition 格式。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass
class ToolCallRequest:
    """LLM 返回的工具调用请求。"""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallResult:
    """工具执行后的结果。"""
    call_id: str
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    summary: str = ""  # 简短摘要，供 LLM 下一轮决策使用
    sse_events: list[dict] = field(default_factory=list)  # B-082: 额外 SSE 事件
    # v19.2: 未截断的完整原始结果。result 可能被 ToolDispatcher 截断以保护 LLM
    # 上下文；REPORT_READY 等需要完整数据的 SSE 事件必须用此字段，
    # 否则前端拿到的长报告 content 是截断版（曾导致下载/持久化内容缺失）。
    full_result: Any = None


@dataclass
class GenerateWithToolsResult:
    """generate_with_tools() 的统一返回类型。

    可能有两种情况:
    - 纯文本回复 (text 非空, tool_calls 为空): LLM 选择直接回答
    - 工具调用请求 (tool_calls 非空): LLM 决定调用工具
    """
    text: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    model: str = ""
    tokens: int = 0
    # v17.1: 纯输出 tokens（completion）——TokenBudget 账本只记输出
    completion_tokens: int = 0
    finish_reason: str = ""  # "stop" | "tool_calls" | "length" | "error"
    # DeepSeek thinking 模式: 多轮工具调用时必须回传上一轮 assistant 的
    # reasoning_content，否则 API 报 400 ("must be passed back to the API")
    reasoning_content: str = ""
    # DeepSeek thinking 模式下 completion_tokens 包含 reasoning_tokens（思维链）。
    # 账本只记"可见输出"部分，须把思维链 token 单独剔除，否则 12 轮 ReAct 决策
    # 的思考会凭空榨干 MAX_TOKENS_BUDGET，导致 LLM 被迫提前停止写报告（B-167）。
    reasoning_tokens: int = 0

    def to_assistant_message(self) -> dict:
        """转为可回传给 API 的 assistant 消息（协议细节封装在此，业务层不手拼）。

        涵盖 DeepSeek thinking 模式的回传要求: 有 reasoning_content 时
        必须随 assistant 消息一并回传，否则下一轮请求 400。
        """
        import json as _json

        msg: dict = {"role": "assistant"}
        if self.tool_calls:
            msg["content"] = None
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name,
                              "arguments": _json.dumps(tc.arguments, ensure_ascii=False)}}
                for tc in self.tool_calls
            ]
        else:
            msg["content"] = self.text
        if self.reasoning_content:
            msg["reasoning_content"] = self.reasoning_content
        return msg
