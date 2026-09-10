"""LLM 抽象接口，含 tenacity 重试机制 + 工具调用支持。

LLM trace 埋点：在 generate() 内部自动记录全量输入输出到 Tracer。
子类的 generate() 需调用 _record_llm_trace() 上报，或包裹 _generate_with_retry。
"""
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .openai_compatible_provider import StreamResult

from .tool_call import ToolDefinition, GenerateWithToolsResult

logger = logging.getLogger(__name__)


def _record_llm_trace(
    model: str,
    system: str,
    user: str,
    response_content: str,
    tokens_total: int = 0,
    latency_ms: float = 0.0,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    cached: bool = False,
    degraded: bool = False,
    error: Optional[str] = None,
):
    """上报一次 LLM 调用到 Tracer（无 Trace 时静默跳过，不影响业务）。

    供各 Provider 子类在 generate() 成功/失败后调用。
    """
    try:
        from ..observability.tracer import Tracer
        Tracer.record_llm_call(
            model=model,
            system_prompt=system,
            user_prompt=user,
            response=response_content,
            tokens_total=tokens_total,
            latency_ms=latency_ms,
            temperature=temperature,
            max_tokens=max_tokens,
            cached=cached,
            degraded=degraded,
            error=error,
        )
    except Exception:
        # trace 失败绝不能影响业务
        pass


@dataclass
class LLMResponse:
    content: str
    model: str
    tokens: int
    degraded: bool = False
    # v17.1: 纯输出 tokens（completion）——TokenBudget 账本只记输出，
    # 输入上下文由 CONTEXT_GUARDRAIL_TOKENS 护栏单独管辖
    completion_tokens: int = 0
    # v17.2: 生成结束原因（stop/length/tool_calls/content_filter）。
    # == "length" 即命中 max_tokens 上限被静默截断，上层应 WARN + 追加提示用户。
    finish_reason: str = "stop"
    # DeepSeek thinking 模式下 completion_tokens 含 reasoning_tokens（思维链）。
    # 账本只记"可见输出"部分，须把思维链 token 单独剔除，否则多轮 ReAct 决策
    # 的思考会凭空榨干 MAX_TOKENS_BUDGET（B-167）。
    reasoning_tokens: int = 0


class LLMError(Exception):
    """LLM 基础错误。"""
    pass


class RateLimitError(LLMError):
    """限流错误（可重试）。"""
    pass


class APITimeoutError(LLMError):
    """超时错误（可重试）。"""
    pass


class LLMFatalError(LLMError):
    """不可恢复错误（不重试）。"""
    pass


class LLMProvider(ABC):
    """LLM 抽象基类。"""

    @abstractmethod
    async def generate(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        thinking_disabled: bool = True,
    ) -> LLMResponse:
        """生成回复。thinking_disabled=True 时关闭推理/思考模式（如 DeepSeek 非思考）。

        v17: 默认 True（默认安全策略）—— 快速、稳定、小预算不被思考吃光。
        需要推理的调用点显式传 False。
        """
        ...

    @abstractmethod
    async def generate_stream(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        thinking_disabled: bool = True,
    ) -> "StreamResult":
        """流式生成回复。

        统一契约：协程 await 后返回 StreamResult（异步迭代器，可
        ``async for text in stream`` 读取 content，耗尽后读
        ``stream.finish_reason``）。调用方式必须是
        ``async for text in await llm.generate_stream(...)``。

        注意：不要把本方法实现成 async generator（内部 yield）——
        那会导致 await 调用点报 "'async_generator' object can't be
        used in 'await' expression"，两种流式签名并存曾引发
        "'async for' requires an object with __aiter__ method, got coroutine"。
        """
        ...

    @abstractmethod
    async def generate_with_tools(
        self,
        system: str,
        messages: list[dict],
        tools: list[ToolDefinition],
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        thinking_disabled: bool = True,
    ) -> GenerateWithToolsResult:
        """带工具调用的生成 — LLM 可选择回复文本或调用工具。

        Args:
            system: 系统提示词
            messages: 对话历史 [{"role": "user"|"assistant"|"tool", "content": ...}, ...]
            tools: 可用工具定义列表
            model: 模型名 (None = 默认)
            max_tokens: 最大输出 tokens
            temperature: 采样温度

        Returns:
            GenerateWithToolsResult: 文本回复 或 工具调用请求列表
        """
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """估算 token 数。"""
        ...

    @abstractmethod
    async def close(self):
        """清理资源。"""
        ...
