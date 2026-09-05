"""Anthropic Claude 实现。"""
import asyncio
import json
import logging
from typing import Optional, AsyncGenerator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

import anthropic as _anthropic

from .provider import (
    LLMProvider,
    LLMResponse,
    LLMFatalError,
    RateLimitError,
    APITimeoutError,
)
from .openai_compatible_provider import StreamResult
from .tool_call import ToolDefinition, ToolCallRequest, GenerateWithToolsResult
from ..config import settings
from ..data.llm_cache import LLMResponseCache

logger = logging.getLogger(__name__)
_llm_cache = LLMResponseCache(ttl_seconds=3600)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM 提供者。"""

    def __init__(self):
        self._client = None
        self._async_client = None
        # 后台流式生产任务强引用（防止 create_task 出的任务被 GC）
        self._stream_tasks: set = set()

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        return self._client

    def _get_async_client(self):
        if self._async_client is None:
            import anthropic
            self._async_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        return self._async_client

    def count_tokens(self, text: str) -> int:
        chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    async def generate(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        thinking_disabled: bool = True,
    ) -> LLMResponse:
        from .provider import _record_llm_trace
        import time as _time

        model = model or settings.anthropic_model

        # S-03: Check LLM response cache
        cache_key = _llm_cache.cache_key(system, user, model, max_tokens, temperature)
        cached = _llm_cache.get(cache_key)
        if cached is not None:
            logger.debug(f"LLM cache hit for key {cache_key[:12]}...")
            _record_llm_trace(
                model=model, system=system, user=user,
                response_content=cached,
                tokens_total=self.count_tokens(cached),
                cached=True, temperature=temperature, max_tokens=max_tokens,
            )
            return LLMResponse(
                content=cached, model=model, tokens=self.count_tokens(cached),
                completion_tokens=self.count_tokens(cached),
            )

        _t0 = _time.monotonic()
        try:
            response = await self._generate_with_retry(system, user, model, max_tokens, temperature)
            _latency_ms = (_time.monotonic() - _t0) * 1000
            _record_llm_trace(
                model=model, system=system, user=user,
                response_content=response.content,
                tokens_total=response.tokens,
                latency_ms=_latency_ms,
                temperature=temperature, max_tokens=max_tokens,
            )
            # S-03: Store in cache
            _llm_cache.set(cache_key, response.content)
            return response
        except Exception as e:
            _latency_ms = (_time.monotonic() - _t0) * 1000
            _record_llm_trace(
                model=model, system=system, user=user,
                response_content="",
                latency_ms=_latency_ms,
                temperature=temperature, max_tokens=max_tokens,
                error=f"{type(e).__name__}: {str(e)[:200]}",
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
        reraise=True,
    )
    async def _generate_with_retry(
        self,
        system: str,
        user: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        try:
            client = self._get_async_client()
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            content = response.content[0].text if response.content else ""
            return LLMResponse(
                content=content,
                model=model,
                tokens=response.usage.input_tokens + response.usage.output_tokens,
                completion_tokens=response.usage.output_tokens,
            )
        except _anthropic.RateLimitError as e:
            logger.warning(f"Anthropic 限流: {e}")
            raise RateLimitError(str(e))
        except _anthropic.APITimeoutError as e:
            logger.warning(f"Anthropic 超时: {e}")
            raise APITimeoutError(str(e))
        except (_anthropic.APIStatusError, _anthropic.APIConnectionError, _anthropic.APIResponseValidationError) as e:
            logger.error(f"Anthropic 错误: {e}")
            raise LLMFatalError(str(e))

    async def generate_stream(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        thinking_disabled: bool = True,
    ) -> "StreamResult":
        """流式生成 — 与 OpenAI 兼容版签名统一：await 后返回 StreamResult。

        用法::

            stream = await llm.generate_stream(sys, user)
            async for text in stream:
                ...
        重试/降级在后台生产任务中执行，失败时以降级文本 emit，
        不向上抛异常（与 OpenAI 兼容版行为一致）。
        """
        result = StreamResult()

        async def _produce() -> None:
            try:
                async for text in self._generate_stream_iter(
                    system, user, model, max_tokens, temperature
                ):
                    result._emit(text)
            except Exception as e:  # noqa: BLE001 — 流式路径统一降级，与 openai 版对齐
                result._emit(f"\n\n（AI 响应中断: {str(e)[:100]}。请稍后重试或缩短问题。）")
            finally:
                result._done_mark()

        task = asyncio.create_task(_produce())
        self._stream_tasks.add(task)
        task.add_done_callback(self._stream_tasks.discard)
        return result

    async def _generate_stream_iter(
        self,
        system: str,
        user: str,
        model: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[str, None]:
        """内部流式生成器（含重试 + 优雅降级），由 generate_stream 的后台任务消费。"""
        model = model or settings.anthropic_model

        last_error = None
        for attempt in range(3):  # 最多重试3次
            try:
                client = self._get_async_client()
                async with client.messages.stream(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                ) as stream:
                    async for text in stream.text_stream:
                        yield text
                return  # 成功完成，退出重试循环
            except (_anthropic.RateLimitError, _anthropic.APITimeoutError) as e:
                last_error = e
                if attempt < 2:
                    wait_time = (2 ** attempt) * 1.5
                    logger.warning(f"流式生成中断 (尝试 {attempt+1}/3)，{wait_time}s 后重试: {e}")
                    import asyncio
                    await asyncio.sleep(wait_time)
                    # 后续重试降级到 Haiku（更快更稳定）
                    if attempt >= 1:
                        model = settings.anthropic_fast_model
                        logger.info(f"降级到快速模型: {model}")
                else:
                    break  # 已达重试上限
            except (_anthropic.APIStatusError, _anthropic.APIConnectionError, _anthropic.APIResponseValidationError) as e:
                last_error = e
                logger.warning(f"流式生成失败 (尝试 {attempt+1}/3): {e}")
                break  # 非可重试错误，不重试

        # 所有重试都失败 → 优雅降级
        logger.error(f"流式生成最终失败: {last_error}")
        yield f"\n\n（AI 响应中断: {str(last_error)[:100]}。请稍后重试或缩短问题。）"

    async def generate_with_fallback(
        self,
        system: str,
        user: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """带降级的生成：主模型 → 快速模型 → 空回答。"""
        # 尝试主模型
        try:
            return await self.generate(system, user, settings.anthropic_model, max_tokens, temperature)
        except LLMFatalError as e:
            logger.error(f"主模型不可用: {e}")
        except (RateLimitError, APITimeoutError) as e:
            logger.warning(f"主模型暂时不可用: {e}")

        # 降级到快速模型
        try:
            logger.info("降级到 Haiku 快速模型...")
            return await self.generate(system, user, settings.anthropic_fast_model, max_tokens, temperature)
        except (LLMFatalError, RateLimitError, APITimeoutError) as e:
            logger.critical(f"所有 LLM 不可用: {e}")

        # 完全不可用
        return LLMResponse(
            content="（AI 服务繁忙，以下为检索到的相关文献片段，仅供参考）",
            model="none",
            tokens=0,
            degraded=True,
        )

    # ===== 工具调用 (Tool Calling) =====

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
        """带工具调用的生成 — Anthropic 原生 tool_use 支持。

        thinking_disabled 与 OpenAI 兼容层签名对齐; Anthropic 走原生
        tool_use（无 DeepSeek 式思考回传约束），当前不消费该参数。
        """
        model = model or settings.anthropic_model

        anthropic_tools = [t.to_anthropic_format() for t in tools]

        try:
            client = self._get_async_client()
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                tools=anthropic_tools,
                messages=messages,
            )

            tokens = response.usage.input_tokens + response.usage.output_tokens
            completion_tokens = response.usage.output_tokens
            text_parts = []
            tool_calls = []

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(ToolCallRequest(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    ))

            if tool_calls:
                return GenerateWithToolsResult(
                    text="\n".join(text_parts) if text_parts else "",
                    tool_calls=tool_calls,
                    model=model,
                    tokens=tokens,
                    completion_tokens=completion_tokens,
                    finish_reason="tool_calls",
                )
            return GenerateWithToolsResult(
                text="\n".join(text_parts),
                model=model,
                tokens=tokens,
                completion_tokens=completion_tokens,
                finish_reason=response.stop_reason or "stop",
            )

        except _anthropic.RateLimitError as e:
            logger.warning(f"Anthropic tool_call 限流: {e}")
            raise RateLimitError(str(e))
        except _anthropic.APITimeoutError as e:
            logger.warning(f"Anthropic tool_call 超时: {e}")
            raise APITimeoutError(str(e))
        except (_anthropic.APIStatusError, _anthropic.APIConnectionError, _anthropic.APIResponseValidationError) as e:
            logger.error(f"Anthropic tool_call 错误: {e}")
            raise LLMFatalError(str(e))

    async def close(self):
        if self._async_client:
            await self._async_client.close()
        if self._client:
            self._client.close()
