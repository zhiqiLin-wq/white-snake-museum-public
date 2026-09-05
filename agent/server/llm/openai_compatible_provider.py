"""通用 OpenAI 兼容 LLM 提供者。

支持所有使用 OpenAI 兼容协议的 API:
  - DeepSeek:  base_url="https://api.deepseek.com/v1"
  - Qwen (通义千问): base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
  - 文心一言 (ERNIE Bot): base_url="https://qianfan.baidubce.com/v2"
  - 其他兼容 OpenAI 协议的服务

Usage:
    provider = GenericOpenAIProvider(
        api_key="sk-xxx",
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-v4-flash",
    )
    response = await provider.generate(system="...", user="...")
"""
import asyncio
import json
import logging
from typing import Optional, AsyncGenerator, Awaitable, Callable

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .provider import (
    LLMProvider,
    LLMResponse,
    LLMFatalError,
    RateLimitError,
    APITimeoutError,
)
from .tool_call import ToolDefinition, ToolCallRequest, GenerateWithToolsResult
from ..data.llm_cache import LLMResponseCache

logger = logging.getLogger(__name__)
_llm_cache = LLMResponseCache(ttl_seconds=3600)


class StreamResult:
    """流式生成结果：既是异步迭代器（yield content 文本），
    又暴露 ``finish_reason`` 字段供迭代结束后读取结束原因。

    用法::

        stream = await llm.generate_stream(sys, user, max_tokens=8192)
        async for text in stream:
            send_to_frontend(text)
        if stream.finish_reason == "length":
            warn(f"回答被截断 at max_tokens")
    """
    # 注意: 不使用 __slots__（__anext__ 是 async 方法，CPython 会用到 _xxx
    # 内部属性，slot 命名冲突会抛 ValueError "_done in __slots__ conflicts"）

    def __init__(self) -> None:
        import asyncio as _aio
        self._queue = _aio.Queue()
        self._done = False
        self.finish_reason: str = "stop"

    def _emit(self, text: str) -> None:
        """生产者（Provider）把下一段文本塞进来。非协程，立刻入队。"""
        self._queue.put_nowait(text)

    def _done_mark(self) -> None:
        """标记结束，异步迭代结束。"""
        self._done = True
        try:
            self._queue.put_nowait(None)  # type: ignore[arg-type]
        except Exception:
            pass

    def __aiter__(self) -> "StreamResult":
        return self

    async def __anext__(self) -> str:
        if self._queue.empty() and self._done:
            raise StopAsyncIteration
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return str(item)


class GenericOpenAIProvider(LLMProvider):
    """通用 OpenAI 兼容协议 LLM 提供者。

    可配置 base_url / api_key / model，适配 DeepSeek、Qwen、文心一言等。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        default_model: str,
        fast_model: Optional[str] = None,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._fast_model = fast_model or default_model
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def fast_model(self) -> str:
        return self._fast_model

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self._client

    def count_tokens(self, text: str) -> int:
        """启发式 token 估算 (中文 ~1.5 字/token，英文 ~4 字/token)。"""
        chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    # ===== 非流式生成 =====

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

        model = model or self._default_model

        cache_key = _llm_cache.cache_key(
            system + ("<NO_THINKING>" if thinking_disabled else ""),
            user, model, max_tokens, temperature,
        )
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
            response = await self._generate_with_retry(
                system, user, model, max_tokens, temperature, thinking_disabled,
            )
            _latency_ms = (_time.monotonic() - _t0) * 1000
            _record_llm_trace(
                model=model, system=system, user=user,
                response_content=response.content,
                tokens_total=response.tokens,
                latency_ms=_latency_ms,
                temperature=temperature, max_tokens=max_tokens,
            )
            _llm_cache.set(cache_key, response.content)
            return response
        except asyncio.CancelledError:
            # 调用方在响应前取消（典型为上层 asyncio.wait_for 超时）。
            # CancelledError 是 BaseException，不会被下面的 except Exception 捕获，
            # 不补记录的话 trace 里会完全缺失这类"超时兜底"调用的痕迹。
            _latency_ms = (_time.monotonic() - _t0) * 1000
            _record_llm_trace(
                model=model, system=system, user=user,
                response_content="",
                latency_ms=_latency_ms,
                temperature=temperature, max_tokens=max_tokens,
                error="CancelledError: 调用方在响应返回前取消（典型为 wait_for 超时）",
            )
            raise
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
        thinking_disabled: bool = False,
    ) -> LLMResponse:
        client = self._get_client()
        try:
            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }
            if thinking_disabled:
                # DeepSeek V4 默认思考模式，关闭思考走非推理，content 直接返回
                body["thinking"] = {"type": "disabled"}
            resp = await client.post("/chat/completions", json=body)
            if resp.status_code == 429:
                raise RateLimitError(f"LLM 限流 (429)")
            if resp.status_code >= 500:
                raise APITimeoutError(f"LLM 服务端错误 ({resp.status_code})")
            if resp.status_code != 200:
                body = resp.text[:300]
                raise LLMFatalError(f"LLM API 错误 ({resp.status_code}): {body}")

            data = resp.json()
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "") or ""
            if not content.strip():
                # 兜底: 思考模式下小 max_tokens 预算可能被 reasoning 全部吃光导致 content 为空。
                # 下游 json.loads("") 会静默失败走降级，这里读 reasoning_content 尽力挽回。
                reasoning = message.get("reasoning_content", "") or ""
                if reasoning.strip():
                    logger.warning(
                        "LLM content 为空，使用 reasoning_content 兜底 (%d chars)。"
                        "建议检查该调用的 max_tokens 是否过小或未关思考",
                        len(reasoning),
                    )
                    content = reasoning
            usage = data.get("usage", {})
            total_tokens = usage.get("total_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            finish_reason = data.get("choices", [{}])[0].get("finish_reason", "stop")

            return LLMResponse(
                content=content,
                model=model,
                tokens=total_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
            )
        except (RateLimitError, APITimeoutError):
            raise
        except LLMFatalError:
            raise
        except httpx.TimeoutException as e:
            raise APITimeoutError(f"LLM 请求超时: {e}")
        except json.JSONDecodeError as e:
            raise LLMFatalError(f"LLM 返回不可解析的响应: {e}")
        except httpx.HTTPStatusError as e:
            raise LLMFatalError(f"LLM API 错误 ({e.response.status_code})")
        except httpx.RequestError as e:
            raise APITimeoutError(f"LLM 网络错误: {e}")

    # ===== 流式生成 =====

    async def generate_stream(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        thinking_disabled: bool = True,
    ) -> "StreamResult":
        """流式生成 — 带重试 + 优雅降级。

        返回一个 StreamResult：可 ``async for text in result`` 读取 content，
        耗尽后通过 ``result.finish_reason`` 读取结束原因
        （``stop``/``length``/``content_filter`` 等），用于静默截断检测。
        """
        result = StreamResult()
        model = model or self._default_model
        last_error: Optional[Exception] = None

        for attempt in range(3):
            try:
                client = self._get_client()
                body = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True,
                }
                if thinking_disabled:
                    # DeepSeek V4 默认思考模式，关闭思考走非推理，content 直接返回
                    body["thinking"] = {"type": "disabled"}
                async with client.stream(
                    "POST",
                    "/chat/completions",
                    json=body,
                ) as response:
                    if response.status_code == 429:
                        raise RateLimitError("LLM 限流 (429)")
                    if response.status_code >= 500:
                        raise APITimeoutError(f"LLM 服务端错误 ({response.status_code})")
                    if response.status_code != 200:
                        body_text = (await response.aread()).decode()[:300]
                        raise LLMFatalError(f"LLM API 错误 ({response.status_code}): {body_text}")

                    # v16.1: 累积 content 与 reasoning，空 content 兜底 reasoning
                    total_content = ""
                    total_reasoning = ""
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            choices = chunk.get("choices", [])
                            if choices and choices[0].get("finish_reason"):
                                result.finish_reason = choices[0]["finish_reason"]
                            delta = (choices[0] if choices else {}).get("delta", {})
                            content = delta.get("content", "") or ""
                            if content:
                                total_content += content
                                result._emit(content)
                            reasoning = delta.get("reasoning_content", "") or ""
                            if reasoning:
                                total_reasoning += reasoning
                        except json.JSONDecodeError:
                            continue

                    if not total_content.strip() and total_reasoning.strip():
                        logger.warning(
                            "LLM 流式 content 为空，使用 reasoning_content 兜底 (%d chars)。"
                            "建议检查该调用的 max_tokens 是否过小或未关思考",
                            len(total_reasoning),
                        )
                        result._emit(total_reasoning)
                result._done_mark()
                return result

            except (RateLimitError, APITimeoutError) as e:
                last_error = e
                if attempt < 2:
                    wait_time = (2 ** attempt) * 1.5
                    logger.warning(f"LLM 流式中断 (尝试 {attempt+1}/3)，{wait_time}s 后重试: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    break
            except LLMFatalError as e:
                last_error = e
                break
            except httpx.TimeoutException as e:
                last_error = APITimeoutError(str(e))
                if attempt < 2:
                    await asyncio.sleep((2 ** attempt) * 1.5)
                else:
                    break
            except (json.JSONDecodeError, httpx.HTTPStatusError, httpx.RequestError) as e:
                last_error = e
                logger.warning(f"LLM 流式错误 (尝试 {attempt+1}/3): {e}")
                if attempt >= 2:
                    break
                await asyncio.sleep((2 ** attempt) * 1.5)

        logger.error(f"LLM 流式生成最终失败: {last_error}")
        result.finish_reason = "error"
        result._emit(f"\n\n(AI 响应中断: {str(last_error)[:100]}. 请稍后重试或缩短问题.)")
        result._done_mark()
        return result

    # ===== 带降级的生成 =====

    async def generate_with_fallback(
        self,
        system: str,
        user: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """带降级的生成: 主模型 -> 快速模型 -> 空回答。"""
        try:
            return await self.generate(system, user, self._default_model, max_tokens, temperature)
        except LLMFatalError as e:
            logger.error(f"LLM 主模型不可用: {e}")
        except (RateLimitError, APITimeoutError) as e:
            logger.warning(f"LLM 主模型暂时不可用: {e}")

        try:
            logger.info("降级到快速模型...")
            return await self.generate(
                system, user,
                self._fast_model,
                min(max_tokens, 1024),
                temperature,
            )
        except (LLMFatalError, RateLimitError, APITimeoutError) as e:
            logger.critical(f"LLM 所有模型不可用: {e}")

        return LLMResponse(
            content="(AI 服务繁忙，以下为检索到的相关文献片段，仅供参考)",
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
        """带工具调用的生成 — OpenAI 兼容 function calling。

        v17: thinking_disabled 默认 True（默认安全策略），
        需要推理的调用点（如 ReAct 决策轮）显式传 False。
        """
        model = model or self._default_model

        openai_tools = [t.to_openai_format() for t in tools]

        api_messages = [{"role": "system", "content": system}]
        api_messages.extend(messages)

        request_body: dict = {
            "model": model,
            "messages": api_messages,
            "tools": openai_tools,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if thinking_disabled:
            # DeepSeek V4 默认思考模式，关闭思考走非推理，content 直接返回
            request_body["thinking"] = {"type": "disabled"}

        try:
            client = self._get_client()
            resp = await client.post(
                "/chat/completions",
                json=request_body,
            )
            if resp.status_code == 429:
                raise RateLimitError("LLM 限流 (429)")
            if resp.status_code >= 500:
                raise APITimeoutError(f"LLM 服务端错误 ({resp.status_code})")
            if resp.status_code != 200:
                body = resp.text[:300]
                raise LLMFatalError(f"LLM tool_call 错误 ({resp.status_code}): {body}")

            data = resp.json()
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            finish_reason = choice.get("finish_reason", "stop")
            usage = data.get("usage", {})

            text = message.get("content", "") or ""
            # DeepSeek thinking 模式: reasoning_content 必须保留，
            # 下一轮请求回传给 API（否则 400: "must be passed back"）
            reasoning_content = message.get("reasoning_content", "") or ""

            tool_calls = []
            raw_tool_calls = message.get("tool_calls", [])
            for tc in raw_tool_calls:
                func = tc.get("function", {})
                args_str = func.get("arguments", "{}")
                try:
                    arguments = json.loads(args_str)
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append(ToolCallRequest(
                    id=tc.get("id", ""),
                    name=func.get("name", ""),
                    arguments=arguments,
                ))

            if tool_calls:
                return GenerateWithToolsResult(
                    text=text,
                    tool_calls=tool_calls,
                    model=model,
                    tokens=usage.get("total_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    finish_reason="tool_calls" if finish_reason == "tool_calls" else finish_reason,
                    reasoning_content=reasoning_content,
                )
            return GenerateWithToolsResult(
                text=text,
                model=model,
                tokens=usage.get("total_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                finish_reason=finish_reason,
                reasoning_content=reasoning_content,
            )

        except (RateLimitError, APITimeoutError):
            raise
        except LLMFatalError:
            raise
        except httpx.TimeoutException as e:
            raise APITimeoutError(f"LLM tool_call 超时: {e}")
        except json.JSONDecodeError as e:
            raise LLMFatalError(f"LLM tool_call 返回不可解析: {e}")
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            raise LLMFatalError(f"LLM tool_call 网络错误: {e}")

    async def generate_with_tools_stream(
        self,
        system: str,
        messages: list[dict],
        tools: list[ToolDefinition],
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        thinking_disabled: bool = True,
        on_reasoning: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> GenerateWithToolsResult:
        """流式 function calling — v17 思考透传专用。

        - 思考增量 (reasoning_content delta) 经 on_reasoning 回调实时上报，
          供上层转为 SSE thinking_delta 事件推给前端（用户不再觉得卡住）
        - tool_calls delta 按 index 聚合，结束时返回与非流式 generate_with_tools
          结构完全一致的 GenerateWithToolsResult（含 reasoning_content 供多轮回传）
        - 流式失败自动降级为非流式 generate_with_tools（思考全文一次性回调）
        """
        model = model or self._default_model

        openai_tools = [t.to_openai_format() for t in tools]
        api_messages = [{"role": "system", "content": system}]
        api_messages.extend(messages)

        body: dict = {
            "model": model,
            "messages": api_messages,
            "tools": openai_tools,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            # DeepSeek 支持流式末块返回 usage
            "stream_options": {"include_usage": True},
        }
        if thinking_disabled:
            body["thinking"] = {"type": "disabled"}

        try:
            client = self._get_client()
            async with client.stream("POST", "/chat/completions", json=body) as response:
                if response.status_code == 429:
                    raise RateLimitError("LLM 限流 (429)")
                if response.status_code >= 500:
                    raise APITimeoutError(f"LLM 服务端错误 ({response.status_code})")
                if response.status_code != 200:
                    body_text = (await response.aread()).decode()[:300]
                    raise LLMFatalError(f"LLM tool_call 流式错误 ({response.status_code}): {body_text}")

                total_content = ""
                total_reasoning = ""
                finish_reason = "stop"
                tokens = 0
                completion_tokens = 0
                # tool_calls delta 聚合: index -> {"id", "name", "arguments"}
                tc_agg: dict[int, dict] = {}

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if chunk.get("usage"):
                        tokens = chunk["usage"].get("total_tokens", tokens)
                        completion_tokens = chunk["usage"].get("completion_tokens", completion_tokens)

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    choice = choices[0]
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta", {}) or {}

                    # 思考增量: 累积 + 实时回调
                    reasoning = delta.get("reasoning_content", "") or ""
                    if reasoning:
                        total_reasoning += reasoning
                        if on_reasoning:
                            try:
                                await on_reasoning(reasoning)
                            except Exception:  # 回调异常不中断主流程
                                logger.exception("on_reasoning 回调失败")

                    content = delta.get("content", "") or ""
                    if content:
                        total_content += content

                    # tool_calls delta 按 index 聚合
                    for tcd in delta.get("tool_calls", []) or []:
                        idx = tcd.get("index", 0)
                        agg = tc_agg.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                        if tcd.get("id"):
                            agg["id"] = tcd["id"]
                        fn = tcd.get("function") or {}
                        if fn.get("name"):
                            agg["name"] = fn["name"]
                        if fn.get("arguments"):
                            agg["arguments"] += fn["arguments"]

            # 聚合为标准 tool_calls 列表
            tool_calls: list[ToolCallRequest] = []
            for idx in sorted(tc_agg.keys()):
                agg = tc_agg[idx]
                try:
                    arguments = json.loads(agg["arguments"]) if agg["arguments"] else {}
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append(ToolCallRequest(id=agg["id"], name=agg["name"], arguments=arguments))

            if tool_calls:
                finish_reason = "tool_calls"
            return GenerateWithToolsResult(
                text=total_content,
                tool_calls=tool_calls,
                model=model,
                tokens=tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
                reasoning_content=total_reasoning,
            )

        except (RateLimitError, APITimeoutError, LLMFatalError) as e:
            # 流式不可用时降级非流式（保持功能可用性优先）
            logger.warning(f"generate_with_tools_stream 失败，降级非流式: {e}")
            result = await self.generate_with_tools(
                system=system, messages=messages, tools=tools,
                model=model, max_tokens=max_tokens, temperature=temperature,
                thinking_disabled=thinking_disabled,
            )
            if on_reasoning and result.reasoning_content:
                try:
                    await on_reasoning(result.reasoning_content)
                except Exception:
                    logger.exception("on_reasoning 回调失败 (降级路径)")
            return result

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
