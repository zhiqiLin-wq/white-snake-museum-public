"""回归测试：流式 LLM 调用超时保护 + 空异常消息补类型名。

测试场景：
1. generate_with_tools_stream 的 aiter_lines 挂死时须在 STREAM_LINE_TIMEOUT 秒内超时
2. generate_stream 的 aiter_lines 挂死时须标记 error
3. agent_loop.run_stream 兜底捕获空消息异常时，error 事件须含异常类型名
4. supervisor _stream_literature_qa 同理
"""
import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))

import pytest


class _HungResponse:
    """模拟 200 OK 后流式 aiter_lines 永不 yield 的挂死场景。"""
    status_code = 200
    async def __aenter__(self):
        return self
    async def __aexit__(self, *a):
        return False
    async def aiter_lines(self):
        # 用 Event 而非 sleep，避免被 asyncio.sleep patch 干扰
        await asyncio.Event().wait()  # never completes
        yield  # pragma: no cover


# ---------------------------------------------------------------------------
# 1. generate_with_tools_stream 超时测试
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_with_tools_stream_timeout_on_hung_stream():
    """aiter_lines 挂死时须在 STREAM_LINE_TIMEOUT 秒内抛 APITimeoutError。"""
    from server.llm.openai_compatible_provider import GenericOpenAIProvider
    from server.llm.provider import APITimeoutError

    provider = GenericOpenAIProvider(
        api_key="fake", base_url="http://fake", default_model="fake"
    )
    fake_client = MagicMock()
    fake_client.stream = MagicMock(return_value=_HungResponse())
    provider._client = fake_client

    # 让降级非流式也立即失败，避免等待
    with patch.object(provider, "generate_with_tools",
                      new=AsyncMock(side_effect=APITimeoutError("fallback failed"))):
        with patch("server.llm.openai_compatible_provider.STREAM_LINE_TIMEOUT", 0.1):
            with pytest.raises(APITimeoutError):
                await provider.generate_with_tools_stream(
                    system="s", messages=[{"role": "user", "content": "hi"}],
                    tools=[], model="fake",
                )


# ---------------------------------------------------------------------------
# 2. generate_stream 超时测试
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_stream_timeout_marks_error():
    """generate_stream 的 aiter_lines 挂死时，StreamResult 须标记 error。"""
    from server.llm.openai_compatible_provider import GenericOpenAIProvider

    provider = GenericOpenAIProvider(
        api_key="fake", base_url="http://fake", default_model="fake"
    )
    fake_client = MagicMock()
    fake_client.stream = MagicMock(return_value=_HungResponse())
    provider._client = fake_client

    with patch("server.llm.openai_compatible_provider.STREAM_LINE_TIMEOUT", 0.05):
        with patch("asyncio.sleep", new=AsyncMock(return_value=None)):
            result = await provider.generate_stream(
                system="s", user="hi", model="fake",
            )
    assert result.finish_reason == "error"
    combined = ""
    async for text in result:
        combined += text
    assert "超时" in combined or "timeout" in combined.lower()


# ---------------------------------------------------------------------------
# 3. agent_loop.run_stream 空异常消息补类型名
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_loop_run_stream_empty_exception_includes_type_name():
    """run_stream 兜底捕获 str(e)=='' 的异常时，ERROR 事件消息须含异常类型名。"""
    from server.agent_loop import AgentLoop

    class _SilentError(Exception):
        """消息为空的异常，模拟底层流式 read timeout（str(e)=='')."""

    loop = AgentLoop.__new__(AgentLoop)

    async def _boom(*a, **kw):
        raise _SilentError()
        yield  # pragma: no cover

    loop._run_stream_inner = _boom

    events = []
    async for ev in loop.run_stream("test query", thread_id="t1"):
        events.append(ev)

    error_events = [e for e in events if e.get("type") == "error"]
    assert len(error_events) == 1
    msg = error_events[0]["data"]["message"]
    assert "_SilentError" in msg, f"消息须含异常类型名，实际: {msg}"
    assert "()" not in msg, f"消息不能含空括号，实际: {msg}"


# ---------------------------------------------------------------------------
# 4. supervisor _stream_literature_qa 空异常消息补类型名
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_supervisor_literature_qa_empty_exception_includes_type_name():
    """_stream_literature_qa 兜底捕获空消息异常时，ERROR 事件须含异常类型名。"""
    from server.langgraph.supervisor import SupervisorGraph

    class _SilentError(Exception):
        pass

    sup = SupervisorGraph.__new__(SupervisorGraph)
    sup.llm = MagicMock()
    sup.retriever = MagicMock()
    sup.prompts = {}
    sup.mcp = None
    sup.node_client = None
    sup._request_user_id = ""
    sup._request_system_context = ""

    class _FakeAgentLoop:
        async def run_stream(self, *a, **kw):
            raise _SilentError()
            yield  # pragma: no cover

    with patch("server.agent_loop.AgentLoop", return_value=_FakeAgentLoop()):
        events = []
        async for ev in sup._stream_literature_qa("test", "t1", {}):
            events.append(ev)

    error_events = [e for e in events if e.get("type") == "error"]
    assert len(error_events) == 1
    msg = error_events[0]["data"]["message"]
    assert "_SilentError" in msg, f"消息须含异常类型名，实际: {msg}"
    assert "()" not in msg, f"消息不能含空括号，实际: {msg}"
