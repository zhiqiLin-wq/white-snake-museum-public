"""ReAct 决策轮超时降级（L1+L2+L3）回归测试（测试先行）。

覆盖 2026-09-09 改动：DeepSeek thinking 模式卡死时，单轮整体 90s 超时后分层降级：
- L1: 关闭 thinking 用非流式 generate_with_tools 重试一次（60s 上限）
- L2: 重试也超时 → 用 fast_result.docs 兜底回答
- L3: 全失败 → ERROR 事件带异常类型名（修复空括号 bug）

测试目标:
1. 流式超时触发 → 进入 L1 重试
2. L1 成功 → 正常进入 ReAct 后续（不报 ERROR）
3. L1 也超时 → 进入 L2，用 fast_result 兜底，产出 chunk
4. fast_result 也空 → L3 ERROR，消息含异常类型名，非空括号
5. 兜底 wrapper run_stream 捕获未知异常时带 type(e).__name__

运行: python tests/test_react_round_timeout.py  (或 pytest)
"""
import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from agent.server.agent_loop import AgentLoop  # noqa: E402
from agent.server.llm.tool_call import (  # noqa: E402
    GenerateWithToolsResult, ToolCallRequest, ToolCallResult,
)


def _mk_tc(cid, name, arguments):
    return ToolCallRequest(id=cid, name=name, arguments=arguments)


def _mk_result(text="", calls=None, tokens=100):
    return GenerateWithToolsResult(
        text=text, tool_calls=calls or [],
        completion_tokens=tokens,
        finish_reason="tool_calls" if calls else "stop",
    )


def _mk_tool_result(success=True, result=None, error=""):
    return ToolCallResult(
        call_id="tc-x", tool_name="search_literature",
        success=success,
        result=result if result is not None else {"error": error},
        full_result=result if result is not None else {"error": error},
        summary="", sse_events=[], error=error or None,
    )


class _FakeAnswerStream:
    def __init__(self, chunks):
        self._chunks = chunks
        self.finish_reason = "stop"

    def __aiter__(self):
        async def gen():
            for t in self._chunks:
                yield t
        return gen()


def _make_agent(llm_script, tool_handler, fast_result=None):
    """构造 AgentLoop，可控制 fast_result 是否返回 docs。"""
    llm = MagicMock()
    llm.generate_with_tools_stream = AsyncMock(side_effect=llm_script)
    llm.generate_with_tools = AsyncMock(side_effect=llm_script)  # L1 重试也用同脚本

    async def _fake_gen_stream(*a, **k):
        return _FakeAnswerStream(["降级", "回答"])
    llm.generate_stream = _fake_gen_stream

    retriever = MagicMock()
    mcp = MagicMock()
    mcp.list_tools = MagicMock(return_value=[])

    agent = AgentLoop(
        llm=llm, retriever=retriever, prompt_registry=None,
        mcp_registry=mcp, user_id="u-test",
    )
    agent._try_fast_answer = AsyncMock(return_value=fast_result)
    agent._post_verify_citations = AsyncMock(side_effect=lambda ans, srcs: ans)
    agent._call_tool = AsyncMock(side_effect=tool_handler)
    return agent


async def _collect(agent, query="对比清代和现当代版本中许仙的描写差异",
                   thread_id="t-timeout"):
    events = []
    async for ev in agent.run_stream(query, thread_id=thread_id):
        events.append(ev)
    return events


class TestReactRoundTimeout(unittest.TestCase):

    def _run(self, coro):
        return asyncio.run(coro)

    # ---- L1: 流式超时 → 重试成功 ----

    def test_stream_timeout_triggers_l1_retry_success(self):
        """流式首调卡 90s 超时 → L1 重试返回正常 tool_calls，不报 ERROR。"""
        async def _hang_stream(*a, **k):
            # 模拟 DeepSeek thinking 卡死：永远不返回
            await asyncio.sleep(200)
            return _mk_result()

        async def _l1_success(*a, **k):
            return _mk_result(text="L1 重试成功，直接回答")

        llm = MagicMock()
        llm.generate_with_tools_stream = AsyncMock(side_effect=_hang_stream)
        llm.generate_with_tools = AsyncMock(side_effect=_l1_success)

        async def _fake_gen_stream(*a, **k):
            return _FakeAnswerStream([""])
        llm.generate_stream = _fake_gen_stream

        retriever = MagicMock()
        mcp = MagicMock()
        mcp.list_tools = MagicMock(return_value=[])

        agent = AgentLoop(
            llm=llm, retriever=retriever, prompt_registry=None,
            mcp_registry=mcp, user_id="u-test",
        )
        agent._try_fast_answer = AsyncMock(return_value=None)
        agent._post_verify_citations = AsyncMock(side_effect=lambda ans, srcs: ans)
        agent._call_tool = AsyncMock(return_value=_mk_tool_result())

        # 把超时阈值改小，避免测试跑 90s
        import agent.server.agent_loop as al_mod
        orig = al_mod.REACT_ROUND_TIMEOUT_SECONDS
        al_mod.REACT_ROUND_TIMEOUT_SECONDS = 0.3
        al_mod.REACT_RETRY_TIMEOUT_SECONDS = 0.3
        try:
            events = self._run(_collect(agent))
        finally:
            al_mod.REACT_ROUND_TIMEOUT_SECONDS = orig
            al_mod.REACT_RETRY_TIMEOUT_SECONDS = getattr(orig, 'orig_retry', 60)

        types = [e.get("type") for e in events]
        # L1 成功 → 不应报 ERROR
        self.assertNotIn("error", types,
                        f"L1 重试成功不应报 ERROR，事件: {types}")
        # 应有 status 提示降级重试
        status_text = " ".join(str(e.get("data", "")) for e in events
                                if e.get("type") == "status")
        self.assertIn("超时", status_text)
        # 应有最终答案
        chunks = "".join(str(e.get("data", "")) for e in events
                         if e.get("type") == "chunk")
        self.assertIn("L1", chunks)

    # ---- L2: L1 也超时 → fast_result 兜底 ----

    def test_l1_timeout_uses_fast_result_fallback(self):
        """L1 重试也超时 → 用 _try_fast_answer 返回的 docs 兜底回答。"""
        async def _hang(*a, **k):
            await asyncio.sleep(200)
            return _mk_result()

        llm = MagicMock()
        llm.generate_with_tools_stream = AsyncMock(side_effect=_hang)
        llm.generate_with_tools = AsyncMock(side_effect=_hang)

        async def _fake_gen_stream(*a, **k):
            return _FakeAnswerStream(["L2", "兜底", "回答"])
        llm.generate_stream = _fake_gen_stream

        retriever = MagicMock()
        mcp = MagicMock()
        mcp.list_tools = MagicMock(return_value=[])

        agent = AgentLoop(
            llm=llm, retriever=retriever, prompt_registry=None,
            mcp_registry=mcp, user_id="u-test",
        )
        # fast_result 返回有 docs 的结构
        fast = {
            "docs": [{"content": "许仙描写片段", "metadata": {"chapter_title": "清"}}],
            "sufficient": False, "fast_intent": "comparison",
        }
        agent._try_fast_answer = AsyncMock(return_value=fast)
        agent._post_verify_citations = AsyncMock(side_effect=lambda ans, srcs: ans)

        import agent.server.agent_loop as al_mod
        orig = al_mod.REACT_ROUND_TIMEOUT_SECONDS
        al_mod.REACT_ROUND_TIMEOUT_SECONDS = 0.2
        try:
            events = self._run(_collect(agent))
        finally:
            al_mod.REACT_ROUND_TIMEOUT_SECONDS = orig

        types = [e.get("type") for e in events]
        # 不应报 ERROR（L2 兜底成功）
        self.assertNotIn("error", types,
                        f"L2 兜底成功不应报 ERROR: {types}")
        # 应有 chunk
        chunks = "".join(str(e.get("data", "")) for e in events
                         if e.get("type") == "chunk")
        self.assertTrue(chunks, "L2 兜底必须产出 chunk")
        # 应有 status 提示
        status_text = " ".join(str(e.get("data", "")) for e in events
                                if e.get("type") == "status")
        self.assertIn("兜底", status_text)

    # ---- L3: 全失败 → ERROR 带异常类型名 ----

    def test_all_timeout_emits_error_with_type_name(self):
        """L1+L2 全失败且无 fast_result → ERROR 事件消息含异常类型名（不空括号）。"""
        async def _hang(*a, **k):
            await asyncio.sleep(200)
            return _mk_result()

        llm = MagicMock()
        llm.generate_with_tools_stream = AsyncMock(side_effect=_hang)
        llm.generate_with_tools = AsyncMock(side_effect=_hang)

        async def _fake_gen_stream(*a, **k):
            return _FakeAnswerStream([""])
        llm.generate_stream = _fake_gen_stream

        retriever = MagicMock()
        mcp = MagicMock()
        mcp.list_tools = MagicMock(return_value=[])

        agent = AgentLoop(
            llm=llm, retriever=retriever, prompt_registry=None,
            mcp_registry=mcp, user_id="u-test",
        )
        agent._try_fast_answer = AsyncMock(return_value=None)
        agent._post_verify_citations = AsyncMock(side_effect=lambda ans, srcs: ans)

        import agent.server.agent_loop as al_mod
        orig = al_mod.REACT_ROUND_TIMEOUT_SECONDS
        al_mod.REACT_ROUND_TIMEOUT_SECONDS = 0.2
        try:
            events = self._run(_collect(agent))
        finally:
            al_mod.REACT_ROUND_TIMEOUT_SECONDS = orig

        errors = [e for e in events if e.get("type") == "error"]
        self.assertTrue(errors, "L3 必须下发 ERROR 事件")
        msg = errors[0].get("data", {}).get("message", "")
        # 消息必须含异常类型名（如 TimeoutError/asyncio.TimeoutError），不能是空括号
        self.assertTrue(
            "TimeoutError" in msg or "asyncio" in msg or "（" in msg,
            f"ERROR 消息必须含异常类型名，实际: {msg!r}"
        )
        # 修复空括号 bug 的关键断言：括号内不能为空
        self.assertFalse(
            msg.endswith("（）。") or msg.endswith("()。"),
            f"ERROR 消息不能是空括号，实际: {msg!r}"
        )

    # ---- 兜底 wrapper 也要带类型名（修复空括号 bug） ----

    def test_run_stream_wrapper_error_carries_type_name(self):
        """run_stream 兜底捕获未知异常时，消息必须含异常类型名，不是空括号。"""
        # 让 _run_stream_inner 抛一个无消息异常
        class _SilentError(Exception):
            pass

        async def _explode(*a, **k):
            raise _SilentError()  # 无消息

        llm = MagicMock()
        llm.generate_with_tools_stream = AsyncMock(side_effect=_explode)

        retriever = MagicMock()
        mcp = MagicMock()
        mcp.list_tools = MagicMock(return_value=[])

        agent = AgentLoop(
            llm=llm, retriever=retriever, prompt_registry=None,
            mcp_registry=mcp, user_id="u-test",
        )
        agent._try_fast_answer = AsyncMock(return_value=None)
        agent._post_verify_citations = AsyncMock(side_effect=lambda ans, srcs: ans)

        events = self._run(_collect(agent))
        errors = [e for e in events if e.get("type") == "error"]
        self.assertTrue(errors, "必须下发 ERROR")
        msg = errors[0].get("data", {}).get("message", "")
        # 类型名必须出现
        self.assertIn("_SilentError", msg,
                      f"兜底 ERROR 必须含异常类型名，实际: {msg!r}")


if __name__ == "__main__":
    unittest.main()
