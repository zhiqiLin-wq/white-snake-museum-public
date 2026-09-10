"""v20: LLM 调用 trace 埋点覆盖测试（测试先行）。

背景：llm_calls.log 此前只记录非流式 generate()，四个 LLM 入口中
generate_stream / generate_with_tools / generate_with_tools_stream
全部零埋点 —— ReAct 决策轮（工具调用）在日志中完全缺失，
排查"模型为什么这样决策"时无据可查。

本测试锁定：四个入口在成功/失败时都会上报 Tracer，且
llm_call_context 的 node/purpose 被正确带到记录里。
"""
import asyncio
import json
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.server.llm.openai_compatible_provider import (
    GenericOpenAIProvider,
    StreamResult,
)
from agent.server.llm.tool_call import ToolDefinition
from agent.server.observability.tracer import Tracer, llm_call_context


# ===== 测试桩 =====

class _FakeResp:
    def __init__(self, data, status_code=200):
        self.status_code = status_code
        self._data = data
        self.text = json.dumps(data, ensure_ascii=False)

    def json(self):
        return self._data


class _FakePostClient:
    def __init__(self, data):
        self._data = data

    async def post(self, *args, **kwargs):
        return _FakeResp(self._data)


class _FakeStreamResponse:
    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamClient:
    def __init__(self, lines):
        self._lines = lines

    def stream(self, *args, **kwargs):
        return _FakeStreamResponse(self._lines)


def _tool_call_sse_lines():
    """构造 DeepSeek 流式 function calling 的 SSE 帧序列。"""
    return [
        'data: ' + json.dumps({"choices": [{"delta": {"reasoning_content": "需要检索断桥异文"},
                                             "finish_reason": None}]}, ensure_ascii=False),
        'data: ' + json.dumps({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1",
             "function": {"name": "diff_passages", "arguments": ""}}]}}]}, ensure_ascii=False),
        'data: ' + json.dumps({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '{"passage_a": "程本", "passage_b": "稗编本"}'}}]}}]}, ensure_ascii=False),
        'data: ' + json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                               "usage": {"total_tokens": 120, "completion_tokens": 80}}, ensure_ascii=False),
        'data: [DONE]',
    ]


def _one_tool():
    return ToolDefinition(
        name="diff_passages",
        description="逐字对勘两段异文",
        parameters={"type": "object", "properties": {
            "passage_a": {"type": "string"},
            "passage_b": {"type": "string"}}},
    )


class TestStreamResultTrace(unittest.TestCase):
    """StreamResult._done_mark 在结束时上报 trace。"""

    def setUp(self):
        self.trace = Tracer.start_trace("c-stream", "q")

    def tearDown(self):
        Tracer.clear()

    def test_done_mark_records_full_text(self):
        r = StreamResult()
        r.trace_meta = {
            "model": "m", "system": "sys", "user": "usr",
            "temperature": 0.7, "max_tokens": 100, "t0": time.monotonic(),
        }
        with llm_call_context("agent_loop", "final_answer",
                              upstream="react_loop"):
            r._emit("你好")
            r._emit("，白蛇")
            r._done_mark()
        self.assertEqual(len(self.trace.llm_calls), 1)
        rec = self.trace.llm_calls[0]
        self.assertEqual(rec.node, "agent_loop")
        self.assertEqual(rec.purpose, "final_answer")
        self.assertEqual(rec.response, "你好，白蛇")
        self.assertEqual(rec.system_prompt, "sys")

    def test_done_mark_with_error_marked_degraded(self):
        r = StreamResult()
        r.trace_meta = {
            "model": "m", "system": "s", "user": "u",
            "temperature": 0.7, "max_tokens": 100, "t0": time.monotonic(),
        }
        r._emit("片段")
        r._done_mark(error="RuntimeError: boom")
        rec = self.trace.llm_calls[0]
        self.assertTrue(rec.degraded)
        self.assertIn("boom", rec.error)

    def test_no_meta_no_record(self):
        """未设置 trace_meta（非生产用法）时静默不上报。"""
        r = StreamResult()
        r._emit("x")
        r._done_mark()
        self.assertEqual(len(self.trace.llm_calls), 0)


class TestToolsStreamTrace(unittest.TestCase):
    """generate_with_tools_stream（ReAct 决策轮）埋点。"""

    def setUp(self):
        self.trace = Tracer.start_trace("c-tools", "q")
        self.provider = GenericOpenAIProvider(
            api_key="k", base_url="http://x", default_model="m-deepseek")
        self.provider._get_client = lambda: _FakeStreamClient(_tool_call_sse_lines())

    def tearDown(self):
        Tracer.clear()

    def test_react_round_recorded_with_tool_calls_and_reasoning(self):
        with llm_call_context("agent_loop", "react_tool_round",
                              upstream="react_loop"):
            result = asyncio.get_event_loop().run_until_complete(
                self.provider.generate_with_tools_stream(
                    system="系统提示",
                    messages=[{"role": "user", "content": "对勘断桥异文"}],
                    tools=[_one_tool()],
                    max_tokens=4096,
                    thinking_disabled=False,
                ))
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "diff_passages")
        self.assertEqual(len(self.trace.llm_calls), 1,
                         "ReAct 决策轮必须写入 llm trace")
        rec = self.trace.llm_calls[0]
        self.assertEqual(rec.node, "agent_loop")
        self.assertEqual(rec.purpose, "react_tool_round")
        self.assertIn("diff_passages", rec.response)
        self.assertIn("需要检索断桥异文", rec.response,
                      "reasoning_content 应一并入日志（排查决策依据）")
        self.assertIn("对勘断桥异文", rec.user_prompt)
        self.assertEqual(rec.tokens_total, 120)


class TestToolsNonStreamTrace(unittest.TestCase):
    """generate_with_tools（非流式工具调用/降级路径）埋点。"""

    def setUp(self):
        self.trace = Tracer.start_trace("c-tools-ns", "q")
        self.provider = GenericOpenAIProvider(
            api_key="k", base_url="http://x", default_model="m-deepseek")
        self.provider._get_client = lambda: _FakePostClient({
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "content": "",
                    "tool_calls": [{"id": "c2", "function": {
                        "name": "save_long_report",
                        "arguments": '{"action": "create", "title": "异文对勘报告"}'
                    }}],
                },
            }],
            "usage": {"total_tokens": 50, "completion_tokens": 30},
        })

    def tearDown(self):
        Tracer.clear()

    def test_non_stream_tool_round_recorded(self):
        with llm_call_context("agent_loop", "react_tool_round",
                              upstream="report_rescue"):
            result = asyncio.get_event_loop().run_until_complete(
                self.provider.generate_with_tools(
                    system="s",
                    messages=[{"role": "user", "content": "封卷"}],
                    tools=[_one_tool()],
                    max_tokens=2048,
                ))
        self.assertEqual(result.tool_calls[0].name, "save_long_report")
        self.assertEqual(len(self.trace.llm_calls), 1)
        rec = self.trace.llm_calls[0]
        self.assertEqual(rec.purpose, "react_tool_round")
        self.assertIn("save_long_report", rec.response)


if __name__ == "__main__":
    unittest.main(verbosity=2)
