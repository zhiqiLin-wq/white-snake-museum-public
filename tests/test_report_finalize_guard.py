"""报告封卷闭环 + user_id 透传回归测试（v20）。

覆盖 2026-09-09「断桥对勘」事故的两类根因：
1. supervisor 降级分支创建 AgentLoop 时漏传 user_id/system_context
   → 报告写入 agent/reports/default/，用户文件库永远不可见。
2. ReAct 循环耗尽时报告已 create/append 但从未 finalize
   → 无下载卡片、文件库不可见，模型却在回答中谎称"可下载"。
   v20 机制：写作阶段工具闸门 + 封卷救援（预留预算）+ 后端自动封卷兜底。

运行方式:
  cd white-snake-museum-public
  python tests/test_report_finalize_guard.py   # 或 pytest
"""

import ast
import asyncio
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from agent.server.agent_loop import AgentLoop  # noqa: E402
from agent.server.langgraph.supervisor import SupervisorGraph  # noqa: E402
from agent.server.llm.tool_call import (  # noqa: E402
    GenerateWithToolsResult,
    ToolCallRequest,
    ToolCallResult,
)


# ---------------------------------------------------------------- 辅助桩

def _mk_tc(cid: str, name: str, arguments: dict) -> ToolCallRequest:
    return ToolCallRequest(id=cid, name=name, arguments=arguments)


def _mk_result(text: str = "", calls=None, tokens: int = 100) -> GenerateWithToolsResult:
    return GenerateWithToolsResult(
        text=text,
        tool_calls=calls or [],
        completion_tokens=tokens,
        finish_reason="tool_calls" if calls else "stop",
    )


def _mk_tool_result(success: bool, result: dict = None, error: str = "") -> ToolCallResult:
    return ToolCallResult(
        call_id="tc-x",
        tool_name="save_long_report",
        success=success,
        result=result if result is not None else {"error": error},
        full_result=result if result is not None else {"error": error},
        summary="",
        sse_events=[],
        error=error or None,
    )


class _FakeAnswerStream:
    """模拟 generate_stream 返回的异步迭代器（带 finish_reason 属性）。"""

    def __init__(self, chunks):
        self._chunks = chunks
        self.finish_reason = "stop"

    def __aiter__(self):
        async def gen():
            for t in self._chunks:
                yield t
        return gen()


def _make_agent_loop(llm_script: list, tool_side_effect):
    """构造最小 AgentLoop：快路径关闭（强制走 ReAct），LLM/工具全部脚本化。"""
    llm = MagicMock()
    llm.generate_with_tools_stream = AsyncMock(side_effect=llm_script)

    async def _fake_gen_stream(*a, **k):
        return _FakeAnswerStream(["最终", "答案文本"])

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
    agent._call_tool = AsyncMock(side_effect=tool_side_effect)
    return agent


def _report_tool_handler(calls_log: list):
    """save_long_report 工具的脚本化执行；非报告工具返回失败（正常情况下闸门会拦下）。"""
    async def _handler(name, params, on_progress=None):
        action = params.get("action", "") if isinstance(params, dict) else ""
        calls_log.append((name, action))
        if name != "save_long_report":
            return _mk_tool_result(False, error=f"测试桩：{name} 不应被执行")
        if action == "create":
            return _mk_tool_result(True, {"reportId": "rpt_test_1", "title": "断桥对勘报告"})
        if action == "append":
            return _mk_tool_result(True, {
                "reportId": "rpt_test_1", "sectionIndex": 1,
                "sectionTitle": params.get("section_title", ""),
                "sectionCount": 1, "totalChars": 800,
            })
        if action == "finalize":
            return _mk_tool_result(True, {
                "reportReady": True, "reportId": "rpt_test_1",
                "title": "断桥对勘报告", "sectionCount": 2,
                "sections": ["一、体例", "封卷说明"],
                "totalChars": 1600, "content": "# 断桥对勘报告\n",
            })
        return _mk_tool_result(False, error=f"未知 action {action}")
    return _handler


def _permissive_tool_handler(calls_log: list, report_id="rpt_test_2"):
    """非报告工具也返回成功（用于补充取证配额放行测试）。"""
    async def _handler(name, params, on_progress=None):
        action = params.get("action", "") if isinstance(params, dict) else ""
        calls_log.append((name, action))
        if name != "save_long_report":
            return _mk_tool_result(True, {"ok": True, "tool": name})
        if action == "create":
            return _mk_tool_result(True, {"reportId": report_id, "title": "配额测试报告"})
        if action == "append":
            return _mk_tool_result(True, {
                "reportId": report_id, "sectionIndex": 1,
                "sectionCount": 1, "totalChars": 500,
            })
        if action == "finalize":
            return _mk_tool_result(True, {
                "reportReady": True, "reportId": report_id,
                "title": "配额测试报告", "sectionCount": 1,
                "totalChars": 800, "content": "# 配额测试报告\n",
            })
        return _mk_tool_result(False, error=f"未知 action {action}")
    return _handler


async def _collect(agent, query="逐字对勘两本断桥一折的异文细节", thread_id="t-test"):
    events = []
    async for ev in agent.run_stream(query, thread_id=thread_id):
        events.append(ev)
    return events


# ---------------------------------------------------------------- 测试

class TestReportFinalizeGuard(unittest.TestCase):

    def _run(self, coro):
        return asyncio.run(coro)

    # ---- 封卷救援：模型始终不 finalize → 后端自动封卷兜底

    def test_auto_finalize_when_model_never_finalizes(self):
        """create→append→补充取证（配额 2 放行、第 3 次拦截）→模型想直接回答
        →救援3轮仍不封卷 → 后端自动 append 封卷说明 + finalize，report_ready 必须下发。"""
        calls_log: list = []
        script = [
            _mk_result(calls=[_mk_tc("c1", "save_long_report",
                                     {"action": "create", "title": "断桥对勘报告"})]),
            _mk_result(calls=[_mk_tc("c2", "save_long_report",
                                     {"action": "append", "report_id": "rpt_test_1",
                                      "section_title": "二、逐字共享证据", "content": "…"})]),
            # v20.1: 写作阶段取证 —— 前 2 次走补充取证配额（放行执行），第 3 次配额用尽拦截
            _mk_result(calls=[_mk_tc("c3", "diff_passages",
                                     {"left": "a", "right": "b"})]),
            _mk_result(calls=[_mk_tc("c4", "diff_passages",
                                     {"left": "c", "right": "d"})]),
            _mk_result(calls=[_mk_tc("c5", "diff_passages",
                                     {"left": "e", "right": "f"})]),
            # 模型试图不封卷直接回答 → 拦截转入救援
            _mk_result(text="完整报告已生成（约3000字），可点击下载。"),
            # 救援 3 轮：模型仍拒绝封卷，反复想直接回答
            _mk_result(text="我直接给你结论吧。"),
            _mk_result(text="信息足够了，直接回答。"),
            _mk_result(text="好的，以下是回答。"),
        ]
        agent = _make_agent_loop(script, _report_tool_handler(calls_log))

        events = self._run(_collect(agent))
        types = [e.get("type") for e in events]

        # 1) report_ready 必须下发（自动封卷）
        self.assertIn("report_ready", types,
                      f"未下发 report_ready，事件序列: {types}")
        # 2) 补充取证配额=2：diff_passages 恰好执行 2 次，第 3 次被拦截
        tool_names = [n for n, _ in calls_log]
        self.assertEqual(tool_names.count("diff_passages"), 2,
                         f"补充取证配额应放行 2 次、拦截第 3 次: {calls_log}")
        # 3) 自动封卷路径：create → append(模型) → append(封卷说明) → finalize
        actions = [a for n, a in calls_log if n == "save_long_report"]
        self.assertEqual(actions[0], "create")
        self.assertIn("finalize", actions)
        self.assertGreaterEqual(actions.count("append"), 2,
                                "应包含模型 append + 系统封卷说明 append")
        # 4) 最终回答照常生成
        chunks = "".join(str(e.get("data", "")) for e in events if e.get("type") == "chunk")
        self.assertIn("最终", chunks)
        # 5) 状态事件含封卷收尾提示与配额提示
        status_text = " ".join(str(e.get("data", "")) for e in events if e.get("type") == "status")
        self.assertIn("封卷", status_text)
        self.assertIn("配额", status_text)

    # ---- 封卷救援：模型在救援轮补做 finalize → 正常封卷，不触发自动兜底

    def test_rescue_succeeds_when_model_finalizes(self):
        calls_log: list = []
        script = [
            _mk_result(calls=[_mk_tc("c1", "save_long_report",
                                     {"action": "create", "title": "断桥对勘报告"})]),
            _mk_result(calls=[_mk_tc("c2", "save_long_report",
                                     {"action": "append", "report_id": "rpt_test_1",
                                      "section_title": "二、逐字共享证据", "content": "…"})]),
            _mk_result(text="完整报告已生成，可点击下载。"),  # 拦截 → 救援
            # 救援第 1 轮：模型补做 finalize
            _mk_result(calls=[_mk_tc("c4", "save_long_report",
                                     {"action": "finalize", "report_id": "rpt_test_1"})]),
        ]
        agent = _make_agent_loop(script, _report_tool_handler(calls_log))

        events = self._run(_collect(agent))
        types = [e.get("type") for e in events]

        self.assertIn("report_ready", types)
        actions = [a for n, a in calls_log if n == "save_long_report"]
        self.assertIn("finalize", actions)
        # 模型自己封卷 → 不应有系统"封卷说明"append（append 仅 1 次，来自模型正文）
        self.assertEqual(actions.count("append"), 1,
                         f"模型已封卷时不应触发自动兜底: {actions}")

    # ---- 正常流程不受影响：不写报告的问答不触发任何报告机制

    def test_no_report_flow_unchanged(self):
        calls_log: list = []

        async def _handler(name, params, on_progress=None):
            calls_log.append((name, params.get("action", "")))
            return _mk_tool_result(True, {"found": 1})

        script = [
            _mk_result(calls=[_mk_tc("c1", "search_literature", {"query": "雷峰塔"})]),
            _mk_result(text="基于检索结果，雷峰塔……"),  # 正常直接回答
        ]
        agent = _make_agent_loop(script, _handler)
        events = self._run(_collect(agent, query="雷峰塔象征什么"))
        types = [e.get("type") for e in events]
        self.assertNotIn("report_ready", types)
        self.assertIn("chunk", types)

    # ---- v20.1: 补充取证配额 —— 前 2 次放行、第 3 次硬拦、模型随后正常封卷

    def test_supplementary_evidence_quota(self):
        calls_log: list = []
        script = [
            _mk_result(calls=[_mk_tc("c1", "save_long_report",
                                     {"action": "create", "title": "配额测试"})]),
            _mk_result(calls=[_mk_tc("c2", "search_literature", {"query": "断桥"})]),
            _mk_result(calls=[_mk_tc("c3", "diff_passages", {"left": "a", "right": "b"})]),
            _mk_result(calls=[_mk_tc("c4", "get_chapter_full_text", {"chapter_number": 3})]),
            _mk_result(calls=[_mk_tc("c5", "save_long_report",
                                     {"action": "finalize", "report_id": "rpt_test_2"})]),
            # 封卷完成后模型给出摘要回答（情况 A，结束循环）
            _mk_result(text="报告已完成，以下是摘要……"),
        ]
        agent = _make_agent_loop(script, _permissive_tool_handler(calls_log))
        events = self._run(_collect(agent, query="深度对勘，需要写报告"))
        types = [e.get("type") for e in events]

        self.assertIn("report_ready", types, f"事件序列: {types}")
        tool_names = [n for n, _ in calls_log]
        self.assertIn("search_literature", tool_names)
        self.assertIn("diff_passages", tool_names)
        self.assertNotIn("get_chapter_full_text", tool_names,
                         f"第 3 次补充取证必须被配额拦截: {calls_log}")
        actions = [a for n, a in calls_log if n == "save_long_report"]
        self.assertIn("finalize", actions)

    # ---- v20.1: 迭代耗尽 → 优雅中断（DONE 带 interrupted + 断点落盘）

    def test_interrupted_done_carries_checkpoint(self):
        from agent.server.agent_loop import (
            _load_interrupt_checkpoint, _clear_interrupt_checkpoint,
        )
        _clear_interrupt_checkpoint("t-interrupt")
        calls_log: list = []

        async def _handler(name, params, on_progress=None):
            calls_log.append(name)
            return _mk_tool_result(True, {"found": 1})

        # 每轮都调工具、永不回答 → 18 轮迭代耗尽
        script = [
            _mk_result(calls=[_mk_tc(f"c{i}", "search_literature", {"query": f"q{i}"})])
            for i in range(20)
        ]
        agent = _make_agent_loop(script, _handler)
        try:
            events = self._run(_collect(agent, query="一个穷举式的复杂问题",
                                        thread_id="t-interrupt"))
            done = [e for e in events if e.get("type") == "done"]
            self.assertTrue(done)
            self.assertTrue(done[-1]["data"].get("interrupted"),
                            f"迭代耗尽必须标记 interrupted: {done[-1]['data']}")
            self.assertEqual(done[-1]["data"].get("interruptReason"), "iterations")
            ckpt = _load_interrupt_checkpoint("t-interrupt")
            self.assertIsNotNone(ckpt, "中断断点必须落盘")
            self.assertEqual(ckpt["reason"], "iterations")
            self.assertIn("search_literature", ckpt["tools"])
            # 阶段性答案必须生成（不是沉默截断）
            chunks = "".join(str(e.get("data", "")) for e in events if e.get("type") == "chunk")
            self.assertIn("最终", chunks)
        finally:
            _clear_interrupt_checkpoint("t-interrupt")

    # ---- v20.1: 续答 —— 用户说「继续」时注入上轮断点，正常完成后断点清除

    def test_continuation_injects_checkpoint_and_clears_on_finish(self):
        from agent.server.agent_loop import (
            _save_interrupt_checkpoint, _load_interrupt_checkpoint,
            _clear_interrupt_checkpoint,
        )
        _clear_interrupt_checkpoint("t-cont")
        _save_interrupt_checkpoint("t-cont", {
            "reason": "iterations", "iterations": 18,
            "tools": ["search_literature", "diff_passages"],
            "open_report_id": None, "report_was_open": False,
            "query": "逐字对勘两本断桥一折的异文细节",
            "ts": "2026-09-09 01:00:00",
        })
        try:
            captured: dict = {}
            agent = _make_agent_loop([], None)

            async def _gwts(system, messages, **kwargs):
                captured["messages"] = messages
                return _mk_result(text="好的，接着未竟事项继续。")

            agent.llm.generate_with_tools_stream = _gwts

            async def _handler(name, params, on_progress=None):
                return _mk_tool_result(True, {"ok": 1})
            agent._call_tool = AsyncMock(side_effect=_handler)

            events = self._run(_collect(agent, query="继续分析，接着做",
                                        thread_id="t-cont"))
            types = [e.get("type") for e in events]
            self.assertIn("chunk", types)
            joined = " ".join(str(m.get("content", "")) for m in captured.get("messages", []))
            self.assertIn("上轮中断断点", joined, "续答必须注入上轮断点上下文")
            self.assertIn("逐字对勘", joined)
            # 本轮正常完成 → 断点清除
            self.assertIsNone(_load_interrupt_checkpoint("t-cont"))
        finally:
            _clear_interrupt_checkpoint("t-cont")


class TestReportReopen(unittest.TestCase):
    """v20.1: 已封卷报告 reopen 续写。"""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_reopen_finalized_report_allows_append(self):
        from agent.server.mcp.tools.save_long_report import handler, REPORTS_DIR, _safe_id
        uid = "u-test-reopen-flow"

        async def _flow():
            r = await handler(action="create", user_id=uid, title="续写测试", topic="t")
            rid = r["reportId"]
            await handler(action="append", user_id=uid, report_id=rid,
                          section_title="一、初编", content="初编内容")
            fin = await handler(action="finalize", user_id=uid, report_id=rid)
            self.assertTrue(fin["reportReady"])
            # 封卷后 append 必须被拒绝
            blk = await handler(action="append", user_id=uid, report_id=rid,
                                section_title="x", content="y")
            self.assertIn("error", blk)
            # reopen 后可续写，章节序号接续
            ro = await handler(action="reopen", user_id=uid, report_id=rid)
            self.assertEqual(ro["reportId"], rid)
            self.assertEqual(ro["reopenCount"], 1)
            ap = await handler(action="append", user_id=uid, report_id=rid,
                               section_title="二、补编", content="补编内容")
            self.assertEqual(ap["sectionIndex"], 2)
            fin2 = await handler(action="finalize", user_id=uid, report_id=rid)
            self.assertTrue(fin2["reportReady"])
            self.assertEqual(fin2["sectionCount"], 2)
            # 再次 reopen → reopenCount 递增（幂等可重复续写）
            ro2 = await handler(action="reopen", user_id=uid, report_id=rid)
            self.assertEqual(ro2["reopenCount"], 2)
            return rid

        rid = self._run(_flow())
        shutil.rmtree(REPORTS_DIR / _safe_id(uid), ignore_errors=True)
        self.assertTrue(rid.startswith("rpt_"))


class TestAgentLoopFactoryGuard(unittest.TestCase):
    """v20.1: AST 静态守卫 —— supervisor 中所有 AgentLoop 构造必须走 _make_agent_loop。"""

    def test_no_direct_construction_outside_factory(self):
        sup_path = _project_root / "agent" / "server" / "langgraph" / "supervisor.py"
        tree = ast.parse(sup_path.read_text(encoding="utf-8"))
        factory_found = False
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "_make_agent_loop":
                    factory_found = True
                for sub in ast.walk(node):
                    if (isinstance(sub, ast.Call)
                            and isinstance(sub.func, ast.Name)
                            and sub.func.id == "AgentLoop"
                            and node.name != "_make_agent_loop"):
                        violations.append((node.name, sub.lineno))
        self.assertTrue(factory_found, "supervisor 必须定义 _make_agent_loop 工厂方法")
        self.assertEqual(violations, [],
                         f"发现绕过 _make_agent_loop 工厂的直接构造（函数, 行号）: {violations}")


class TestSupervisorFallbackUserId(unittest.TestCase):
    """supervisor 章节正则降级分支必须透传 user_id/system_context。"""

    def _run(self, coro):
        return asyncio.run(coro)

    def _make_supervisor(self):
        sup = SupervisorGraph(
            llm=MagicMock(),
            retriever=MagicMock(),
            prompt_registry=None,
            skill_registry=None,
        )
        sup._request_user_id = "u-123"
        sup._request_system_context = "系统上下文-X"
        return sup

    def test_stream_fallback_passes_user_id(self):
        """_stream_dynasty_compare 正则降级分支：AgentLoop 必须带 user_id。"""
        captured = {}

        class FakeAgentLoop:
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)

            async def run_stream(self, query, thread_id="", history=None):
                yield {"type": "done", "data": {}}

        sup = self._make_supervisor()
        state = {"messages": [{"role": "user",
                               "content": "请逐字对勘两本断桥一折的异文细节"}],
                 "thread_id": "t1"}

        async def _go():
            events = []
            with patch("agent.server.agent_loop.AgentLoop", FakeAgentLoop):
                async for ev in sup._stream_dynasty_compare(
                        "请逐字对勘两本断桥一折的异文细节", "t1", state):
                    events.append(ev)
            return events

        events = self._run(_go())
        self.assertTrue(events)
        self.assertEqual(captured.get("user_id"), "u-123",
                         f"流式降级分支漏传 user_id: {captured}")
        self.assertEqual(captured.get("system_context"), "系统上下文-X")

    def test_exec_fallback_passes_user_id(self):
        """_exec_dynasty_compare 非流式降级分支：AgentLoop 必须带 user_id。"""
        captured = {}

        class FakeAgentLoop:
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)

            async def run(self, query, thread_id="", history=None):
                return {"answer": "ok", "sources": []}

        sup = self._make_supervisor()
        state = {"messages": [{"role": "user",
                               "content": "请逐字对勘两本断桥一折的异文细节"}],
                 "thread_id": "t1"}

        async def _go():
            with patch("agent.server.agent_loop.AgentLoop", FakeAgentLoop):
                return await sup._exec_dynasty_compare(state)

        result = self._run(_go())
        self.assertEqual(result["subgraph_result"]["workflow"], "agent_loop_fallback")
        self.assertEqual(captured.get("user_id"), "u-123",
                         f"非流式降级分支漏传 user_id: {captured}")
        self.assertEqual(captured.get("system_context"), "系统上下文-X")


if __name__ == "__main__":
    unittest.main(verbosity=2)
