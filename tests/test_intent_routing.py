"""意图路由回归测试（v12 — LLM 自选路径 + harness 护栏）。

验证 supervisor._classify_intent 的路由决策纯逻辑：
- 路径能力卡 prompt 下 LLM 的选择被尊重
- evolution 重路径护栏：无演化语义指示 → 降级 literature_qa（拦截分支）
- evolution 正例（含演化语义）→ 放行
- 非法意图 → 白名单回退
- LLM 失败 → 降级
- route_reason 透传（LLM 生成 / 护栏说明 / 降级说明）

回归查询取自 agent.log 近期真实用户查询，覆盖历史误路由形态。

运行方式:
  cd white-snake-museum-public
  python tests/test_intent_routing.py   # 或 pytest tests/test_intent_routing.py
"""

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from agent.server.langgraph.supervisor import SupervisorGraph  # noqa: E402
from agent.server.llm.provider import LLMFatalError  # noqa: E402


def _make_supervisor(llm_response: str | None = None, llm_error: Exception | None = None):
    """构造最小 SupervisorGraph：prompt_registry=None 走硬编码能力卡 prompt，
    LLM 为 mock（返回预设 JSON 或抛错），retriever/skill_registry 为哑对象。"""
    llm = MagicMock()
    if llm_error is not None:
        llm.generate = AsyncMock(side_effect=llm_error)
    else:
        resp = MagicMock()
        resp.content = llm_response or "{}"
        llm.generate = AsyncMock(return_value=resp)
    sup = SupervisorGraph(
        llm=llm,
        retriever=MagicMock(),
        prompt_registry=None,  # 走 fallback 能力卡 prompt 分支
        skill_registry=None,
    )
    return sup


def _state(query: str, history: list[dict] | None = None):
    return {
        "messages": [{"role": "user", "content": query}] + (history or []),
        "intent": "",
        "intent_confidence": 0.0,
        "keywords": [],
    }


class TestIntentRouting(unittest.TestCase):

    def _run(self, coro):
        return asyncio.run(coro)

    # ---- 护栏：拦截分支 ----

    def test_evolution_guard_intercepts_non_evolution_query(self):
        """真实误路由案例：韵文枚举+变化询问被 LLM 误判 evolution → 护栏拦截。"""
        sup = _make_supervisor(llm_response=(
            '{"intent": "evolution", "keywords": ["韵文", "诗词"], '
            '"route_reason": "涉及各版本变化", '
            '"evolution_scenario": "trace", "evolution_target": "韵文", '
            '"evolution_target_type": "theme"}'
        ))
        result = self._run(sup._classify_intent(
            _state("白蛇传各版本中的韵文诗词有哪些？前后发生了什么变化？")))
        self.assertEqual(result["intent"], "literature_qa")
        self.assertIn("护栏拦截", result["route_reason"])

    def test_evolution_guard_intercepts_plain_symbolism(self):
        """真实误路由案例：单点释义'象征什么'无演化语义 → 拦截。"""
        sup = _make_supervisor(llm_response='{"intent": "evolution", "keywords": []}')
        result = self._run(sup._classify_intent(_state("雷峰塔在白蛇传叙事中象征着什么")))
        self.assertEqual(result["intent"], "literature_qa")

    # ---- 护栏：放行分支 ----

    def test_evolution_passes_for_real_evolution_query(self):
        """真演化查询（含演变语义）→ 放行 evolution。"""
        sup = _make_supervisor(llm_response=(
            '{"intent": "evolution", "keywords": ["雷峰塔"], '
            '"route_reason": "要求追踪叙事演变并解释成因", '
            '"evolution_scenario": "trace", "evolution_target": "雷峰塔", '
            '"evolution_target_type": "location"}'
        ))
        result = self._run(sup._classify_intent(_state("追踪雷峰塔的叙事演变，分析其变化的成因")))
        self.assertEqual(result["intent"], "evolution")
        self.assertEqual(result["evolution_target"], "雷峰塔")

    def test_evolution_guard_uses_history_context(self):
        """省略句：当前消息无演化词但对话历史有 → 放行（历史参与护栏判定）。"""
        sup = _make_supervisor(llm_response='{"intent": "evolution", "keywords": []}')
        result = self._run(sup._classify_intent(_state(
            "那它后来呢",
            history=[{"role": "user", "content": "追踪断桥母题在朝代间的变迁"}],
        )))
        self.assertEqual(result["intent"], "evolution")

    # ---- 白名单与降级 ----

    def test_invalid_intent_falls_back(self):
        """LLM 输出白名单外的意图 → 回退 literature_qa。"""
        sup = _make_supervisor(llm_response='{"intent": "chat_gpt_mode", "keywords": []}')
        result = self._run(sup._classify_intent(_state("随便聊聊白蛇传")))
        self.assertEqual(result["intent"], "literature_qa")

    def test_llm_failure_degrades(self):
        """LLM 异常 → 降级 literature_qa + 降级说明。"""
        sup = _make_supervisor(llm_error=LLMFatalError("boom"))
        result = self._run(sup._classify_intent(_state("白蛇传讲了什么")))
        self.assertEqual(result["intent"], "literature_qa")
        self.assertIn("降级", result["route_reason"])

    # ---- route_reason 透传与正常路由 ----

    def test_route_reason_passthrough(self):
        """正常分类：route_reason 原样透传（前端展示依赖此字段）。"""
        sup = _make_supervisor(llm_response=(
            '{"intent": "literature_qa", "keywords": ["许仙", "法海"], '
            '"route_reason": "共现段落清单由通用检索直接回答"}'
        ))
        result = self._run(sup._classify_intent(
            _state("许仙与法海在哪些段落同时出现？")))
        self.assertEqual(result["intent"], "literature_qa")
        self.assertEqual(result["route_reason"], "共现段落清单由通用检索直接回答")

    def test_ui_command_respected(self):
        """UI 指令类路径按 LLM 判断直达（open_chapter 在白名单内）。"""
        sup = _make_supervisor(llm_response='{"intent": "open_chapter", "keywords": []}')
        result = self._run(sup._classify_intent(_state("帮我打开第六章")))
        self.assertEqual(result["intent"], "open_chapter")


if __name__ == "__main__":
    unittest.main(verbosity=2)
