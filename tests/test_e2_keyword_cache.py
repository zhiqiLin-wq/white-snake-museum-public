"""E2 关键词持久化缓存的回归测试（测试先行）。

覆盖 2026-09-09 改动：reexplore 轮次重复扫描时复用首轮生成的 LLM 关键词，
避免每轮 6 母题 × 15s 超时上限的 LLM 调用累积延迟。

设计要点:
- 缓存键: (motif_id, motif_name, motif_core) 三元组，避免 M1 在不同 target 下复用错误关键词
- 持久化: 落盘到 data/e2_keywords/{hash}.json，reexplore 轮次（同一 thread_id）命中
- 命中即跳过 LLM 调用，未命中才走 generate_keywords 原逻辑
- 指令复述/过短回退逻辑保留，缓存写入前先校验质量

运行: python tests/test_e2_keyword_cache.py  (或 pytest)
"""
import asyncio
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from agent.server.langgraph.evolution_workbench import (  # noqa: E402
    EvolutionWorkbench,
    _E2_KEYWORD_DIR,
    _e2_keyword_cache_key,
    _load_e2_keyword_cache,
    _save_e2_keyword_cache,
    _clear_e2_keyword_cache,
)
from agent.server.llm.provider import LLMResponse  # noqa: E402


def _mk_motif(mid="M1", name="镇压与囚禁", core="白娘子被镇压在塔下的母题"):
    return {"id": mid, "name": name, "core": core, "type": "invariant"}


def _mk_workbench(tmp_dir: Path, llm_call_count: list) -> EvolutionWorkbench:
    """构造最小 EvolutionWorkbench，LLM 调用计数到 llm_call_count[0]。"""
    llm = MagicMock()

    async def _fake_generate(*a, **k):
        llm_call_count[0] += 1
        return LLMResponse(
            content="雷峰塔 镇压 永镇西湖 祭塔 塔下 囚禁",
            model="test", tokens=10, completion_tokens=10, finish_reason="stop",
        )

    llm.generate = _fake_generate
    llm.generate_stream = AsyncMock()

    retriever = MagicMock()
    # 跳过实际检索：单元格返回空
    async def _fake_filtered_retrieve(*a, **k):
        return []
    retriever.retrieve = AsyncMock(return_value=[])

    wb = EvolutionWorkbench(
        llm=llm, retriever=retriever, prompt_registry=None,
        mcp_registry=None, checkpointer=None, node_client=None,
    )
    # 替换检索为空，避免单元格检索阶段额外耗时
    wb._filtered_retrieve = _fake_filtered_retrieve
    return wb


class TestE2KeywordCache(unittest.TestCase):

    def setUp(self):
        self._orig_dir = _E2_KEYWORD_DIR
        self.tmp = tempfile.mkdtemp(prefix="e2_kw_")
        # 通过 monkeypatch 把缓存目录指到临时目录
        import agent.server.langgraph.evolution_workbench as wb_mod
        wb_mod._E2_KEYWORD_DIR = Path(self.tmp)

    def tearDown(self):
        import agent.server.langgraph.evolution_workbench as wb_mod
        wb_mod._E2_KEYWORD_DIR = self._orig_dir
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, coro):
        return asyncio.run(coro)

    # ---- 缓存键设计：三元组保证不同 target 不串味 ----

    def test_cache_key_includes_id_name_core(self):
        """缓存键必须包含 id+name+core，防止同名母题在不同 target 下串味。"""
        key1 = _e2_keyword_cache_key(_mk_motif("M1", "镇压", "白蛇被镇压"))
        key2 = _e2_keyword_cache_key(_mk_motif("M1", "镇压", "许仙被镇压"))
        self.assertNotEqual(key1, key2, "不同 core 应有不同的缓存键")

        key3 = _e2_keyword_cache_key(_mk_motif("M2", "镇压", "白蛇被镇压"))
        self.assertNotEqual(key1, key3, "不同 id 应有不同的缓存键")

    # ---- 首次扫描：LLM 被调用，结果落盘 ----

    def test_first_scan_calls_llm_and_persists(self):
        llm_calls = [0]
        wb = _mk_workbench(Path(self.tmp), llm_calls)
        motif = _mk_motif()

        # 首次：生成关键词
        state = {
            "motifs": [motif], "dynasties": ["唐"],
            "reexplore_count": 0, "thread_id": "t-test",
        }
        self._run(wb._scan_motif_matrix(state))

        self.assertEqual(llm_calls[0], 1, "首次扫描应调用 LLM 1 次")
        cached = _load_e2_keyword_cache(motif)
        self.assertIsNotNone(cached, "关键词必须落盘")
        self.assertIn("雷峰塔", cached["keywords"])

    # ---- reexplore 轮次：命中缓存，不再调 LLM ----

    def test_reexplore_round_hits_cache_skips_llm(self):
        """reexplore 第二轮扫描同一母题时应命中缓存，LLM 调用次数不增加。"""
        llm_calls = [0]
        wb = _mk_workbench(Path(self.tmp), llm_calls)
        motif = _mk_motif()

        # 首轮
        state1 = {
            "motifs": [motif], "dynasties": ["唐"],
            "reexplore_count": 0, "thread_id": "t-test",
        }
        self._run(wb._scan_motif_matrix(state1))
        self.assertEqual(llm_calls[0], 1)

        # reexplore 轮次：相同 motif
        state2 = {
            "motifs": [motif], "dynasties": ["唐", "宋"],
            "reexplore_count": 1, "thread_id": "t-test",
        }
        self._run(wb._scan_motif_matrix(state2))
        self.assertEqual(llm_calls[0], 1,
                         "reexplore 应命中缓存，LLM 调用次数不应增加")

    # ---- 缓存不命中：motif 内容变化时重新生成 ----

    def test_motif_core_change_invalidates_cache(self):
        """母题 core 变化（target 不同导致分解不同）应使缓存失效。"""
        llm_calls = [0]
        wb = _mk_workbench(Path(self.tmp), llm_calls)

        motif_v1 = _mk_motif(core="白蛇被镇压在塔下")
        state1 = {"motifs": [motif_v1], "dynasties": ["唐"],
                  "reexplore_count": 0, "thread_id": "t1"}
        self._run(wb._scan_motif_matrix(state1))
        self.assertEqual(llm_calls[0], 1)

        # 同 id 但 core 变化 → 不应命中
        motif_v2 = _mk_motif(core="许仙被镇压在塔下（视角变化）")
        state2 = {"motifs": [motif_v2], "dynasties": ["唐"],
                  "reexplore_count": 0, "thread_id": "t2"}
        self._run(wb._scan_motif_matrix(state2))
        self.assertEqual(llm_calls[0], 2,
                         "core 变化应失效缓存，重新调 LLM")

    # ---- 进度事件：命中缓存也要推送 THINKING_STEP ----

    def test_cache_hit_still_emits_progress_event(self):
        """缓存命中时仍应推送「跳过」进度事件，让前端看到推进而非静止。"""
        llm_calls = [0]
        wb = _mk_workbench(Path(self.tmp), llm_calls)
        motif = _mk_motif()

        # 首轮落盘
        state1 = {"motifs": [motif], "dynasties": ["唐"],
                  "reexplore_count": 0, "thread_id": "t"}
        self._run(wb._scan_motif_matrix(state1))

        # 第二轮捕获 live 事件
        events = []
        queue = asyncio.Queue()

        def _emit(ev):
            try:
                queue.put_nowait(ev)
            except Exception:
                pass
        wb._live_queue = queue
        wb._emit_live = _emit  # type: ignore

        state2 = {"motifs": [motif], "dynasties": ["唐"],
                  "reexplore_count": 1, "thread_id": "t"}
        self._run(wb._scan_motif_matrix(state2))

        while not queue.empty():
            events.append(queue.get_nowait())

        thinking_msgs = [
            e.get("message", "") or e.get("data", {}).get("message", "")
            for e in events
            if e.get("type") == "thinking_step"
        ]
        self.assertTrue(
            any("跳过" in m or "已有" in m or "缓存" in m for m in thinking_msgs),
            f"缓存命中应推送跳过进度事件，实际: {thinking_msgs}"
        )


if __name__ == "__main__":
    unittest.main()
