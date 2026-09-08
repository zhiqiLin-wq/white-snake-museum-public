"""智能标注实时进度系统的回归测试（测试先行）。

覆盖进度反馈契约：
1. discovery_pass 的 progress_callback 收到结构化三元组 (done, total, found)
   - done 从 1 单调递增到 total（窗口总数）
   - found 为累计发现实体数（非递减）
   - 最后一次回调 done == total，且回调次数 == 窗口数
2. 空输入 fast-fail：paragraphs / categories 为空抛 ValueError
3. 进度回调必须是可 await 的协程函数调用（实时反馈不允许丢失）

运行: python -m pytest server/rag/tests/test_annotation_progress.py -v
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    import pytest
except ImportError:  # pragma: no cover - venv 未装 pytest 时的最小 shim
    class _SkipTest(Exception):
        pass

    class _PytestShim:
        @staticmethod
        def skip(msg=""):
            raise _SkipTest(msg)

    pytest = _PytestShim()

from server.mcp.tools.annotation_pipeline import discovery_pass


# ------------------------------------------------------------------ mocks

class _FakeResp:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    """每个窗口稳定返回 1 个候选实体。"""

    def __init__(self, entity_per_window: int = 1):
        self.entity_per_window = entity_per_window
        self.calls = 0

    async def generate(self, system, user, **kwargs):
        self.calls += 1
        discoveries = [
            {
                "entity": f"许仙{self.calls}_{i}",
                "category": "person",
                "paragraph_index": 0,
                "explanation": "fake discovery",
            }
            for i in range(self.entity_per_window)
        ]
        return _FakeResp(json.dumps({"discoveries": discoveries}, ensure_ascii=False))


class _FakePromptRegistry:
    def render(self, name, context):
        return ("system-prompt", "user-prompt")


def _make_paragraphs(n: int):
    return [{"index": i, "text": f"这是第 {i} 段正文，许仙去了杭州。"} for i in range(n)]


# ------------------------------------------------------------------ tests

def test_progress_callback_emits_structured_tuples():
    """progress_callback 必须收到 (done:int, total:int, found:int) 三元组。"""
    llm = _FakeLLM(entity_per_window=2)
    paragraphs = _make_paragraphs(6)
    categories = ["person", "location"]

    calls = []

    async def on_progress(done, total, found):
        calls.append((done, total, found))

    result = asyncio.run(discovery_pass(
        paragraphs, categories, llm, _FakePromptRegistry(),
        window_size=3, overlap=1, concurrency=2,
        progress_callback=on_progress,
    ))

    # 6 段 / window_size=3 / overlap=1 -> 至少 2 个窗口
    assert len(calls) >= 2, "每个窗口完成时都必须实时回调一次"

    # 每次回调都是三个 int
    for done, total, found in calls:
        assert isinstance(done, int) and isinstance(total, int) and isinstance(found, int)

    # done 严格单调递增，从 1 到 total
    dones = [c[0] for c in calls]
    assert dones == list(range(1, len(calls) + 1)), f"done 必须从 1 连续递增: {dones}"

    # total 全程恒定，且最后一次 done == total
    totals = {c[1] for c in calls}
    assert len(totals) == 1
    total = totals.pop()
    assert calls[-1][0] == total == len(calls)

    # found 非递减，且最终 found == LLM 产出总数（每窗口 2 个 × total 窗口）
    founds = [c[2] for c in calls]
    assert all(b >= a for a, b in zip(founds, founds[1:]))
    assert founds[-1] == 2 * total

    # 去重后实体全部带回（每窗口实体名唯一，不去重）
    assert len(result) == 2 * total


def test_progress_found_accumulates_unique_entities():
    """found 为累计原始发现数（去重前），结果为去重后列表。"""
    llm = _FakeLLM(entity_per_window=1)
    paragraphs = _make_paragraphs(4)

    calls = []

    async def on_progress(done, total, found):
        calls.append((done, total, found))

    result = asyncio.run(discovery_pass(
        paragraphs, ["person"], llm, _FakePromptRegistry(),
        window_size=2, overlap=0, concurrency=1,
        progress_callback=on_progress,
    ))

    total = calls[-1][1]
    assert calls[-1][2] == total  # 每窗口 1 个
    assert len(result) == total


def test_discovery_pass_fast_fails_on_empty_input():
    async def _cb(done, total, found):
        pass

    def _expect_value_error(coro):
        try:
            asyncio.run(coro)
        except ValueError:
            return
        raise AssertionError("expected ValueError for empty input")

    _expect_value_error(discovery_pass(
        [], ["person"], _FakeLLM(), _FakePromptRegistry(),
        progress_callback=_cb,
    ))
    _expect_value_error(discovery_pass(
        _make_paragraphs(2), [], _FakeLLM(), _FakePromptRegistry(),
        progress_callback=_cb,
    ))


def test_progress_callback_is_optional():
    """不传 progress_callback 时流程正常完成（向后兼容）。"""
    llm = _FakeLLM(entity_per_window=1)
    result = asyncio.run(discovery_pass(
        _make_paragraphs(3), ["person"], llm, _FakePromptRegistry(),
        window_size=2, overlap=0, concurrency=1,
    ))
    assert len(result) >= 1
    assert llm.calls >= 1


if __name__ == "__main__":
    test_progress_callback_emits_structured_tuples()
    test_progress_found_accumulates_unique_entities()
    test_discovery_pass_fast_fails_on_empty_input()
    test_progress_callback_is_optional()
    print("all annotation progress tests passed")
