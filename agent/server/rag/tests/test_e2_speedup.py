"""E2 矩阵扫描提速补丁的回归测试（测试先行）。

覆盖四个补丁的正确性门禁：
1. embedder 动态 padding 与固定 max_length 产出向量一致（cosine≈1）——质量门禁
2. TagRetriever 空标签（E2 skip_tagging 路径）不再触发标签语义通道的无效 embedding
3. TagRetriever.retrieve 的重活在工作线程执行，不阻塞事件循环（并发可重叠）
4. E2 单元格扫描对相同 (query, filters, top_k) 的检索跨朝代去重复用

运行: python -m pytest server/rag/tests/test_e2_speedup.py -v
"""
import asyncio
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

class _SkipTest(Exception):
    pass


try:
    import pytest
except ImportError:
    # venv 未装 pytest 时的最小 shim，支持 python 直接运行本文件
    class _PytestShim:
        @staticmethod
        def skip(msg=""):
            raise _SkipTest(msg)

    pytest = _PytestShim()

from server.rag.tag_retriever import TagRetriever


# ---------------------------------------------------------------- mocks

class MockEmbedder:
    """计数 + 可阻塞的假 embedder。"""

    def __init__(self, block: float = 0.0, dim: int = 32):
        self.embed_calls = 0
        self.embed_query_calls = 0
        self.block = block
        self.dim = dim

    def embed(self, texts):
        self.embed_calls += 1
        if self.block:
            time.sleep(self.block)
        return [np.ones(self.dim, dtype=float).tolist() for _ in texts]

    def embed_query(self, text):
        self.embed_query_calls += 1
        if self.block:
            time.sleep(self.block)
        return np.ones(self.dim, dtype=float).tolist()


class MockVS:
    def query(self, qvec, top_k=100, where=None):
        return [{"chunk_id": "c1", "score": 0.9, "content": "x",
                 "metadata": {"dynasty": "唐"}}]


class MockTagStore:
    def __init__(self, chunk_ids=()):
        self.chunk_ids = list(chunk_ids)

    def tag_vector(self, cid):
        return np.ones(32, dtype=float)

    def summary_vector(self, cid):
        return None

    def summary_text(self, cid):
        return None

    def norm_label(self, cid):
        return {}

    def idf(self, term):
        return 0.0

    plot_unit_sim = None


# ---------------------------------------------------------------- tests

def test_dynamic_padding_matches_max_length():
    """补丁1 门禁：动态 padding 与固定 512 padding 向量 cosine >= 0.9999。"""
    from server.rag.embedder import Embedder, BGE_QUERY_INSTRUCTION

    emb = Embedder("BAAI/bge-large-zh-v1.5")
    emb.start_loading()
    deadline = time.time() + 150
    while not emb.is_ready and time.time() < deadline:
        time.sleep(1)
    if not emb.is_ready:
        pytest.skip(f"embedding model unavailable: {emb.load_error}")

    import torch

    def fwd(texts, padding):
        enc = emb._tokenizer(
            [BGE_QUERY_INSTRUCTION + t for t in texts],
            padding=padding, truncation=True, max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = emb._model(**enc)
        cls = torch.nn.functional.normalize(
            out.last_hidden_state[:, 0, :], p=2, dim=1)
        return cls.numpy()

    q = "雷峰塔 镇压 白娘子 永镇西湖 塔下祭塔"
    v_fixed = fwd([q], "max_length")[0]
    v_dyn = fwd([q], True)[0]
    v_mix = fwd([q, "话说南宋年间临安府有一官人姓许名仙" * 20], True)[0]

    assert float(np.dot(v_fixed, v_dyn)) >= 0.9999, "单条动态 padding 向量偏移"
    assert float(np.dot(v_fixed, v_mix)) >= 0.9999, "混批动态 padding 向量偏移"


def test_empty_label_skips_tag_embedding():
    """补丁2 门禁：空标签（skip_tagging）只做 1 次 dense embedding，
    不再对空标签文本白跑一次标签语义 embedding。"""
    emb = MockEmbedder()
    tr = TagRetriever(MockVS(), emb, MockTagStore(chunk_ids=["c1"]))

    ranked = asyncio.run(tr.retrieve(
        "雷峰塔 镇压 白娘子", {}, top_k=5,
        corpus=None, meta_lookup=None, return_scores=True,
    ))

    assert ranked, "应返回 dense 路召回结果"
    assert emb.embed_query_calls == 1, f"dense embedding 应恰好 1 次, 实际 {emb.embed_query_calls}"
    assert emb.embed_calls == 0, f"空标签不应触发标签/summary embedding, 实际 {emb.embed_calls} 次"


def test_retrieve_does_not_block_event_loop():
    """补丁3 门禁：两次 retrieve 的阻塞计算在线程池重叠执行（~0.3s），
    若在事件循环串行则 ~0.6s+。"""
    emb = MockEmbedder(block=0.3)
    tr = TagRetriever(MockVS(), emb, MockTagStore(chunk_ids=[]))

    async def run():
        return await asyncio.gather(*[
            tr.retrieve(f"查询{i}", {}, top_k=5, return_scores=True)
            for i in range(2)
        ])

    t0 = time.perf_counter()
    asyncio.run(run())
    dt = time.perf_counter() - t0

    assert dt < 0.55, f"retrieve 疑似仍在事件循环串行执行: {dt:.2f}s"


def test_e2_retrieve_cache_dedup():
    """补丁4 门禁：同一母题跨朝代扫描时，无朝代过滤的宽查询（L4/L5）
    查询文本与过滤条件完全相同，应跨单元格去重复用。

    5 级 × 2 朝 = 10 次检索；L4/L5 各复用 1 次 → 8 次。
    """
    from server.langgraph.evolution_workbench import EvolutionWorkbench

    stub = types.SimpleNamespace()
    calls = []

    def make_doc(dynasty, variant):
        # 内容按 (朝代, 级别变体) 稳定（同一 chunk 无论哪级检索返回内容一致，
        # _scan_one_cell 按 content[:80] 去重；不同级别可召回不同段落）
        return {"content": f"证据-{dynasty}-{variant}",
                "metadata": {"dynasty": dynasty, "chapter_title": "x"}}

    async def fake_filtered_retrieve(query, filters=None, top_k=None, skip_tagging=False):
        calls.append((query, tuple(sorted((filters or {}).items())), top_k))
        dy = (filters or {}).get("dynasty")
        if dy:
            # 过滤级：L1（关键词+母题名+core）返回本朝 1 条；L2/L3 返回空
            if query.startswith("KW 镇塔 镇塔"):
                return [make_doc(dy, "过滤级段落甲")]
            return []
        # 无过滤宽查询（L4/L5）：返回两朝文档，由各格子本地过滤
        return [make_doc("唐", "宽查段落乙"), make_doc("宋", "宽查段落乙")]

    stub._filtered_retrieve = fake_filtered_retrieve
    cached = EvolutionWorkbench._make_cached_retriever(stub)

    motif = {"id": "M1", "name": "镇塔", "core": "镇压囚禁",
             "_search_keywords": "KW 镇塔"}

    async def run():
        return await asyncio.gather(*[
            EvolutionWorkbench._scan_one_cell(stub, motif, d, retrieve_fn=cached)
            for d in ("唐", "宋")
        ])

    results = asyncio.run(run())

    assert [r["count"] for r in results] == [2, 2], \
        f"两格各应命中 2 条（L1+L4），实际 {[r['count'] for r in results]}"
    assert len(calls) == 8, \
        f"L4/L5 宽查询跨朝去重后应 8 次检索，实际 {len(calls)} 次"


# 无 pytest 时的独立运行入口
if __name__ == "__main__":
    _tests = [
        test_dynamic_padding_matches_max_length,
        test_empty_label_skips_tag_embedding,
        test_retrieve_does_not_block_event_loop,
        test_e2_retrieve_cache_dedup,
    ]
    _passed = _failed = _skipped = 0
    for t in _tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            _passed += 1
        except _SkipTest:
            print(f"SKIP  {t.__name__}（模型不可用）")
            _skipped += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            _failed += 1
    print(f"\n{_passed} passed, {_failed} failed, {_skipped} skipped")
    sys.exit(1 if _failed else 0)
