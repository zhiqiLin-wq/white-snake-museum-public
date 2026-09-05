"""融合权重扫描：向量权重 w 从 0.0 到 1.0 步长 0.1，看哪个权重下检索指标最优。

记忆只写入一次，对每个 w 新建 HybridRetriever（复用 memory_db/chroma/bm25/reranker），
跑检索算 MRR/Recall@K。
"""

import asyncio
import math
import shutil
import sqlite3
import sys
from pathlib import Path

_eval_dir = Path(__file__).resolve().parent
_project_root = _eval_dir.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_eval_dir))

from common import read_json
from agent.server.rag.config import rag_config
from agent.server.rag.embedder import Embedder
from agent.server.rag.bm25_retriever import BM25Retriever
from agent.server.rag.reranker import Reranker
from agent.server.context.wiring import build_memory_system
from agent.server.context.hybrid_retriever import HybridRetriever


class DummyLLM:
    async def generate(self, *args, **kwargs):
        class R:
            content = ""
        return R()


def _compute(pairs, K):
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0
    recall = 0.0
    mrr = 0.0
    for relevant, ranked in pairs:
        relevant = set(relevant)
        top = ranked[:K]
        matched = [i for i, rid in enumerate(top) if rid in relevant]
        recall += len(set(top) & relevant) / len(relevant) if relevant else 0.0
        if matched:
            mrr += 1.0 / (matched[0] + 1)
    return recall / n, mrr / n


def _reset_access(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("UPDATE long_term_memories SET last_accessed_at = NULL, access_count = 0")
        conn.commit()
    finally:
        conn.close()


async def main():
    data = read_json("d1_retrieval.json")
    memories = data["memories"]
    queries = data["queries"]

    print("[1/3] 加载 embedder...")
    embedder = Embedder(model_name=rag_config.embedding_model,
                        local_path=rag_config.embedding_local_path or None)
    embedder.start_loading()
    for _ in range(60):
        if embedder.is_ready:
            break
        await asyncio.sleep(1)

    import chromadb
    chroma_dir = _eval_dir / "chroma_eval"
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir, ignore_errors=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    mem_db = _eval_dir / "eval_memory.db"
    ctx_db = _eval_dir / "eval_context.db"
    for p in (mem_db, ctx_db):
        if p.exists():
            p.unlink()

    print("[2/3] 装配 + 写入记忆...")
    bm25 = BM25Retriever()
    reranker = Reranker(model_name=rag_config.reranker_model, backend=rag_config.reranker_backend)
    reranker.start_loading()

    ms = build_memory_system(
        llm=DummyLLM(), embedder=embedder, chroma_client=client,
        memory_db_path=mem_db, context_db_path=ctx_db,
        fifo_max_tokens=90000, summary_model="", memory_extraction_model="",
        bm25_retriever=bm25, reranker=reranker,
    )
    store = ms.recall_store

    id_map = {}
    for m in memories:
        mid = store.add_memory(m["content"], memory_type=m["memory_type"], importance=m["importance"])
        if mid:
            id_map[m["id"]] = mid
    ms.hybrid_retriever.rebuild_bm25_index()

    for _ in range(120):
        if reranker.is_ready:
            break
        await asyncio.sleep(1)
    print(f"  写入 {len(id_map)} 记忆，reranker 就绪: {reranker.is_ready}")

    # 预计算 relevant（实际 message_id）
    relevant_map = []
    for q in queries:
        rel = [id_map[r] for r in q["relevant_ids"] if r in id_map]
        relevant_map.append((q["query"], rel))

    print("[3/3] 扫描权重...\n")
    print(f"{'w(向量)':>8}{'Recall@1':>10}{'Recall@5':>10}{'Recall@10':>10}{'MRR':>10}")
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        retriever = HybridRetriever(
            memory_db=ms.memory_db, embedder=embedder, chroma_client=client,
            bm25_retriever=bm25, reranker=reranker, fusion_weight=w,
        )
        pairs = []
        for query, rel in relevant_map:
            if not rel:
                continue
            _reset_access(mem_db)
            res = retriever.search(query, source="memories", score_threshold=0.0, vector_top_k=10)
            ranked = [m["message_id"] for m in res["memories"]]
            pairs.append((rel, ranked))
        r1, mrr1 = _compute(pairs, 1)
        r5, mrr5 = _compute(pairs, 5)
        r10, mrr10 = _compute(pairs, 10)
        print(f"{w:>8.1f}{r1:>10.4f}{r5:>10.4f}{r10:>10.4f}{mrr1:>10.4f}")

    await ms.context_manager.close()
    if ms.scheduler is not None:
        await ms.scheduler.stop()


if __name__ == "__main__":
    asyncio.run(main())
