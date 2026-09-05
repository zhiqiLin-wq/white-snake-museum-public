"""D1 检索评测：在真实检索管道上评测召回指标。

流程：
1. 读 d1_retrieval.json（记忆 + 查询 + 相关性标注）
2. 用真实 build_memory_system 装配，把记忆真实写入（embed + ChromaDB + FTS）
3. 对每条查询跑 HybridRetriever.search，取 top-K
4. 比对 relevant_ids，算 Recall@K / Precision@K / Hit@K / MRR / nDCG@K

每次检索前重置 last_accessed_at，保证各查询的初始条件一致。
"""

import asyncio
import json
import math
import shutil
import sqlite3
import sys
from pathlib import Path

_eval_dir = Path(__file__).resolve().parent
_project_root = _eval_dir.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_eval_dir))

from common import read_json, DATA_DIR


class DummyLLM:
    async def generate(self, *args, **kwargs):
        class R:
            content = ""
        return R()


def compute_metrics(pairs, K):
    """pairs: list of (relevant_set, ranked_ids)。"""
    n = len(pairs)
    if n == 0:
        return {}
    recall = precision = hit = mrr = ndcg = 0.0
    for relevant, ranked in pairs:
        relevant = set(relevant)
        top = ranked[:K]
        matched = [i for i, rid in enumerate(top) if rid in relevant]
        recall += len(set(top) & relevant) / len(relevant) if relevant else 0.0
        precision += len(set(top) & relevant) / K
        hit += 1.0 if matched else 0.0
        if matched:
            mrr += 1.0 / (matched[0] + 1)
        dcg = sum(1.0 / math.log2(i + 2) for i, rid in enumerate(top) if rid in relevant)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), K)))
        ndcg += dcg / idcg if idcg > 0 else 0.0
    return {
        f"recall@{K}": round(recall / n, 4),
        f"precision@{K}": round(precision / n, 4),
        f"hit@{K}": round(hit / n, 4),
        "mrr": round(mrr / n, 4),
        f"ndcg@{K}": round(ndcg / n, 4),
    }


def _reset_access(db_path: Path):
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
    print(f"[数据] 记忆 {len(memories)} 条，查询 {len(queries)} 条")

    from agent.server.config import settings
    from agent.server.rag.config import rag_config
    from agent.server.rag.embedder import Embedder
    from agent.server.context.wiring import build_memory_system

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
    import os
    from agent.server.rag.bm25_retriever import BM25Retriever
    from agent.server.rag.reranker import Reranker
    use_reranker = os.getenv("USE_RERANKER", "1") == "1"
    bm25 = BM25Retriever()
    reranker = (Reranker(model_name=rag_config.reranker_model, backend=rag_config.reranker_backend)
                if use_reranker else None)
    if reranker is not None:
        reranker.start_loading()

    ms = build_memory_system(
        llm=DummyLLM(), embedder=embedder, chroma_client=client,
        memory_db_path=mem_db, context_db_path=ctx_db,
        fifo_max_tokens=90000, summary_model="", memory_extraction_model="",
        bm25_retriever=bm25, reranker=reranker,
    )
    store = ms.recall_store
    retriever = ms.hybrid_retriever

    id_map = {}  # 数据集 id (m001) -> 实际 message_id
    for m in memories:
        mid = store.add_memory(m["content"], memory_type=m["memory_type"], importance=m["importance"])
        if mid:
            id_map[m["id"]] = mid
    print(f"  写入 {len(id_map)}/{len(memories)} 条记忆")

    # 重建 BM25 索引 + 等待 reranker 就绪
    retriever.rebuild_bm25_index()
    if reranker is not None:
        for _ in range(120):
            if reranker.is_ready:
                break
            await asyncio.sleep(1)
    print(f"  BM25 就绪: {getattr(bm25, '_built', False)}, reranker 就绪: {reranker.is_ready if reranker else '关闭'}")

    print("[3/3] 评测检索...")
    results = {K: [] for K in (1, 3, 5, 10)}
    for q in queries:
        relevant = [id_map[r] for r in q["relevant_ids"] if r in id_map]
        if not relevant:
            continue
        _reset_access(mem_db)
        res = retriever.search(q["query"], source="memories",
                               score_threshold=0.0, vector_top_k=10)
        ranked = [m["message_id"] for m in res["memories"]]
        for K in results:
            results[K].append((relevant, ranked))

    print("\n===== 检索指标 =====")
    print(f"{'指标':<14}{'K=1':>10}{'K=3':>10}{'K=5':>10}{'K=10':>10}")
    metric_names = ["recall", "precision", "hit", "ndcg"]
    for name in metric_names:
        row = f"{name:14s}"
        for K in (1, 3, 5, 10):
            m = compute_metrics(results[K], K)
            row += f"{m[f'{name}@{K}']:>10.4f}"
        print(row)
    # MRR 与 K 无关，单独一行
    mrr_val = compute_metrics(results[1], 1)["mrr"]
    print(f"{'mrr':14s}{mrr_val:>10.4f}{'':>10}{'':>10}{'':>10}")

    await ms.context_manager.close()
    if ms.scheduler is not None:
        await ms.scheduler.stop()


if __name__ == "__main__":
    asyncio.run(main())
