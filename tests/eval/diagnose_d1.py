"""D1 检索诊断：五段拆解，定位召回丢失在哪一段。

对每条查询逐环节统计相关记忆的命中情况：
1. 字面候选 (BM25 top-50)          —— 生产 _candidate_search(fts_top_k=50)
2. 向量候选 (top-50)              —— 生产 vec_k*5 = 10*5 = 50
3. 并集候选 (召回上限)             —— 字面 ∪ 向量
4. 融合排序 top-10 (rerank 前)     —— 复现融合公式 + 排序 + [:10]
5. rerank 后 top-10 (最终)         —— 生产 _rerank 重排 + [:10]

目的：把「字面召回 84.4% → 融合 51.6%」中间 33% 的丢失，拆成
「候选池没捞到 / 融合排序排错 / reranker 排错」三段，用数据定位主凶。

注意：本脚本与 evaluate_d1.py 用同一套 BM25 + reranker 装配，
字面检索与评测同源（修复此前诊断用 FTS5、评测用 BM25 的不一致）。
"""

import asyncio
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path

_eval_dir = Path(__file__).resolve().parent
_project_root = _eval_dir.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_eval_dir))

from common import read_json


class DummyLLM:
    async def generate(self, *args, **kwargs):
        class R:
            content = ""
        return R()


async def main():
    data = read_json("d1_retrieval.json")
    memories = data["memories"]
    queries = data["queries"]

    from agent.server.rag.config import rag_config
    from agent.server.rag.embedder import Embedder
    from agent.server.rag.bm25_retriever import BM25Retriever
    from agent.server.rag.reranker import Reranker
    from agent.server.context.wiring import build_memory_system

    use_reranker = os.getenv("USE_RERANKER", "1") == "1"

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
    retriever = ms.hybrid_retriever

    id_map = {}
    for m in memories:
        mid = ms.recall_store.add_memory(
            m["content"], memory_type=m["memory_type"], importance=m["importance"])
        if mid:
            id_map[m["id"]] = mid

    retriever.rebuild_bm25_index()
    if reranker is not None:
        for _ in range(120):
            if reranker.is_ready:
                break
            await asyncio.sleep(1)
    print(f"  写入 {len(id_map)} 条记忆，BM25 built={getattr(bm25, '_built', False)}，"
          f"reranker 就绪={reranker.is_ready if reranker else '关闭'}")

    print("[3/3] 五段拆解诊断...\n")

    FTS_K = 50    # 生产 _search_memories 的 fts_top_k (DEFAULT_FTS_TOP_K)
    VEC_K = 50    # 生产 vec_k*5 = 10*5
    FINAL_K = 10  # 生产 vec_k = 10

    # 每段聚合：命中相关记忆总数（recall 用）+ 至少命中 1 条的查询数（hit 用）
    stage_rel = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    stage_hit = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    total_rel = 0
    valid = 0

    detail = []  # 环节3有、环节5没有的丢失样例

    for q in queries:
        relevant = [id_map[r] for r in q["relevant_ids"] if r in id_map]
        if not relevant:
            continue
        valid += 1
        total_rel += len(relevant)
        rel_set = set(relevant)

        # 环节1: 字面候选 BM25 top-50
        bm25_c = retriever._candidate_search(q["query"], FTS_K, "memory_extraction")
        bm25_ids = [c["message_id"] for c in bm25_c]

        # 环节2: 向量候选 top-50
        qe = embedder.embed([q["query"]])[0]
        chroma_results = retriever._memory_collection.query(
            query_embeddings=[qe],
            n_results=min(VEC_K, retriever._memory_collection.count()),
        )
        vec_c = retriever._format_chroma_results(chroma_results)
        vec_ids = [d["message_id"] for d in vec_c]

        # 环节3: 并集候选（召回上限）
        union_ids = list(dict.fromkeys(bm25_ids + vec_ids))

        # 环节4: 融合排序 top-10（rerank 前，直接复用生产的 _fuse_candidates）
        fused = retriever._fuse_candidates(bm25_c, vec_c)
        fused.sort(key=lambda x: x["score"], reverse=True)
        fused_ids = [m["message_id"] for m in fused[:FINAL_K]]

        # 环节5: rerank 后 top-10（最终）
        reranked = retriever._rerank(q["query"], fused, FINAL_K)
        final_ids = [m["message_id"] for m in reranked[:FINAL_K]]

        stage_ids = {1: bm25_ids, 2: vec_ids, 3: union_ids, 4: fused_ids, 5: final_ids}
        for s in range(1, 6):
            h = set(stage_ids[s]) & rel_set
            stage_rel[s] += len(h)
            stage_hit[s] += 1 if h else 0

        # 丢失样例：环节3捞到、环节5没捞回的相关记忆
        lost = rel_set & set(union_ids) - set(final_ids)
        if lost:
            detail.append((q["query"][:40], len(relevant),
                           len(rel_set & set(bm25_ids)),
                           len(rel_set & set(vec_ids)),
                           len(lost)))

    names = {1: "1 字面候选 BM25 top50",
             2: "2 向量候选 top50",
             3: "3 并集候选(召回上限)",
             4: "4 融合排序 top10",
             5: "5 rerank后 top10(最终)"}

    print(f"查询数(有效): {valid}，相关记忆总数: {total_rel}\n")
    print(f"{'环节':<24}{'查询命中率':>10}{'相关召回率':>10}")
    for s in range(1, 6):
        print(f"{names[s]:<24}{stage_hit[s] / valid:>10.4f}{stage_rel[s] / total_rel:>10.4f}")

    print("\n=== 丢失定位（相关召回率差值） ===")
    print(f"  候选池缺口 (1.0 - 环节3):   {1.0 - stage_rel[3] / total_rel:>6.4f}  (相关记忆连并集候选都没进)")
    print(f"  融合排序丢失 (环节3-环节4):  {stage_rel[3] / total_rel - stage_rel[4] / total_rel:>6.4f}  (进了候选但融合排错)")
    print(f"  reranker丢失 (环节4-环节5):  {stage_rel[4] / total_rel - stage_rel[5] / total_rel:>6.4f}  (融合排对但rerank又排错)")

    print("\n=== 环节3有、环节5没有的丢失样例 ===")
    print("(query | 相关数 | 字面命中 | 向量命中 | 最终丢失)")
    for d in detail[:20]:
        print(f"  {d}")

    await ms.context_manager.close()
    if ms.scheduler is not None:
        await ms.scheduler.stop()


if __name__ == "__main__":
    asyncio.run(main())
