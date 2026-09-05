"""融合排序公式扫描：量化「候选池已定前提下，不同排序函数的 Recall@10 上限」。

回答的核心问题：换融合公式到底能不能救 Recall，还是瓶颈在信号本身（要换排序器）。

对每条查询，候选池 = 字面 top-50 ∪ 向量 top-50（与生产 _search_memories 一致），
对同一候选池试不同排序函数取 top-10，对比 Recall@10 / MRR：

- oracle:              相关记忆全排最前（理论上界）
- pure_vec:            纯向量 cos 相似度
- pure_bm25_score:     纯 BM25 原始分（保留长尾分辨率）
- pure_bm25_rank:      纯 BM25 排名倒数（当前字面信号，丢分辨率）
- current:             生产当前公式 (vec*0.5 + 排名倒数*0.5, 单边*0.8)
- current_no_penalty:  当前公式去掉单边 0.8 降权（缺的那路按 0 加权）
- norm_w{0.1..0.9}:    min-max 归一化两路后线性加权，扫权重
- rrf_k60:             Reciprocal Rank Fusion (k=60)

只用宏平均（与 evaluate_d1.py 一致）。sanity check: current 应复现 evaluate
无 reranker 的 Recall@10 ≈ 0.5621（本脚本 reranker=None，即纯融合排序路径）。
"""

import asyncio
import os
import shutil
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


def compute(pairs, K):
    """宏平均 Recall@K + MRR@K。pairs: list[(relevant_list, ranked_ids)]。"""
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


async def main():
    data = read_json("d1_retrieval.json")
    memories = data["memories"]
    queries = data["queries"]

    from agent.server.rag.config import rag_config
    from agent.server.rag.embedder import Embedder
    from agent.server.rag.bm25_retriever import BM25Retriever
    from agent.server.context.wiring import build_memory_system
    from agent.server.context.hybrid_retriever import _expand_synonyms

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
    ms = build_memory_system(
        llm=DummyLLM(), embedder=embedder, chroma_client=client,
        memory_db_path=mem_db, context_db_path=ctx_db,
        fifo_max_tokens=90000, summary_model="", memory_extraction_model="",
        bm25_retriever=bm25, reranker=None,
    )
    retriever = ms.hybrid_retriever

    id_map = {}
    for m in memories:
        mid = ms.recall_store.add_memory(
            m["content"], memory_type=m["memory_type"], importance=m["importance"])
        if mid:
            id_map[m["id"]] = mid

    retriever.rebuild_bm25_index()
    print(f"  写入 {len(id_map)} 条记忆，BM25 built={getattr(bm25, '_built', False)}")

    print("[3/3] 扫描融合排序公式...\n")

    FTS_K = 50
    VEC_K = 50

    scorer_names = ["oracle", "pure_vec", "pure_bm25_score", "pure_bm25_rank",
                    "current", "current_no_penalty", "rrf_k60"]
    scorer_names += [f"norm_w{w}" for w in (0.1, 0.3, 0.5, 0.7, 0.9)]
    pairs_by_scorer = {name: [] for name in scorer_names}

    for q in queries:
        relevant = [id_map[r] for r in q["relevant_ids"] if r in id_map]
        if not relevant:
            continue
        rel_set = set(relevant)

        # 字面候选（含同义词扩展），保留 BM25 原始分
        expanded = _expand_synonyms(q["query"])
        bm25_results = retriever._bm25.search(
            expanded, top_k=FTS_K, metadata_filters={"source": "memory_extraction"})
        bm25_ids = [r["metadata"]["message_id"] for r in bm25_results]
        bm25_score = {r["metadata"]["message_id"]: r["bm25_score"] for r in bm25_results}
        bm25_rank = {mid: 1.0 - i / len(bm25_ids) for i, mid in enumerate(bm25_ids)}
        bm25_pos = {mid: i + 1 for i, mid in enumerate(bm25_ids)}

        # 向量候选
        qe = embedder.embed([q["query"]])[0]
        chroma_results = retriever._memory_collection.query(
            query_embeddings=[qe],
            n_results=min(VEC_K, retriever._memory_collection.count()),
        )
        vec_c = retriever._format_chroma_results(chroma_results)
        vec_ids = [d["message_id"] for d in vec_c]
        vec_score = {d["message_id"]: d["score"] for d in vec_c}
        vec_pos = {mid: i + 1 for i, mid in enumerate(vec_ids)}

        union_ids = list(dict.fromkeys(bm25_ids + vec_ids))

        # 归一化参数（per-query min-max）
        bmin, bmax = min(bm25_score.values()), max(bm25_score.values())
        vmin, vmax = min(vec_score.values()), max(vec_score.values())

        def s_pure_vec(mid):
            return vec_score.get(mid, 0.0)

        def s_pure_bm25_score(mid):
            return bm25_score.get(mid, 0.0)

        def s_pure_bm25_rank(mid):
            return bm25_rank.get(mid, 0.0)

        def s_current(mid):
            vs = vec_score.get(mid)
            bs = bm25_rank.get(mid)
            if vs is not None and bs is not None:
                return vs * 0.5 + bs * 0.5
            if bs is not None:
                return bs * 0.8
            return vs * 0.8

        def s_current_no_penalty(mid):
            vs = vec_score.get(mid, 0.0)
            bs = bm25_rank.get(mid, 0.0)
            return vs * 0.5 + bs * 0.5

        def s_rrf(mid, k=60):
            vp = vec_pos.get(mid, 1000)
            bp = bm25_pos.get(mid, 1000)
            return 1.0 / (k + vp) + 1.0 / (k + bp)

        def make_norm(w):
            def f(mid):
                vs = vec_score.get(mid)
                bs = bm25_score.get(mid)
                nv = (vs - vmin) / (vmax - vmin) if vs is not None and vmax > vmin else 0.0
                nb = (bs - bmin) / (bmax - bmin) if bs is not None and bmax > bmin else 0.0
                return w * nv + (1.0 - w) * nb
            return f

        scorers = {
            "pure_vec": s_pure_vec,
            "pure_bm25_score": s_pure_bm25_score,
            "pure_bm25_rank": s_pure_bm25_rank,
            "current": s_current,
            "current_no_penalty": s_current_no_penalty,
            "rrf_k60": s_rrf,
        }
        for w in (0.1, 0.3, 0.5, 0.7, 0.9):
            scorers[f"norm_w{w}"] = make_norm(w)

        # oracle: 相关记忆全排最前
        oracle_ranked = ([r for r in relevant if r in union_ids]
                         + [m for m in union_ids if m not in rel_set])
        pairs_by_scorer["oracle"].append((relevant, oracle_ranked))

        for name, scorer in scorers.items():
            ranked = sorted(union_ids, key=scorer, reverse=True)
            pairs_by_scorer[name].append((relevant, ranked))

    print(f"{'排序函数':<22}{'Recall@1':>10}{'Recall@5':>10}{'Recall@10':>10}{'MRR':>10}")
    for name in scorer_names:
        r1, _ = compute(pairs_by_scorer[name], 1)
        r5, _ = compute(pairs_by_scorer[name], 5)
        r10, mrr = compute(pairs_by_scorer[name], 10)
        print(f"{name:<22}{r1:>10.4f}{r5:>10.4f}{r10:>10.4f}{mrr:>10.4f}")

    await ms.context_manager.close()
    if ms.scheduler is not None:
        await ms.scheduler.stop()


if __name__ == "__main__":
    asyncio.run(main())
