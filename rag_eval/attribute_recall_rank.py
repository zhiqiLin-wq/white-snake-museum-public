"""核心答案排名归因：把「D/S 找不到」切成「差一点进 top-10」vs「根本排不上」。

背景:
    attribute_recall.py 已定位到 86% 的丢失核心答案是「D/S 的 top-10 没有」。
    但「top-10 没有」包含两种完全不同的情况，解法相反:
      sortable      : 核心 chunk 在 D/S 的 top-11~50，差一点进 top-10
                      -> 调排序 / 融合权重 / 重排 就能救
      unretrievable : 核心 chunk 连 D/S 的 top-50 都不在
                      -> 底层 dense/BM25 根本没理解，必须换模型 / 微调 / 更好的分词

本脚本重跑检索（不复用日志），对每条 query 跑:
    D: dense-only  top-50  (HybridRetriever, 严格对齐生产 research_penalty/version_boost)
    S: BM25-only  top-50  (bm25.search)
    F: 融合       top-10  (判断哪些 rel=2 核心已命中，命中则跳过)

对每个「F top-10 未命中」的 rel=2 核心 chunk，按它在 D/S 的最优排名分类:
    fusion_buried : D 或 S 的 top-1~10 有它（融合没把它带进 top-10，融合损失）
    sortable      : D 或 S 的 top-11~50 有它（差一点，调排序可救）
    unretrievable : D/S 的 top-50 都没有（根本排不上，换底层能力）

Usage:
    python rag_eval/attribute_recall_rank.py
    python rag_eval/attribute_recall_rank.py --out results/recall_rank_attribution.json
"""
import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
for p in [str(PROJECT_ROOT), str(AGENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rag_eval.detail_log import build_components, build_retrievers, preprocess_query
from rag_eval.loader import load_queries, load_qrels

DATASETS = ["academic_deepseek", "academic_qwen", "daily_deepseek", "daily_qwen"]
D_TOP_K = 50   # 排名归因分界：差一点 vs 根本排不上
S_TOP_K = 50


def rank_map(results: list[dict], top_k: int) -> dict[str, int]:
    """返回 {chunk_id: rank}，rank 从 1 开始，超出 top_k 的不收录。"""
    rmap: dict[str, int] = {}
    for i, r in enumerate(results[:top_k], start=1):
        cid = r.get("chunk_id", "")
        if cid:
            rmap[cid] = i
    return rmap


def classify(best_rank: int) -> str:
    if best_rank <= 10:
        return "fusion_buried"
    if best_rank <= 50:
        return "sortable"
    return "unretrievable"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    print("[INIT] 加载 embedder / vector_store / BM25 ...", flush=True)
    embedder, vector_store, bm25, expander = build_components()
    d_retriever, s_bm25, f_retriever = build_retrievers(vector_store, bm25)

    gt = PROJECT_ROOT / "rag_eval" / "ground_truth"

    counter = Counter()
    detail: list[dict] = []
    n_core_total = 0
    n_core_hit_f = 0

    t0 = time.monotonic()
    for ds in DATASETS:
        queries = load_queries(gt / f"queries_{ds}.jsonl")
        qrels = load_qrels(gt / f"qrels_{ds}_src_only.tsv")
        print(f"\n[{ds}] {len(queries)} queries", flush=True)

        for qi, q in enumerate(queries):
            qid = q.get("query_id", "")
            rels = qrels.get(qid, {})
            core_set = {c for c, r in rels.items() if r == 2}
            if not core_set:
                continue
            n_core_total += len(core_set)

            pp = preprocess_query(q.get("text", ""), expander)
            eq_dict = {
                "dense_query": pp["dense_query"],
                "sparse_query": pp["sparse_query"],
                "sparse_weights": pp["sparse_weights"],
                "research_penalty": pp["research_penalty"],
                "version_boost": pp["version_boost"],
            }

            # F top-10：判断哪些核心已命中
            f_results = asyncio.run(f_retriever.retrieve(eq_dict, top_k=10))
            f_top10 = {r.get("chunk_id", "") for r in f_results}

            hit_core = core_set & f_top10
            n_core_hit_f += len(hit_core)
            missed_core = core_set - f_top10
            if not missed_core:
                continue

            # D top-50 / S top-50（只对 F 未命中的 query 跑，省时间）
            d_results = asyncio.run(d_retriever.retrieve(eq_dict, top_k=D_TOP_K))
            s_results = asyncio.run(asyncio.to_thread(
                s_bm25.search, pp["sparse_query"], S_TOP_K, None, pp["sparse_weights"]
            ))
            d_rank = rank_map(d_results, D_TOP_K)
            s_rank = rank_map(s_results, S_TOP_K)

            for cid in missed_core:
                dr = d_rank.get(cid, 999)
                sr = s_rank.get(cid, 999)
                best = min(dr, sr)
                cat = classify(best)
                counter[cat] += 1
                detail.append({
                    "dataset": ds,
                    "query_id": qid,
                    "chunk_id": cid,
                    "dense_rank": dr if dr <= D_TOP_K else None,
                    "sparse_rank": sr if sr <= S_TOP_K else None,
                    "best_rank": best if best <= 50 else None,
                    "category": cat,
                })

            if (qi + 1) % 50 == 0:
                el = time.monotonic() - t0
                print(f"  [{ds}] {qi + 1}/{len(queries)} ({el:.0f}s)", flush=True)

    elapsed = time.monotonic() - t0
    n_lost = sum(counter.values())

    print("\n" + "=" * 70)
    print("核心答案排名归因 — D/S top-50 定位")
    print("=" * 70)
    print(f"rel=2 核心 chunk 总数: {n_core_total}")
    print(f"F top-10 已命中:        {n_core_hit_f} ({n_core_hit_f / n_core_total * 100:.1f}%)")
    print(f"F top-10 未命中:        {n_lost} ({n_lost / n_core_total * 100:.1f}%)")
    print()
    print("【丢失核心的三类归因】(占丢失核心的比例)")
    order = ["fusion_buried", "sortable", "unretrievable"]
    labels = {
        "fusion_buried": "D/S top-1~10 有 -> 融合没带进 top-10（融合损失）",
        "sortable": "D/S top-11~50 有 -> 差一点进 top-10（调排序/融合权重可救）",
        "unretrievable": "D/S top-50 都没有 -> 根本排不上（换模型/微调/分词）",
    }
    for cat in order:
        cnt = counter.get(cat, 0)
        pct = cnt / n_lost * 100 if n_lost else 0
        print(f"  {cat:<16} {cnt:>5}  ({pct:5.1f}%)  {labels[cat]}")
    print()
    print(f"总耗时 {elapsed:.0f}s")
    print("=" * 70)

    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps({
            "n_core_total": n_core_total,
            "n_core_hit_f": n_core_hit_f,
            "distribution": dict(counter),
            "detail": detail,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n明细 JSON 已写: {out}")


if __name__ == "__main__":
    main()
