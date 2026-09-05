"""候选池大小扫参 — 验证 dense/sparse 的 candidate_k 是否值得扩大。

语料仅 427 chunk，当前 dense_candidate_k=100、sparse_candidate_k=100。
扫 candidate_k ∈ {100, 200, 427} 的多种组合，看 baseline（不带 boost）
recall@10 / zero / MRR 的变化，判断扩大候选池是否有收益。

Usage:
    python rag_eval/candidate_k_sweep.py --datasets academic_deepseek
"""
import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
for p in [str(PROJECT_ROOT), str(AGENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

GT_DIR = Path(__file__).resolve().parent / "ground_truth"
ALL_DATASETS = ["academic_deepseek", "academic_qwen", "daily_deepseek", "daily_qwen"]
# (dense_candidate_k, sparse_candidate_k) 组合
COMBOS = [
    (100, 100),  # 基线
    (200, 100), (427, 100),  # dense 扩大
    (100, 200), (100, 427),  # sparse 扩大
    (200, 200), (427, 427),  # 都扩大
]


def _load_dataset(ds: str):
    from rag_eval.loader import load_corpus, load_queries, load_qrels
    corpus = load_corpus(GT_DIR / "corpus.jsonl")
    queries = load_queries(GT_DIR / f"queries_{ds}.jsonl")
    qrels = load_qrels(GT_DIR / f"qrels_{ds}.tsv")
    return queries, qrels


def _eval_baseline(queries, qrels, retriever, expander):
    """baseline（不带 boost/filters）的 recall@10 / zero / MRR。"""
    from rag_eval.metrics import recall_at_k, mrr
    from server.rag.query_intent import research_penalty_for

    recall_l = []
    mrr_l = []
    zero = 0

    async def _run():
        nonlocal zero
        total = len(queries)
        for idx, q in enumerate(queries):
            qid = q.get("query_id", "")
            text = q.get("text", "")
            rels = qrels.get(qid, {})
            if not rels:
                continue
            rel_set = set(rels.keys())
            expand_result = expander.expand_weighted(text)
            eq_dict = {
                "dense_query": text,
                "sparse_query": expand_result["expanded_query"],
                "sparse_weights": expand_result["token_weights"],
                "research_penalty": research_penalty_for(text),
            }
            results = await retriever.retrieve(eq_dict, top_k=10)
            retrieved = [r.get("chunk_id", "") for r in results]
            rec = recall_at_k(retrieved, rel_set, 10)
            recall_l.append(rec)
            mrr_l.append(mrr(retrieved, rel_set))
            if rec == 0:
                zero += 1
            if (idx + 1) % 200 == 0:
                print(f"    [{idx + 1}/{total}] done")

    asyncio.run(_run())
    mean_recall = sum(recall_l) / len(recall_l) if recall_l else 0.0
    mean_mrr = sum(mrr_l) / len(mrr_l) if mrr_l else 0.0
    return mean_recall, mean_mrr, zero


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="默认全部 4 个数据集")
    args = parser.parse_args()
    datasets = args.datasets or ALL_DATASETS

    from rag_eval.query_analysis import build_retriever

    print("[INIT] 构建检索器...")
    retriever, expander = build_retriever()

    for ds in datasets:
        queries, qrels = _load_dataset(ds)
        print(f"\n== {ds} (queries {len(queries)}) ==")
        print(f"  {'dense_k':>8}{'sparse_k':>10}{'R@10':>10}{'zero':>8}{'MRR':>10}")

        for dk, sk in COMBOS:
            retriever.config.dense_candidate_k = dk
            retriever.config.sparse_candidate_k = sk
            retriever._query_cache.clear()
            r, m, z = _eval_baseline(queries, qrels, retriever, expander)
            mark = "" if (dk, sk) == (100, 100) else ""
            print(f"  {dk:>8}{sk:>10}{r:>10.4f}{z:>8}{m:>10.4f}{mark}")


if __name__ == "__main__":
    main()
