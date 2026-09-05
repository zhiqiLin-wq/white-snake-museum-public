"""版本软 boost 系数扫参 — 找 daily 数据集的温和系数。

对每个数据集跑 baseline（无 boost）与多档 version_boost_factor 对照，
输出 recall@10 / zero / MRR，观察 daily 数据集的微负能否被更低系数压平。

Usage:
    python rag_eval/version_boost_sweep.py --datasets daily_deepseek daily_qwen
"""
import argparse
import asyncio
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
for p in [str(PROJECT_ROOT), str(AGENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

GT_DIR = Path(__file__).resolve().parent / "ground_truth"
ALL_DATASETS = ["academic_deepseek", "academic_qwen", "daily_deepseek", "daily_qwen"]
FACTORS = [1.0, 1.2, 1.5, 2.0]  # 1.0 = baseline（boost 系数 <=1 时不生效）


def _load_dataset(ds: str):
    from rag_eval.loader import load_corpus, load_queries, load_qrels
    corpus = load_corpus(GT_DIR / "corpus.jsonl")
    queries = load_queries(GT_DIR / f"queries_{ds}.jsonl")
    qrels = load_qrels(GT_DIR / f"qrels_{ds}.tsv")
    return corpus, queries, qrels


def _eval(queries, qrels, retriever, expander, use_boost: bool, factor: float):
    """对全量 query 跑检索，返回 {recall[], zero, mrr[]}。"""
    from rag_eval.metrics import recall_at_k, mrr
    from server.rag.query_intent import detect_version_filters, research_penalty_for

    retriever.config.version_boost_factor = factor
    retriever._query_cache.clear()

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

            vb = detect_version_filters(text) if use_boost else None
            expand_result = expander.expand_weighted(text)
            eq_dict = {
                "dense_query": text,
                "sparse_query": expand_result["expanded_query"],
                "sparse_weights": expand_result["token_weights"],
                "research_penalty": research_penalty_for(text),
            }
            results = await retriever.retrieve(eq_dict, top_k=10, version_boost=vb)
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
        corpus, queries, qrels = _load_dataset(ds)
        print(f"\n== {ds} (queries {len(queries)}) ==")
        print(f"  {'配置':<14}{'R@10':>10}{'zero':>8}{'MRR':>10}")

        # baseline（无 boost）
        br, bm, bz = _eval(queries, qrels, retriever, expander, use_boost=False, factor=1.0)
        print(f"  {'baseline':<14}{br:>10.4f}{bz:>8}{bm:>10.4f}")

        # 各 factor
        for f in FACTORS:
            if f == 1.0:
                continue
            r, m, z = _eval(queries, qrels, retriever, expander, use_boost=True, factor=f)
            print(f"  {f'boost x{f}':<14}{r:>10.4f}{z:>8}{m:>10.4f}")


if __name__ == "__main__":
    main()
