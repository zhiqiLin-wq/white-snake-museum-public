"""版本软 boost — 全量数据集对照（baseline vs boosted）。

与 version_filter_test.py（硬过滤）互补：本脚本遍历指定数据集的**全部 query**，
对每条 query 跑两组检索对照：

  baseline: 不带 version_boost（现状，全语料检索）
  boosted:  带 detect_version_filters 输出的 version_boost（软调分端到端）

version_boost 只升匹配版本（× version_boost_factor），别版本不屏蔽不降。
按"有版本信号（boost 非 None）/ 无版本信号（boost None）"分组输出，
以看清改进来源（版本 query 上浮）与副作用（误 boost 是否拖累别版本）。

Usage:
    python rag_eval/version_boost_test.py                  # 默认跑全部 4 个数据集
    python rag_eval/version_boost_test.py --datasets academic_deepseek
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


def _load_dataset(ds: str):
    from rag_eval.loader import load_corpus, load_queries, load_qrels

    corpus = load_corpus(GT_DIR / "corpus.jsonl")
    queries = load_queries(GT_DIR / f"queries_{ds}.jsonl")
    qrels = load_qrels(GT_DIR / f"qrels_{ds}.tsv")
    return corpus, queries, qrels


async def _run(queries, qrels, retriever, expander) -> dict:
    from rag_eval.metrics import recall_at_k, mrr
    from server.rag.query_intent import detect_version_filters, research_penalty_for

    # 分组统计: {"all" | "signal" | "no_signal"} -> mode -> {recall[], mrr[], zero}
    agg = defaultdict(lambda: defaultdict(lambda: {"recall": [], "mrr": [], "zero": 0}))

    total = len(queries)
    for idx, q in enumerate(queries):
        qid = q.get("query_id", "")
        text = q.get("text", "")
        rels = qrels.get(qid, {})
        if not rels:
            continue
        rel_set = set(rels.keys())

        boost = detect_version_filters(text)
        group = "signal" if boost else "no_signal"

        expand_result = expander.expand_weighted(text)
        eq_dict = {
            "dense_query": text,
            "sparse_query": expand_result["expanded_query"],
            "sparse_weights": expand_result["token_weights"],
            "research_penalty": research_penalty_for(text),
        }

        for mode, vb in (("baseline", None), ("boosted", boost)):
            results = await retriever.retrieve(eq_dict, top_k=10, version_boost=vb)
            retrieved = [r.get("chunk_id", "") for r in results]
            rec = recall_at_k(retrieved, rel_set, 10)
            agg["all"][mode]["recall"].append(rec)
            agg["all"][mode]["mrr"].append(mrr(retrieved, rel_set))
            if rec == 0:
                agg["all"][mode]["zero"] += 1
            agg[group][mode]["recall"].append(rec)
            agg[group][mode]["mrr"].append(mrr(retrieved, rel_set))
            if rec == 0:
                agg[group][mode]["zero"] += 1

        if (idx + 1) % 100 == 0:
            print(f"  [{idx + 1}/{total}] done")

    return agg


def _mean(x):
    return sum(x) / len(x) if x else 0.0


def _print_group(label, n, agg_group):
    b = agg_group["baseline"]
    d = agg_group["boosted"]
    print(f"{label:<14}{n:>5}"
          f"{_mean(b['recall']):>10.4f}{_mean(d['recall']):>10.4f}"
          f"{(_mean(d['recall']) - _mean(b['recall'])):>+9.4f}"
          f"{b['zero']:>10}{d['zero']:>10}"
          f"{_mean(b['mrr']):>8.4f}{_mean(d['mrr']):>8.4f}")


def _print_dataset(ds, agg):
    n_all = len(agg["all"]["baseline"]["recall"])
    n_sig = len(agg["signal"]["baseline"]["recall"])
    n_no = len(agg["no_signal"]["baseline"]["recall"])
    print(f"\n  {ds}  (有版本信号 {n_sig} 条 / 无信号 {n_no} 条)")
    print(f"  {'分组':<14}{'n':>5}"
          f"{'baseR@10':>10}{'boostR@10':>10}{'Δ':>9}"
          f"{'baseZero':>10}{'boostZero':>10}"
          f"{'baseMRR':>8}{'boostMRR':>8}")
    print(f"  {'-' * 74}")
    _print_group("全部", n_all, agg["all"])
    _print_group("有版本信号", n_sig, agg["signal"])
    _print_group("无版本信号", n_no, agg["no_signal"])


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
        print(f"\n[RUN] {ds} 全量对照...")
        corpus, queries, qrels = _load_dataset(ds)
        print(f"  corpus {len(corpus)} / queries {len(queries)} / qrels {len(qrels)}")
        agg = asyncio.run(_run(queries, qrels, retriever, expander))
        _print_dataset(ds, agg)


if __name__ == "__main__":
    main()
