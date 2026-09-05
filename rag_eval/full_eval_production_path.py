"""全量评估：走生产代码路径 tag_retriever.retrieve（embed_query + aug_matrix + boost）。

与 full_eval_5path 口径一致，但逐 query 调用生产 TagRetriever.retrieve（而非评估内联计算），
证明生产接线能复现 core recall@10 = 0.6518。

Usage:
    python rag_eval/full_eval_production_path.py
"""
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
for p in [str(PROJECT_ROOT), str(AGENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rag_eval.loader import load_queries, load_qrels
from server.rag.tag_store import TagStore, load_corpus
from server.rag.tag_retriever import TagRetriever

RESULTS = PROJECT_ROOT / "rag_eval" / "results"
GT = PROJECT_ROOT / "rag_eval" / "ground_truth"
DATASETS = ["academic_deepseek", "academic_qwen", "daily_deepseek", "daily_qwen"]


def dcg(gains):
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains))


def compute_metrics(ranked, rel_all, core):
    m = {}
    for k in [5, 10, 20]:
        top = set(ranked[:k])
        m[f"recall@{k}"] = len(top & rel_all) / len(rel_all) if rel_all else 0.0
    m["core_recall@10"] = len(set(ranked[:10]) & core) / len(core) if core else 0.0
    mrr = 0.0
    for i, cid in enumerate(ranked[:10], start=1):
        if cid in rel_all:
            mrr = 1.0 / i
            break
    m["mrr@10"] = mrr
    m["hit@1"] = 1.0 if ranked[:1] and ranked[0] in rel_all else 0.0
    m["hit@10"] = 1.0 if set(ranked[:10]) & rel_all else 0.0
    m["precision@10"] = len(set(ranked[:10]) & rel_all) / 10.0
    gains = [(2 ** 2 - 1) if cid in core else ((2 ** 1 - 1) if cid in rel_all else 0)
             for cid in ranked[:10]]
    ideal = sorted([3] * len(core) + [1] * max(0, len(rel_all) - len(core)) + [0] * 10, reverse=True)[:10]
    d, id_ = dcg(gains), dcg(ideal)
    m["ndcg@10"] = d / id_ if id_ > 0 else 0.0
    return m


METRIC_KEYS = ["recall@5", "recall@10", "recall@20", "core_recall@10",
               "mrr@10", "hit@1", "hit@10", "precision@10", "ndcg@10"]


def main():
    from rag_eval.detail_log import build_components
    print("[1] build ...", flush=True)
    embedder, vector_store, _, _ = build_components()
    tag_store = TagStore(embedder=embedder)
    tag_store.load()
    tag_store.build_tag_vectors()
    tag_store.build_summary_vectors()
    tag_store.build_plot_unit_vectors()
    retriever = TagRetriever(vector_store, embedder, tag_store)
    corpus = load_corpus()
    meta_lookup = {cid: doc.get("metadata", {}) for cid, doc in corpus.items()}
    retriever.build_tag_bm25(corpus)
    retriever.build_tag_aug_dense(corpus)
    print("[1] done", flush=True)

    qlabels_full = {f"{r['dataset']}/{r['query_id']}": r["query_label"]
                    for r in json.load(open(RESULTS / "query_labels_full.json", encoding="utf-8"))}

    items = []
    for ds in DATASETS:
        qs = load_queries(GT / f"queries_{ds}.jsonl")
        qrels = load_qrels(GT / f"qrels_{ds}_src_only.tsv")
        for q in qs:
            qid = q["query_id"]
            rels = qrels.get(qid, {})
            core = {c for c, r in rels.items() if r == 2}
            rel_all = {c for c, r in rels.items() if r >= 1}
            if not core:
                continue
            qlabel = qlabels_full.get(f"{ds}/{qid}", {})
            if not isinstance(qlabel, dict) or "_error" in qlabel:
                continue
            items.append((f"{ds}/{qid}", q["text"], qlabel, core, rel_all))
    print(f"[2] {len(items)} queries", flush=True)

    by_ds = defaultdict(list)
    for idx, (key, qtext, qlabel, core, rel_all) in enumerate(items):
        ranked = asyncio.run(retriever.retrieve(
            qtext, qlabel, top_k=20, corpus=corpus, meta_lookup=meta_lookup))
        m = compute_metrics(ranked, rel_all, core)
        by_ds[key.split("/")[0]].append(m)
        if (idx + 1) % 200 == 0:
            print(f"  [progress] {idx+1}/{len(items)}", flush=True)

    print("\n" + "=" * 90)
    print(f"生产代码路径 tag_retriever.retrieve 全量评估（{len(items)} queries）— macro average")
    print("=" * 90)
    header = f"{'指标':<16}" + "".join(f"{ds:<16}" for ds in DATASETS) + f"{'整体':<16}"
    print(header)
    print("-" * 90)
    for mk in METRIC_KEYS:
        row = f"{mk:<16}"
        all_vals = []
        for ds in DATASETS:
            vals = [m[mk] for m in by_ds[ds]]
            avg = sum(vals) / len(vals) if vals else 0.0
            all_vals.extend(vals)
            row += f"{avg:<16.4f}"
        row += f"{sum(all_vals)/len(all_vals):<16.4f}"
        print(row)


if __name__ == "__main__":
    main()
