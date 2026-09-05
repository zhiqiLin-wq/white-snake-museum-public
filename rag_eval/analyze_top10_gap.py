"""深入分析：核心进了候选池，但为什么进不了前 10。"""
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
for p in [str(PROJECT_ROOT), str(AGENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from rag_eval.loader import load_corpus, load_queries, load_qrels
from server.rag.tag_store import TagStore
from server.rag.tag_retriever import TagRetriever, _minmax

RESULTS = PROJECT_ROOT / "rag_eval" / "results"
GT = PROJECT_ROOT / "rag_eval" / "ground_truth"


def main():
    from rag_eval.detail_log import build_components
    embedder, vector_store, _, _ = build_components()
    corpus = load_corpus(GT / "corpus.jsonl")
    meta_lookup = {cid: doc.get("metadata", {}) for cid, doc in corpus.items()}

    tag_store = TagStore(embedder=embedder)
    tag_store.load()
    tag_store.build_tag_vectors()
    tag_store.build_summary_vectors()
    tag_store.build_plot_unit_vectors()

    retriever = TagRetriever(vector_store, embedder, tag_store)
    retriever.build_tag_bm25(corpus)
    retriever.build_research_bm25(corpus)

    picked = json.load(open(RESULTS / "picked_sortable_queries.json", encoding="utf-8"))
    qlabels = {f"{r['dataset']}/{r['query_id']}": r["query_label"]
               for r in json.load(open(RESULTS / "query_labels.json", encoding="utf-8"))}
    qtext_by_ds = {}
    for ds in ["academic_deepseek", "academic_qwen", "daily_deepseek", "daily_qwen"]:
        qs = load_queries(GT / f"queries_{ds}.jsonl")
        qtext_by_ds[ds] = {q["query_id"]: q["text"] for q in qs}

    # 统计：核心进候选池但没进前10，它们的 fused_base 排名分布 + 硬分组情况
    gap_rank = Counter()
    hard_ratio = 0
    hard_total = 0
    gap_total = 0

    for item in picked:
        ds, qid = item["dataset"], item["query_id"]
        qtext = qtext_by_ds[ds].get(qid, "")
        qlabel = qlabels.get(f"{ds}/{qid}", {})
        if not isinstance(qlabel, dict) or "_error" in qlabel:
            continue
        core = {c for c, r in load_qrels(GT / f"qrels_{ds}_src_only.tsv").get(qid, {}).items() if r == 2}
        sortable = set(item.get("sortable_chunks", []))

        cand = retriever._candidate_scores(qtext, qlabel, meta_lookup)
        dn = _minmax({cid: c["dense_score"] for cid, c in cand.items() if c["dense_score"] is not None})
        sn = _minmax({cid: c["bm25_score"] for cid, c in cand.items() if c["bm25_score"] is not None})
        tn = _minmax({cid: c["tag_score"] for cid, c in cand.items() if c["tag_score"] is not None})
        fused = {cid: 0.4*dn.get(cid,0)+0.4*sn.get(cid,0)+0.2*tn.get(cid,0) for cid in cand}
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        top10 = asyncio.run(retriever.retrieve(qtext, qlabel, top_k=10,
                                               corpus=corpus, meta_lookup=meta_lookup))

        # 每个 sortable 核心：进候选池了吗？进前10了吗？fused 排名？
        for cid in sortable:
            if cid not in cand:
                continue
            if cid in top10:
                continue
            # 进候选池但没进前10
            gap_total += 1
            rank_in_cand = next((i+1 for i, (c, _) in enumerate(ranked) if c == cid), 999)
            gap_rank[min(rank_in_cand, 30)] += 1

    print(f"进了候选池但没进前10的核心: {gap_total}")
    print("它们的 fused_base 排名分布（候选池内）:")
    for r in sorted(gap_rank):
        print(f"  rank {r}: {gap_rank[r]}")


if __name__ == "__main__":
    main()
