"""验证生产 TagRetriever（五路召回 + boost）的效果，对比候选池阈值。"""
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
for p in [str(PROJECT_ROOT), str(AGENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rag_eval.loader import load_corpus, load_queries, load_qrels
from server.rag.tag_store import TagStore
from server.rag.tag_retriever import TagRetriever

RESULTS = PROJECT_ROOT / "rag_eval" / "results"
GT = PROJECT_ROOT / "rag_eval" / "ground_truth"


def main():
    from rag_eval.detail_log import build_components
    print("[1] build_components ...", flush=True)
    embedder, vector_store, _, _ = build_components()
    print("[1] done", flush=True)

    corpus = load_corpus(GT / "corpus.jsonl")
    meta_lookup = {cid: doc.get("metadata", {}) for cid, doc in corpus.items()}

    print("[2] TagStore.load ...", flush=True)
    tag_store = TagStore(embedder=embedder)
    tag_store.load()
    print(f"[2] done, {len(tag_store.chunk_ids)} chunks", flush=True)

    print("[3] build_tag_vectors ...", flush=True)
    tag_store.build_tag_vectors()
    print("[3] done", flush=True)
    print("[3.5] build_summary_vectors ...", flush=True)
    tag_store.build_summary_vectors()
    print("[3.5] done", flush=True)

    print("[4] build bm25 ...", flush=True)
    retriever = TagRetriever(vector_store, embedder, tag_store)
    retriever.build_tag_bm25(corpus)
    retriever.build_research_bm25(corpus)
    print("[4] done", flush=True)

    picked = json.load(open(RESULTS / "picked_sortable_queries.json", encoding="utf-8"))
    query_labels = {
        f"{r['dataset']}/{r['query_id']}": r["query_label"]
        for r in json.load(open(RESULTS / "query_labels.json", encoding="utf-8"))
    }
    qtext_by_ds = {}
    for ds in sorted({x["dataset"] for x in picked}):
        qs = load_queries(GT / f"queries_{ds}.jsonl")
        qtext_by_ds[ds] = {q["query_id"]: q["text"] for q in qs}

    for max_cand in [None, 250]:
        core_recall = []
        salvaged = 0
        total = 0
        for item in picked:
            ds, qid = item["dataset"], item["query_id"]
            qtext = qtext_by_ds[ds].get(qid, "")
            qlabel = query_labels.get(f"{ds}/{qid}", {})
            if not isinstance(qlabel, dict) or "_error" in qlabel:
                continue
            rels = load_qrels(GT / f"qrels_{ds}_src_only.tsv").get(qid, {})
            core = {c for c, r in rels.items() if r == 2}
            sortable_core = set(item.get("sortable_chunks", []))
            total += len(sortable_core)

            top10 = asyncio.run(retriever.retrieve(qtext, qlabel, top_k=10,
                                                   corpus=corpus, meta_lookup=meta_lookup,
                                                   max_candidates=max_cand))
            if core:
                core_recall.append(len(set(top10) & core) / len(core))
            salvaged += len(sortable_core & set(top10))

        print(f"\n[max_candidates={max_cand}] 捞回 {salvaged}/{total} = {salvaged/total:.3f}, "
              f"core recall {sum(core_recall)/len(core_recall):.4f}")


if __name__ == "__main__":
    main()
