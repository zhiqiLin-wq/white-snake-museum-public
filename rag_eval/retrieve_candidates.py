"""阶段 2：检索候选集。

对抽中的 10 个 sortable query，跑 dense top-50 + sparse top-50（带生产 query 预处理：
别名扩展 + research_penalty + version_boost）。候选集 = 标准答案(src_only rel=2) ∪
dense50 ∪ sparse50，按 chunk_id 去重。记录每个候选的 dense_score / bm25_score。
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
for p in [str(PROJECT_ROOT), str(AGENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rag_eval.detail_log import build_components, preprocess_query
from rag_eval.loader import load_corpus, load_queries, load_qrels

RESULTS = PROJECT_ROOT / "rag_eval" / "results"
GT = PROJECT_ROOT / "rag_eval" / "ground_truth"

D_TOP_K = 50
S_TOP_K = 50
CHUNK_TRUNC = 900


def main():
    picked = json.load(open(RESULTS / "picked_sortable_queries.json", encoding="utf-8"))
    print("[INIT] loading embedder / vector_store / BM25 ...", flush=True)
    embedder, vector_store, bm25, expander = build_components()

    corpus = load_corpus(GT / "corpus.jsonl")
    meta_lookup = {cid: doc.get("metadata", {}) for cid, doc in corpus.items()}

    ds_list = sorted({x["dataset"] for x in picked})
    qtext_by_ds, qrels_by_ds = {}, {}
    for ds in ds_list:
        qs = load_queries(GT / f"queries_{ds}.jsonl")
        qtext_by_ds[ds] = {q["query_id"]: q["text"] for q in qs}
        qrels_by_ds[ds] = load_qrels(GT / f"qrels_{ds}_src_only.tsv")

    out = []
    for item in picked:
        ds, qid = item["dataset"], item["query_id"]
        qtext = qtext_by_ds[ds].get(qid, "")
        rels = qrels_by_ds[ds].get(qid, {})
        core = {c for c, r in rels.items() if r == 2}

        pp = preprocess_query(qtext, expander)

        qvec = embedder.embed_query(pp["dense_query"])
        dense_raw = vector_store.query(qvec, top_k=D_TOP_K)
        sparse_raw = bm25.search(pp["sparse_query"], top_k=S_TOP_K, query_weights=pp["sparse_weights"])

        candidates = {}
        for cid in core:
            candidates[cid] = {"chunk_id": cid, "dense_score": None, "bm25_score": None}
        for r in dense_raw:
            cid = r.get("chunk_id", "")
            if not cid:
                continue
            candidates.setdefault(cid, {"chunk_id": cid, "dense_score": None, "bm25_score": None})
            candidates[cid]["dense_score"] = r.get("score")
        for r in sparse_raw:
            cid = r.get("chunk_id", "")
            if not cid:
                continue
            candidates.setdefault(cid, {"chunk_id": cid, "dense_score": None, "bm25_score": None})
            candidates[cid]["bm25_score"] = r.get("bm25_score")

        cand_list = []
        for cid, info in candidates.items():
            meta = meta_lookup.get(cid, {})
            cand_list.append({
                "chunk_id": cid,
                "genre": meta.get("genre", ""),
                "source_type": meta.get("source_type", ""),
                "dense_score": info["dense_score"],
                "bm25_score": info["bm25_score"],
                "text": corpus.get(cid, {}).get("text", "")[:CHUNK_TRUNC],
                "is_core": cid in core,
            })

        out.append({
            "dataset": ds,
            "query_id": qid,
            "query_text": qtext,
            "category": item.get("category", ""),
            "research_penalty": pp["research_penalty"],
            "version_boost": pp["version_boost"],
            "core": sorted(core),
            "sortable_chunks": sorted(item.get("sortable_chunks", [])),
            "candidates": cand_list,
        })
        print(f"  [done] {ds}/{qid} core={len(core)} candidates={len(cand_list)}", flush=True)

    out_path = RESULTS / "retrieved_candidates.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OUT] {out_path} ({len(out)} queries)", flush=True)


if __name__ == "__main__":
    main()
