"""扫描 boost 权重（persons/locations base），看 core recall 和副作用。"""
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
from server.rag.tag_retriever import TagRetriever, _minmax
import server.rag.boost as boost_mod

RESULTS = PROJECT_ROOT / "rag_eval" / "results"
GT = PROJECT_ROOT / "rag_eval" / "ground_truth"


def load_group(pick_file, qtext_by_ds, qlabels):
    picked = json.load(open(RESULTS / pick_file, encoding="utf-8"))
    items = []
    for item in picked:
        ds, qid = item["dataset"], item["query_id"]
        qlabel = qlabels.get(f"{ds}/{qid}", {})
        if not isinstance(qlabel, dict) or "_error" in qlabel:
            continue
        rels = load_qrels(GT / f"qrels_{ds}_src_only.tsv").get(qid, {})
        core = {c for c, r in rels.items() if r == 2}
        rel_all = {c for c, r in rels.items() if r >= 1}
        if not core:
            continue
        items.append((qtext_by_ds[ds].get(qid, ""), qlabel, core, rel_all))
    return items


def main():
    from rag_eval.detail_log import build_components
    print("[1] build ...", flush=True)
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
    print("[1] done", flush=True)

    qlabels = {f"{r['dataset']}/{r['query_id']}": r["query_label"]
               for r in json.load(open(RESULTS / "query_labels.json", encoding="utf-8"))}
    qtext_by_ds = {}
    for ds in ["academic_deepseek", "academic_qwen", "daily_deepseek", "daily_qwen"]:
        qs = load_queries(GT / f"queries_{ds}.jsonl")
        qtext_by_ds[ds] = {q["query_id"]: q["text"] for q in qs}

    sortable = load_group("picked_sortable_queries.json", qtext_by_ds, qlabels)
    unretr = load_group("picked_unretrievable_queries.json", qtext_by_ds, qlabels)

    def eval_group(items, pw, lw):
        rec = []
        side_lost = 0
        side_total = 0
        for qtext, qlabel, core, rel_all in items:
            cand = retriever._candidate_scores(qtext, qlabel, meta_lookup)
            dn = _minmax({cid: c["dense_score"] for cid, c in cand.items() if c["dense_score"] is not None})
            sn = _minmax({cid: c["bm25_score"] for cid, c in cand.items() if c["bm25_score"] is not None})
            tn = _minmax({cid: c["tag_score"] for cid, c in cand.items() if c["tag_score"] is not None})
            fused = {cid: 0.4 * dn.get(cid, 0.0) + 0.4 * sn.get(cid, 0.0) + 0.2 * tn.get(cid, 0.0)
                     for cid in cand}
            before10 = {cid for cid, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:10]}

            top10 = asyncio.run(retriever.retrieve(
                qtext, qlabel, top_k=10, corpus=corpus, meta_lookup=meta_lookup,
                fusion_weights=(0.4, 0.4, 0.2), plot_unit_threshold=0.75,
            ))
            rec.append(len(set(top10) & core) / len(core))
            rel_before = before10 & rel_all
            rel_after = set(top10) & rel_all
            side_total += len(rel_before)
            side_lost += len(rel_before - rel_after)
        return sum(rec) / len(rec), side_lost, side_total

    grid = [
        (0.10, 0.09, "person.10 loc.09（当前）"),
        (0.15, 0.13, "person.15 loc.13"),
        (0.20, 0.18, "person.20 loc.18"),
        (0.30, 0.25, "person.30 loc.25"),
    ]

    print("\n" + "=" * 85)
    print("boost 权重扫描（sortable / unretr core recall + 副作用）")
    print("=" * 85)
    print(f"{'权重':<24}{'sortable':<11}{'unretr':<11}{'副作用':<14}")

    for pw, lw, label in grid:
        boost_mod.PERSON_WEIGHT = pw
        boost_mod.LOCATION_WEIGHT = lw
        s_rec, s_lost, s_total = eval_group(sortable, pw, lw)
        u_rec, u_lost, u_total = eval_group(unretr, pw, lw)
        side = (s_lost + u_lost) / (s_total + u_total) if (s_total + u_total) else 0
        print(f"{label:<24}{s_rec:<11.4f}{u_rec:<11.4f}{side:<14.4f} "
              f"({s_lost+u_lost}/{s_total+u_total})", flush=True)


if __name__ == "__main__":
    main()
