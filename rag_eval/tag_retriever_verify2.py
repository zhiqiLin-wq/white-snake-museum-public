"""综合验证：unretrievable 候选池捞回 + 整体 core recall + 副作用。"""
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
from server.rag.tag_retriever import TagRetriever, _minmax, DENSE_WEIGHT, SPARSE_WEIGHT, TAG_WEIGHT

RESULTS = PROJECT_ROOT / "rag_eval" / "results"
GT = PROJECT_ROOT / "rag_eval" / "ground_truth"


def main():
    from rag_eval.detail_log import build_components
    print("[1] build_components ...", flush=True)
    embedder, vector_store, _, _ = build_components()
    print("[1] done", flush=True)

    corpus = load_corpus(GT / "corpus.jsonl")
    meta_lookup = {cid: doc.get("metadata", {}) for cid, doc in corpus.items()}

    print("[2] TagStore ...", flush=True)
    tag_store = TagStore(embedder=embedder)
    tag_store.load()
    tag_store.build_tag_vectors()
    tag_store.build_summary_vectors()
    tag_store.build_plot_unit_vectors()
    print("[2] done", flush=True)

    print("[3] bm25 ...", flush=True)
    retriever = TagRetriever(vector_store, embedder, tag_store)
    retriever.build_tag_bm25(corpus)
    retriever.build_research_bm25(corpus)
    print("[3] done", flush=True)

    qlabels = {
        f"{r['dataset']}/{r['query_id']}": r["query_label"]
        for r in json.load(open(RESULTS / "query_labels.json", encoding="utf-8"))
    }
    qtext_by_ds = {}
    for ds in ["academic_deepseek", "academic_qwen", "daily_deepseek", "daily_qwen"]:
        qs = load_queries(GT / f"queries_{ds}.jsonl")
        qtext_by_ds[ds] = {q["query_id"]: q["text"] for q in qs}

    # 分组测：unretrievable 候选池捞回 + sortable boost 后 recall
    for pick_file, core_key, mode in [
        ("picked_unretrievable_queries.json", "unretrievable_chunks", "unretrievable"),
        ("picked_sortable_queries.json", "sortable_chunks", "sortable"),
    ]:
        picked = json.load(open(RESULTS / pick_file, encoding="utf-8"))
        pool_hit = 0
        total_target = 0
        core_recall = []
        side_lost = 0
        side_total = 0

        for item in picked:
            ds, qid = item["dataset"], item["query_id"]
            qtext = qtext_by_ds[ds].get(qid, "")
            qlabel = qlabels.get(f"{ds}/{qid}", {})
            if not isinstance(qlabel, dict) or "_error" in qlabel:
                continue
            rels = load_qrels(GT / f"qrels_{ds}_src_only.tsv").get(qid, {})
            core = {c for c, r in rels.items() if r == 2}
            rel1 = {c for c, r in rels.items() if r == 1}
            rel_all = core | rel1
            target = set(item.get(core_key, []))
            total_target += len(target)

            # 候选池（五路召回）
            cand = retriever._candidate_scores(qtext, qlabel, meta_lookup)
            pool_hit += len(target & set(cand.keys()))

            # boost 后 top-10 + 副作用
            top10 = asyncio.run(retriever.retrieve(qtext, qlabel, top_k=10,
                                                   corpus=corpus, meta_lookup=meta_lookup))
            if core:
                core_recall.append(len(set(top10) & core) / len(core))

            # 副作用：boost 前 fused_base top-10 里 rel>=1 被挤出
            dn = _minmax({cid: c["dense_score"] for cid, c in cand.items() if c["dense_score"] is not None})
            sn = _minmax({cid: c["bm25_score"] for cid, c in cand.items() if c["bm25_score"] is not None})
            tn = _minmax({cid: c["tag_score"] for cid, c in cand.items() if c["tag_score"] is not None})
            fused = {cid: DENSE_WEIGHT * dn.get(cid, 0.0) + SPARSE_WEIGHT * sn.get(cid, 0.0)
                     + TAG_WEIGHT * tn.get(cid, 0.0)
                     for cid in cand}
            before10 = {cid for cid, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:10]}
            rel_before = before10 & rel_all
            rel_after = set(top10) & rel_all
            side_total += len(rel_before)
            side_lost += len(rel_before - rel_after)

        print(f"\n[{mode}] 核心目标 {total_target}, 候选池捞回 {pool_hit} ({pool_hit/total_target:.3f}), "
              f"core recall {sum(core_recall)/len(core_recall):.4f} "
              f"({len(core_recall)} queries), 副作用 {side_lost}/{side_total}")


if __name__ == "__main__":
    main()
