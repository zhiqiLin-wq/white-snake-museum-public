"""参数扫描：融合权重（dense/sparse/tag）× plot_unit 语义阈值。"""
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


def load_group(pick_file, qtext_by_ds, qlabels):
    """加载一组 query 的 (qtext, qlabel, core) 列表。"""
    picked = json.load(open(RESULTS / pick_file, encoding="utf-8"))
    items = []
    for item in picked:
        ds, qid = item["dataset"], item["query_id"]
        qlabel = qlabels.get(f"{ds}/{qid}", {})
        if not isinstance(qlabel, dict) or "_error" in qlabel:
            continue
        core = {c for c, r in load_qrels(GT / f"qrels_{ds}_src_only.tsv").get(qid, {}).items() if r == 2}
        if not core:
            continue
        items.append((qtext_by_ds[ds].get(qid, ""), qlabel, core))
    return items


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

    qlabels = {f"{r['dataset']}/{r['query_id']}": r["query_label"]
               for r in json.load(open(RESULTS / "query_labels.json", encoding="utf-8"))}
    qtext_by_ds = {}
    for ds in ["academic_deepseek", "academic_qwen", "daily_deepseek", "daily_qwen"]:
        qs = load_queries(GT / f"queries_{ds}.jsonl")
        qtext_by_ds[ds] = {q["query_id"]: q["text"] for q in qs}

    sortable_items = load_group("picked_sortable_queries.json", qtext_by_ds, qlabels)
    unretr_items = load_group("picked_unretrievable_queries.json", qtext_by_ds, qlabels)

    def eval_group(items, dw, sw, tw, th):
        rec = []
        for qtext, qlabel, core in items:
            top10 = asyncio.run(retriever.retrieve(
                qtext, qlabel, top_k=10, corpus=corpus, meta_lookup=meta_lookup,
                fusion_weights=(dw, sw, tw), plot_unit_threshold=th,
            ))
            rec.append(len(set(top10) & core) / len(core))
        return sum(rec) / len(rec) if rec else 0.0

    fusion_grid = [
        (0.4, 0.4, 0.2, "dense.4 sparse.4 tag.2"),
        (0.45, 0.45, 0.1, "dense.45 sparse.45 tag.1"),
        (0.35, 0.35, 0.3, "dense.35 sparse.35 tag.3"),
    ]
    threshold_grid = [0.65, 0.75, 0.85]

    print("\n" + "=" * 80)
    print("参数扫描（sortable / unretrievable / 加权平均 core recall）")
    print("=" * 80)
    print(f"{'融合权重':<26}{'阈值':<7}{'sortable':<11}{'unretr':<11}{'加权平均':<11}")

    results = []
    for dw, sw, tw, label in fusion_grid:
        for th in threshold_grid:
            s_avg = eval_group(sortable_items, dw, sw, tw, th)
            u_avg = eval_group(unretr_items, dw, sw, tw, th)
            n_s = len(sortable_items)
            n_u = len(unretr_items)
            weighted = (s_avg * n_s + u_avg * n_u) / (n_s + n_u)
            print(f"{label:<26}{th:<7.2f}{s_avg:<11.4f}{u_avg:<11.4f}{weighted:<11.4f}", flush=True)
            results.append((label, th, s_avg, u_avg, weighted))

    best = max(results, key=lambda r: r[4])
    print("\n最优组合（按加权平均）:", best)


if __name__ == "__main__":
    main()
