"""验证：生产 TagRetriever 代码路径（aug_matrix + boost + return_scores）与评估口径一致。

对同一批 query，分别用
  A) 生产代码路径 tag_retriever.retrieve(..., return_scores=True)
  B) 评估内联口径（full_eval_5path 的 dense/bm25/tag 融合 + boost）
计算 top-10，断言两者一致，证明生产接线复现 0.6518。

Usage:
    python rag_eval/verify_tag_pipeline_path.py --limit 20
"""
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
for p in [str(PROJECT_ROOT), str(AGENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rag_eval.loader import load_queries, load_qrels
from server.rag.tag_store import TagStore, load_corpus
from server.rag.tag_retriever import build_query_tags, build_query_tags_disc, _minmax
from server.rag.boost import compute_boost, summary_jaccard, normalize_qtype, PLOT_UNIT_SIM_THRESHOLD

RESULTS = PROJECT_ROOT / "rag_eval" / "results"
GT = PROJECT_ROOT / "rag_eval" / "ground_truth"
DATASETS = ["academic_deepseek", "academic_qwen", "daily_deepseek", "daily_qwen"]
BGE_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
TOP_K = 100


def inline_top10(embedder, tag_store, retriever, aug_matrix, aug_cids,
                 qtext, qlabel, meta_lookup, q_dense_vec, q_tag_vec, q_summary_vec):
    """评估内联口径：标签增强 dense + sparse + tag + boost，返回 top-10 cid 列表。"""
    qtype = normalize_qtype(qlabel.get("query_type", ""))
    qd = np.asarray(q_dense_vec, dtype=float)
    scores = aug_matrix @ qd
    order = np.argsort(scores)[::-1][:TOP_K]
    dense_scores = {aug_cids[i]: float(scores[i]) for i in order}

    qtags = build_query_tags(qlabel)
    bm25_scores = {}
    aug_query = (qtext + " " + " ".join(qtags)).strip()
    for r in retriever._tag_bm25.search(aug_query, top_k=TOP_K):
        cid = r.get("chunk_id")
        if cid:
            bm25_scores[cid] = r.get("bm25_score", 0.0)

    tag_scores = {}
    qt = np.asarray(q_tag_vec, dtype=float)
    cids = tag_store.chunk_ids
    mat = np.asarray([tag_store.tag_vector(c) for c in cids], dtype=float)
    ts = mat @ qt
    for cid, s in zip(cids, ts):
        tag_scores[cid] = float(s)

    candidates = set(dense_scores) | set(bm25_scores) | set(tag_scores)
    dn = _minmax(dense_scores)
    sn = _minmax(bm25_scores)
    tn = _minmax(tag_scores)
    fused = {cid: 0.4 * dn.get(cid, 0) + 0.4 * sn.get(cid, 0) + 0.2 * tn.get(cid, 0)
             for cid in candidates}

    q_genre = qlabel.get("genre", "") or ""
    q_summary = qlabel.get("summary", "") or ""
    q_summary_emb = np.asarray(q_summary_vec, dtype=float) if q_summary else None
    final = {}
    for cid in candidates:
        c_genre = meta_lookup.get(cid, {}).get("genre", "")
        nl = tag_store.norm_label(cid) or {}
        sd = sj = 0.0
        if q_summary:
            sv = tag_store.summary_vector(cid)
            if sv is not None and q_summary_emb is not None:
                se = np.asarray(sv, dtype=float)
                sd = float(q_summary_emb @ se / (np.linalg.norm(q_summary_emb) * np.linalg.norm(se) + 1e-9))
            st = tag_store.summary_text(cid)
            if st:
                sj = summary_jaccard(q_summary, st)
        boost, hard = compute_boost(qlabel, nl, c_genre, sd, sj, tag_store.idf,
                                    tag_store.plot_unit_sim, PLOT_UNIT_SIM_THRESHOLD)
        gf = 1.2 if (c_genre == q_genre and q_genre) else 1.0
        final[cid] = (fused[cid] + boost) * gf
        if hard:
            final[cid] += 10.0
    return [cid for cid, _ in sorted(final.items(), key=lambda x: x[1], reverse=True)[:10]]


def main(limit=20):
    from rag_eval.detail_log import build_components
    print("[1] build ...", flush=True)
    embedder, vector_store, _, _ = build_components()

    tag_store = TagStore(embedder=embedder)
    tag_store.load()
    tag_store.build_tag_vectors()
    tag_store.build_summary_vectors()
    tag_store.build_plot_unit_vectors()

    from server.rag.tag_retriever import TagRetriever
    retriever = TagRetriever(vector_store, embedder, tag_store)
    corpus = load_corpus()
    meta_lookup = {cid: doc.get("metadata", {}) for cid, doc in corpus.items()}
    retriever.build_tag_bm25(corpus)
    retriever.build_tag_aug_dense(corpus)
    aug_cids = retriever._aug_cids
    aug_matrix = retriever._aug_matrix
    print(f"[1] done: aug_matrix {aug_matrix.shape}, {len(corpus)} chunks", flush=True)

    qlabels_full = {f"{r['dataset']}/{r['query_id']}": r["query_label"]
                    for r in json.load(open(RESULTS / "query_labels_full.json", encoding="utf-8"))}
    qtext_by_ds = {}
    for ds in DATASETS:
        qtext_by_ds[ds] = {q["query_id"]: q["text"] for q in load_queries(GT / f"queries_{ds}.jsonl")}

    items = []
    for ds in DATASETS:
        for q in load_queries(GT / f"queries_{ds}.jsonl"):
            qid = q["query_id"]
            qlabel = qlabels_full.get(f"{ds}/{qid}", {})
            if not isinstance(qlabel, dict) or "_error" in qlabel:
                continue
            items.append((f"{ds}/{qid}", qtext_by_ds[ds][qid], qlabel))
            if len(items) >= limit:
                break
        if len(items) >= limit:
            break

    mismatch = 0
    for key, qtext, qlabel in items:
        q_dense = embedder.embed([BGE_INSTRUCTION + qtext])[0]
        q_tag = embedder.embed([" ".join(build_query_tags_disc(qlabel, tag_store.idf))])[0]
        q_summary = embedder.embed([qlabel.get("summary", "") or ""])[0]

        eval_top = inline_top10(embedder, tag_store, retriever, aug_matrix, aug_cids,
                                qtext, qlabel, meta_lookup, q_dense, q_tag, q_summary)

        prod_ranked = asyncio.run(retriever.retrieve(
            qtext, qlabel, top_k=10, corpus=corpus, meta_lookup=meta_lookup, return_scores=True))
        prod_top = [cid for cid, _ in prod_ranked]

        if eval_top != prod_top:
            mismatch += 1
            print(f"\n[MISMATCH] {key}")
            print(f"  eval: {eval_top}")
            print(f"  prod: {prod_top}")
        else:
            print(f"[OK] {key}  top-10 一致")

    print(f"\n结论: {len(items) - mismatch}/{len(items)} 条一致" + ("（生产路径 == 评估口径）" if mismatch == 0 else "，存在不一致！"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    main(limit=args.limit)
