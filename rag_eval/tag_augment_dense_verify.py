"""验证：标签增强 dense（标签词拼开头）能否提升 core recall。"""
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

from rag_eval.loader import load_corpus, load_queries, load_qrels
from server.rag.tag_store import TagStore, _load_vecs, _save_vecs
from server.rag.tag_retriever import TagRetriever, build_query_tags, build_query_tags_disc, _minmax

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
        core = {c for c, r in load_qrels(GT / f"qrels_{ds}_src_only.tsv").get(qid, {}).items() if r == 2}
        if not core:
            continue
        items.append((qtext_by_ds[ds].get(qid, ""), qlabel, core))
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

    # 构建标签增强 dense 向量（标签词拼开头，带缓存）
    print("[2] build tag-augmented dense ...", flush=True)
    aug_cids = list(corpus.keys())
    cached = _load_vecs("tag_aug_dense")
    if cached is not None:
        aug_matrix = np.asarray([cached[cid] for cid in aug_cids], dtype=float)
    else:
        aug_texts = [tag_store.build_tag_augmented_text(cid, corpus[cid]["text"]) for cid in aug_cids]
        all_vecs = []
        for i in range(0, len(aug_texts), 64):
            all_vecs.extend(embedder.embed(aug_texts[i:i + 64]))
        aug_matrix = np.asarray(all_vecs, dtype=float)
        _save_vecs("tag_aug_dense", {cid: v for cid, v in zip(aug_cids, aug_matrix.tolist())})
    print("[2] done", flush=True)

    qlabels = {f"{r['dataset']}/{r['query_id']}": r["query_label"]
               for r in json.load(open(RESULTS / "query_labels.json", encoding="utf-8"))}
    qtext_by_ds = {}
    for ds in ["academic_deepseek", "academic_qwen", "daily_deepseek", "daily_qwen"]:
        qs = load_queries(GT / f"queries_{ds}.jsonl")
        qtext_by_ds[ds] = {q["query_id"]: q["text"] for q in qs}

    groups = [
        ("picked_sortable_queries.json", "sortable"),
        ("picked_unretrievable_queries.json", "unretr"),
    ]

    for pick_file, label in groups:
        items = load_group(pick_file, qtext_by_ds, qlabels)
        rec = []
        for qtext, qlabel, core in items:
            # 用标签增强 dense 替代原始 dense，其余路复用 TagRetriever
            qvec = embedder.embed_query(qtext)
            qvec = np.asarray(qvec, dtype=float)
            scores = aug_matrix @ qvec
            order = np.argsort(scores)[::-1][:100]
            dense_scores = {aug_cids[i]: float(scores[i]) for i in order}

            # 其余路（sparse 标签增强 + 标签语义 + plot_unit + 研究文献）
            qtags = build_query_tags(qlabel)
            bm25_scores = {}
            if retriever._tag_bm25 is not None:
                aug_query = (qtext + " " + " ".join(qtags)).strip()
                for r in retriever._tag_bm25.search(aug_query, top_k=100):
                    cid = r.get("chunk_id")
                    if cid:
                        bm25_scores[cid] = r.get("bm25_score", 0.0)
            tag_scores = {}
            qtags_disc = build_query_tags_disc(qlabel, tag_store.idf)
            if tag_store.tag_vector:
                q_tag_vec = np.asarray(embedder.embed([" ".join(qtags_disc)])[0], dtype=float)
                cids = tag_store.chunk_ids
                mat = np.asarray([tag_store.tag_vector(c) for c in cids], dtype=float)
                ts = mat @ q_tag_vec
                for cid, s in zip(cids, ts):
                    tag_scores[cid] = float(s)

            candidates = set(dense_scores.keys()) | set(bm25_scores.keys()) | set(tag_scores.keys())
            dn = _minmax(dense_scores)
            sn = _minmax(bm25_scores)
            tn = _minmax(tag_scores)
            fused = {cid: 0.4*dn.get(cid,0)+0.4*sn.get(cid,0)+0.2*tn.get(cid,0) for cid in candidates}

            # boost 重排（复用 compute_boost）
            from server.rag.boost import compute_boost, summary_jaccard
            q_genre = qlabel.get("genre", "") or ""
            q_summary = qlabel.get("summary", "") or ""
            q_summary_emb = None
            if q_summary:
                q_summary_emb = np.asarray(embedder.embed([q_summary])[0], dtype=float)
            final = {}
            for cid in candidates:
                c_genre = meta_lookup.get(cid, {}).get("genre", "")
                nl = tag_store.norm_label(cid) or {}
                sd = 0.0
                sj = 0.0
                if q_summary:
                    sv = tag_store.summary_vector(cid)
                    if sv is not None and q_summary_emb is not None:
                        se = np.asarray(sv, dtype=float)
                        sd = float(q_summary_emb @ se / (np.linalg.norm(q_summary_emb) * np.linalg.norm(se) + 1e-9))
                    st = tag_store.summary_text(cid)
                    if st:
                        sj = summary_jaccard(q_summary, st)
                boost, hard = compute_boost(qlabel, nl, c_genre, sd, sj, tag_store.idf,
                                            tag_store.plot_unit_sim, 0.75)
                gf = 1.2 if (c_genre == q_genre and q_genre) else 1.0
                final[cid] = (fused[cid] + boost) * gf
                if hard:
                    final[cid] += 10.0
            top10 = [cid for cid, _ in sorted(final.items(), key=lambda x: x[1], reverse=True)[:10]]
            rec.append(len(set(top10) & core) / len(core))

        print(f"[{label}] core recall = {sum(rec)/len(rec):.4f} ({len(rec)} queries)", flush=True)


if __name__ == "__main__":
    main()
