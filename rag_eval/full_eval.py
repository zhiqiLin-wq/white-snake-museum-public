"""全量评估：多指标 + 按四类 dataset 分组。"""
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

from rag_eval.loader import load_corpus, load_queries, load_qrels
from server.rag.tag_store import TagStore, _load_vecs, _save_vecs
from server.rag.tag_retriever import build_query_tags, build_query_tags_disc, _minmax
from server.rag.boost import compute_boost, summary_jaccard, PLOT_UNIT_SIM_THRESHOLD

RESULTS = PROJECT_ROOT / "rag_eval" / "results"
GT = PROJECT_ROOT / "rag_eval" / "ground_truth"
DATASETS = ["academic_deepseek", "academic_qwen", "daily_deepseek", "daily_qwen"]
BGE_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def batch_embed(embedder, texts, batch_size=64):
    out = []
    for i in range(0, len(texts), batch_size):
        out.extend(embedder.embed(texts[i:i + batch_size]))
    return out


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
    corpus = load_corpus(GT / "corpus.jsonl")
    meta_lookup = {cid: doc.get("metadata", {}) for cid, doc in corpus.items()}

    tag_store = TagStore(embedder=embedder)
    tag_store.load()
    tag_store.build_tag_vectors()
    tag_store.build_summary_vectors()
    tag_store.build_plot_unit_vectors()
    from server.rag.tag_retriever import TagRetriever
    retriever = TagRetriever(vector_store, embedder, tag_store)
    retriever.build_tag_bm25(corpus)
    retriever.build_research_bm25(corpus)
    print("[1] done", flush=True)

    print("[2] tag-aug dense ...", flush=True)
    aug_cids = list(corpus.keys())
    cached = _load_vecs("tag_aug_dense")
    if cached is None:
        print("  缓存缺失，需先跑 tag_augment_dense_verify.py", flush=True)
        sys.exit(1)
    aug_matrix = np.asarray([cached[cid] for cid in aug_cids], dtype=float)
    print("[2] done", flush=True)

    qlabels_full = {f"{r['dataset']}/{r['query_id']}": r["query_label"]
                    for r in json.load(open(RESULTS / "query_labels_full.json", encoding="utf-8"))}

    print("[3] 收集 query ...", flush=True)
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

    # 批量预计算 query 向量（带缓存）
    q_keys = [it[0] for it in items]
    cached_dense = _load_vecs("q_dense_vecs")
    if cached_dense is not None and all(k in cached_dense for k in q_keys):
        q_dense_vecs = [cached_dense[k] for k in q_keys]
        q_tag_vecs = [_load_vecs("q_tag_vecs")[k] for k in q_keys]
        q_summary_vecs = [_load_vecs("q_summary_vecs")[k] for k in q_keys]
    else:
        qtexts = [it[1] for it in items]
        q_dense_vecs = batch_embed(embedder, [BGE_INSTRUCTION + t for t in qtexts])
        q_tag_texts = [" ".join(build_query_tags_disc(it[2], tag_store.idf)) for it in items]
        q_tag_vecs = batch_embed(embedder, q_tag_texts)
        q_summaries = [it[2].get("summary", "") or "" for it in items]
        q_summary_vecs = batch_embed(embedder, q_summaries)
        _save_vecs("q_dense_vecs", {k: v for k, v in zip(q_keys, q_dense_vecs)})
        _save_vecs("q_tag_vecs", {k: v for k, v in zip(q_keys, q_tag_vecs)})
        _save_vecs("q_summary_vecs", {k: v for k, v in zip(q_keys, q_summary_vecs)})
    print(f"[3] done: {len(items)} queries", flush=True)

    by_ds = defaultdict(list)
    for idx, (key, qtext, qlabel, core, rel_all) in enumerate(items):
        qd = np.asarray(q_dense_vecs[idx], dtype=float)
        scores = aug_matrix @ qd
        order = np.argsort(scores)[::-1][:100]
        dense_scores = {aug_cids[i]: float(scores[i]) for i in order}

        qtags = build_query_tags(qlabel)
        bm25_scores = {}
        aug_query = (qtext + " " + " ".join(qtags)).strip()
        for r in retriever._tag_bm25.search(aug_query, top_k=100):
            cid = r.get("chunk_id")
            if cid:
                bm25_scores[cid] = r.get("bm25_score", 0.0)

        tag_scores = {}
        qt = np.asarray(q_tag_vecs[idx], dtype=float)
        cids = tag_store.chunk_ids
        mat = np.asarray([tag_store.tag_vector(c) for c in cids], dtype=float)
        ts = mat @ qt
        for cid, s in zip(cids, ts):
            tag_scores[cid] = float(s)

        candidates = set(dense_scores) | set(bm25_scores) | set(tag_scores)
        dn = _minmax(dense_scores)
        sn = _minmax(bm25_scores)
        tn = _minmax(tag_scores)
        fused = {cid: 0.4*dn.get(cid, 0) + 0.4*sn.get(cid, 0) + 0.2*tn.get(cid, 0) for cid in candidates}

        q_genre = qlabel.get("genre", "") or ""
        q_summary = qlabel.get("summary", "") or ""
        q_summary_emb = np.asarray(q_summary_vecs[idx], dtype=float) if q_summary else None
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
                                        tag_store.plot_unit_sim, PLOT_UNIT_SIM_THRESHOLD)
            gf = 1.2 if (c_genre == q_genre and q_genre) else 1.0
            final[cid] = (fused[cid] + boost) * gf
            if hard:
                final[cid] += 10.0
        ranked = [cid for cid, _ in sorted(final.items(), key=lambda x: x[1], reverse=True)[:20]]
        m = compute_metrics(ranked, rel_all, core)
        by_ds[key.split("/")[0]].append(m)
        if (idx + 1) % 200 == 0:
            print(f"  [progress] {idx+1}/{len(items)}", flush=True)

    # 输出：每个指标 × 每个 dataset 的平均
    print("\n" + "=" * 90)
    print(f"全量评估（{len(items)} queries）— 各指标（macro average）")
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
