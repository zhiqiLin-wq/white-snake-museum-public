"""标签向量库（标签语义检索）验证：对比 sparse 标签增强，算合并总捞回。

- 标签语义路：chunk 标签文本 embed 成向量库，query 标签文本 embed，余弦 top-100
- dense100：原始 dense top-100
- sparse 标签增强 100：标签增强 BM25 top-100
- 合并候选池 = 三者 union
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
for p in [str(PROJECT_ROOT), str(AGENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rag_eval.loader import load_corpus, load_queries
from rag_eval.apply_boost import build_raw_to_canonical, normalize_chunk_label
from rag_eval.tag_augment_retrieval import build_chunk_tags, build_query_tags, build_bm25

RESULTS = PROJECT_ROOT / "rag_eval" / "results"
GT = PROJECT_ROOT / "rag_eval" / "ground_truth"
TOP_K = 100


def main():
    vocab = json.load(open(RESULTS / "merged_vocab2.json", encoding="utf-8"))
    r2c = build_raw_to_canonical(vocab)
    chunk_labels = json.load(open(RESULTS / "chunk_labels.json", encoding="utf-8"))["chunk_labels"]
    query_labels = {
        f"{r['dataset']}/{r['query_id']}": r["query_label"]
        for r in json.load(open(RESULTS / "query_labels.json", encoding="utf-8"))
    }
    picked = json.load(open(RESULTS / "picked_unretrievable_queries.json", encoding="utf-8"))
    corpus = load_corpus(GT / "corpus.jsonl")

    qtext_by_ds = {}
    for ds in sorted({x["dataset"] for x in picked}):
        qs = load_queries(GT / f"queries_{ds}.jsonl")
        qtext_by_ds[ds] = {q["query_id"]: q["text"] for q in qs}

    # 加载组件（embedder + vector_store + bm25 + expander，一次）
    from rag_eval.detail_log import build_components
    embedder, vector_store, _, _ = build_components()

    # 标签文本 + 增强文本
    chunk_tag_text = {}
    augmented = {}
    for cid, v in chunk_labels.items():
        lab = v.get("label") or {}
        if "_error" in lab:
            continue
        nl = normalize_chunk_label(lab, v["genre"], r2c)
        tags = build_chunk_tags(nl, v["genre"])
        summary = nl.get("summary", "")
        chunk_tag_text[cid] = " ".join(tags + ([summary] if summary else []))
        original = corpus.get(cid, {}).get("text", "")
        augmented[cid] = original + " " + chunk_tag_text[cid]

    # embed 标签文本 -> 标签向量库（不加 BGE 指令前缀，标签文本是词拼接）
    chunk_ids = list(chunk_tag_text.keys())
    chunk_matrix = np.asarray(embedder.embed([chunk_tag_text[cid] for cid in chunk_ids]), dtype=float)
    print(f"[INIT] 标签向量库: {len(chunk_ids)} 个 chunk", flush=True)

    # sparse 标签增强 BM25
    aug_bm25 = build_bm25([{"chunk_id": cid, "content": text, "metadata": {}}
                           for cid, text in augmented.items()])

    total_unretr = 0
    tag_hit = dense_hit = sparse_hit = pool_hit = 0
    for item in picked:
        ds, qid = item["dataset"], item["query_id"]
        qtext = qtext_by_ds[ds].get(qid, "")
        qlabel = query_labels.get(f"{ds}/{qid}", {})
        if not isinstance(qlabel, dict):
            continue
        qtags = build_query_tags(qlabel)
        unretr = set(item["unretrievable_chunks"])
        total_unretr += len(unretr)

        # 标签语义路（余弦 top-100）
        q_tag_vec = np.asarray(embedder.embed([" ".join(qtags)])[0], dtype=float)
        scores = chunk_matrix @ q_tag_vec
        tag_top = {chunk_ids[i] for i in np.argsort(scores)[::-1][:TOP_K]}

        # dense100
        qvec = embedder.embed_query(qtext)
        dense_top = {r["chunk_id"] for r in vector_store.query(qvec, top_k=TOP_K)}

        # sparse 标签增强 100
        aug_query = (qtext + " " + " ".join(qtags)).strip()
        sparse_top = {r["chunk_id"] for r in aug_bm25.search(aug_query, top_k=TOP_K)}

        tag_hit += len(unretr & tag_top)
        dense_hit += len(unretr & dense_top)
        sparse_hit += len(unretr & sparse_top)
        pool_hit += len(unretr & (tag_top | dense_top | sparse_top))

        print(f"[{ds}/{qid}] unretr={len(unretr)} 标签语义={len(unretr & tag_top)} "
              f"dense100={len(unretr & dense_top)} sparse增强={len(unretr & sparse_top)} "
              f"合并={len(unretr & (tag_top | dense_top | sparse_top))}", flush=True)

    print("\n" + "=" * 60)
    print(f"unretrievable 核心总数: {total_unretr}")
    print(f"标签语义路 top-100: {tag_hit}/{total_unretr} = {tag_hit/total_unretr:.3f}")
    print(f"dense top-100: {dense_hit}/{total_unretr} = {dense_hit/total_unretr:.3f}")
    print(f"sparse 标签增强 top-100: {sparse_hit}/{total_unretr} = {sparse_hit/total_unretr:.3f}")
    print(f"三者合并候选池: {pool_hit}/{total_unretr} = {pool_hit/total_unretr:.3f}")


if __name__ == "__main__":
    main()
