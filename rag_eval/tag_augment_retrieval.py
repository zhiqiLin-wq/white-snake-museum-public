"""检索层验证：dense top-100 + sparse 标签增强 top-100 能否捞回 unretrievable 核心。

- dense：原始 dense（embed query -> ChromaDB top-100）
- sparse：标签增强 BM25（索引「原文+标签词」，query「原文+标签词」top-100）
- 候选池 = dense100 ∪ sparse100 去重
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
for p in [str(PROJECT_ROOT), str(AGENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rag_eval.loader import load_corpus, load_queries
from rag_eval.apply_boost import build_raw_to_canonical, normalize_chunk_label

RESULTS = PROJECT_ROOT / "rag_eval" / "results"
GT = PROJECT_ROOT / "rag_eval" / "ground_truth"
TOP_K = 100


def build_chunk_tags(label, genre):
    tags = []
    if genre == "research_literature":
        tags += [t["canonical"] for t in label.get("分析主题", []) if isinstance(t, dict)]
        ent = label.get("引用实体", {}) or {}
        tags += [p["canonical"] for p in ent.get("persons", []) if isinstance(p, dict)]
        tags += [l["canonical"] for l in ent.get("locations", []) if isinstance(l, dict)]
        tags += [p["canonical"] for p in label.get("引用情节", []) if isinstance(p, dict)]
    else:
        tags += [p["canonical"] for p in label.get("persons", []) if isinstance(p, dict)]
        tags += [l["canonical"] for l in label.get("locations", []) if isinstance(l, dict)]
        tags += [p["canonical"] for p in label.get("plot_unit", []) if isinstance(p, dict)]
        pd = label.get("plot_detail", "")
        if pd:
            tags.append(pd)
    return [t for t in tags if t and t != "待删除"]


def build_query_tags(query_label):
    tags = []
    for field in ["persons", "locations", "plot_unit", "分析主题"]:
        for x in (query_label.get(field) or []):
            if isinstance(x, dict) and x.get("canonical"):
                tags.append(x["canonical"])
    pd = query_label.get("plot_detail", "")
    if pd:
        tags.append(pd)
    return [t for t in tags if t and t != "待删除"]


def build_bm25(docs):
    from server.rag.bm25_retriever import BM25Retriever, BM25Config
    from server.rag.config import rag_config
    bm25 = BM25Retriever(config=BM25Config(
        k1=rag_config.bm25_k1, b=rag_config.bm25_b, tokenizer=rag_config.bm25_tokenizer,
    ))
    bm25.index(docs)
    return bm25


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

    # 标签增强文本
    augmented = {}
    for cid, v in chunk_labels.items():
        lab = v.get("label") or {}
        if "_error" in lab:
            continue
        nl = normalize_chunk_label(lab, v["genre"], r2c)
        tags = build_chunk_tags(nl, v["genre"])
        summary = nl.get("summary", "")
        if summary:
            tags.append(summary)
        original = corpus.get(cid, {}).get("text", "")
        augmented[cid] = original + " " + " ".join(tags)

    # 标签增强 BM25
    aug_docs = [{"chunk_id": cid, "content": text, "metadata": {}} for cid, text in augmented.items()]
    aug_bm25 = build_bm25(aug_docs)

    # dense 组件（embedder + vector_store）
    from rag_eval.detail_log import build_components
    embedder, vector_store, _, _ = build_components()

    total_unretr = 0
    dense_hit = 0
    sparse_hit = 0
    pool_hit = 0
    for item in picked:
        ds, qid = item["dataset"], item["query_id"]
        qtext = qtext_by_ds[ds].get(qid, "")
        qlabel = query_labels.get(f"{ds}/{qid}", {})
        if not isinstance(qlabel, dict):
            continue
        qtags = build_query_tags(qlabel)
        unretr = set(item["unretrievable_chunks"])
        total_unretr += len(unretr)

        # dense top-100
        qvec = embedder.embed_query(qtext)
        dense_raw = vector_store.query(qvec, top_k=TOP_K)
        dense_retrieved = {r["chunk_id"] for r in dense_raw}

        # sparse 标签增强 top-100
        aug_query = (qtext + " " + " ".join(qtags)).strip()
        aug_results = aug_bm25.search(aug_query, top_k=TOP_K)
        aug_retrieved = {r["chunk_id"] for r in aug_results}

        pool = dense_retrieved | aug_retrieved
        dense_hit += len(unretr & dense_retrieved)
        sparse_hit += len(unretr & aug_retrieved)
        pool_hit += len(unretr & pool)

        print(f"[{ds}/{qid}] unretr={len(unretr)} dense100捞回={len(unretr & dense_retrieved)} "
              f"sparse增强100捞回={len(unretr & aug_retrieved)} 候选池捞回={len(unretr & pool)}", flush=True)

    print("\n" + "=" * 60)
    print(f"unretrievable 核心总数: {total_unretr}")
    print(f"dense top-100 捞回: {dense_hit}/{total_unretr} = {dense_hit/total_unretr:.3f}")
    print(f"sparse 标签增强 top-100 捞回: {sparse_hit}/{total_unretr} = {sparse_hit/total_unretr:.3f}")
    print(f"候选池(dense100 ∪ sparse100) 捞回: {pool_hit}/{total_unretr} = {pool_hit/total_unretr:.3f}")


if __name__ == "__main__":
    main()
