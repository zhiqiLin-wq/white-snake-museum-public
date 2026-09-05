"""详细检索日志：4 数据集 × D/S/F 三配置，逐 query 完整原文 + 命中/MISS + 标准答案。

与评测口径一致（bge-large ONNX + 指令前缀、BM25 k1=1.0/b=0.9、别名扩展 + 研究降权 + 版本 boost）。
每条 query 的 chunk 原文完整写出，不截断。

Usage:
    python rag_eval/detail_log.py
"""
import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(AGENT_DIR))

DATASETS = ["academic_deepseek", "academic_qwen", "daily_deepseek", "daily_qwen"]


def build_components():
    from server.rag.config import rag_config
    from server.rag.embedder import Embedder
    from server.rag.vector_store import VectorStore
    from server.rag.bm25_retriever import BM25Retriever
    from server.rag.query_expander import QueryExpander

    print("[INIT] 加载 embedder ...", flush=True)
    embedder = Embedder(rag_config.embedding_model, local_path=rag_config.embedding_local_path or None)
    embedder.start_loading()
    waited = 0
    while not embedder.is_ready and waited < 300:
        time.sleep(1)
        waited += 1
    if not embedder.is_ready:
        raise RuntimeError("embedder 未就绪")

    vector_store = VectorStore(rag_config.chroma_persist_path, embedder)
    if not vector_store.is_healthy() or vector_store.collection_count() == 0:
        raise RuntimeError("VectorStore 为空或不可用")

    bm25 = BM25Retriever()
    if vector_store._collection is not None:
        all_data = vector_store._collection.get()
        bm25_docs = []
        for i, content in enumerate(all_data.get("documents", [])):
            meta = all_data.get("metadatas", [{}])[i] if i < len(all_data.get("metadatas", [])) else {}
            cid = all_data.get("ids", [""])[i] if i < len(all_data.get("ids", [])) else ""
            bm25_docs.append({"chunk_id": cid, "content": content, "metadata": meta})
        bm25.index(bm25_docs)

    expander = QueryExpander()
    return embedder, vector_store, bm25, expander


def build_retrievers(vector_store, bm25):
    from server.rag.config import rag_config
    from server.rag.retriever import HybridRetriever, HybridRetrieverConfig

    d_cfg = HybridRetrieverConfig(
        bm25_enabled=False, reranker_enabled=False,
        fusion_method="score_normalized", fusion_dense_weight=0.5,
        dense_candidate_k=100, sparse_candidate_k=100,
        version_boost_factor=rag_config.version_boost_factor,
    )
    f_cfg = HybridRetrieverConfig(
        bm25_enabled=True, reranker_enabled=False,
        fusion_method="score_normalized", fusion_dense_weight=0.5,
        dense_candidate_k=100, sparse_candidate_k=100,
        version_boost_factor=rag_config.version_boost_factor,
    )
    d_retriever = HybridRetriever(vector_store, bm25_retriever=None, reranker=None, config=d_cfg)
    f_retriever = HybridRetriever(vector_store, bm25_retriever=bm25, reranker=None, config=f_cfg)
    return d_retriever, bm25, f_retriever


def preprocess_query(text, expander):
    from server.rag.query_intent import research_penalty_for, detect_version_filters
    e = expander.expand_weighted(text)
    return {
        "dense_query": text,
        "sparse_query": e["expanded_query"],
        "sparse_weights": e["token_weights"],
        "research_penalty": research_penalty_for(text),
        "version_boost": detect_version_filters(text),
        "added_aliases": e["added_aliases"],
    }


def run_config(d_retriever, bm25, f_retriever, config_id, pp):
    from server.rag.retriever import HybridRetrieverConfig

    eq_dict = {
        "dense_query": pp["dense_query"],
        "sparse_query": pp["sparse_query"],
        "sparse_weights": pp["sparse_weights"],
        "research_penalty": pp["research_penalty"],
        "version_boost": pp["version_boost"],
    }
    if config_id == "S":
        return asyncio.run(asyncio.to_thread(
            bm25.search, pp["sparse_query"], 10, None, pp["sparse_weights"]
        ))
    elif config_id == "D":
        return asyncio.run(d_retriever.retrieve(eq_dict, top_k=10))
    else:  # F
        return asyncio.run(f_retriever.retrieve(eq_dict, top_k=10))


def fmt_meta(meta):
    return f"genre={meta.get('genre','?')}/dynasty={meta.get('dynasty','?')}/source={meta.get('source_type','?')}"


def main():
    from rag_eval.loader import load_corpus, load_queries, load_qrels
    from rag_eval.metrics import recall_at_k

    embedder, vector_store, bm25, expander = build_components()
    d_retriever, s_bm25, f_retriever = build_retrievers(vector_store, bm25)

    gt = PROJECT_ROOT / "rag_eval" / "ground_truth"
    corpus = load_corpus(gt / "corpus.jsonl")
    id_to_content = {doc_id: doc["text"] for doc_id, doc in corpus.items()}
    id_to_meta = {doc_id: doc["metadata"] for doc_id, doc in corpus.items()}

    ts = time.strftime("%Y%m%dT%H%M%S")
    out_path = PROJECT_ROOT / "rag_eval" / "results" / f"detail_log_{ts}.txt"
    log = open(str(out_path), "w", encoding="utf-8")

    def w(s=""):
        log.write(s + "\n")

    w(f"详细检索日志 — 生成时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"配置: bge-large-zh-v1.5(指令前缀) + BM25(k1=1.0/b=0.9) 融合(dense=0.5/sparse=0.5)")
    w(f"研究文献降权=0.5, 版本 boost=1.2, 无 reranker")
    w("")

    total_q = 0
    for ds in DATASETS:
        queries = load_queries(gt / f"queries_{ds}.jsonl")
        qrels = load_qrels(gt / f"qrels_{ds}_augmented.tsv")
        w("=" * 90)
        w(f"【数据集 {ds}】 {len(queries)} 条 query")
        w("=" * 90)

        for qi, q in enumerate(queries):
            qid = q.get("query_id", "")
            text = q.get("text", "")
            rel = qrels.get(qid, {})
            rel_set = set(rel.keys())
            if not rel_set:
                continue
            total_q += 1

            pp = preprocess_query(text, expander)
            w("")
            w("-" * 90)
            w(f"[{ds}] Query {qi+1}/{len(queries)}  {qid}")
            w(f"原始 query: {text}")
            w(f"预处理: 别名={pp['added_aliases']} | 研究降权={pp['research_penalty']} | 版本boost={pp['version_boost']}")

            # 标准答案（qrels 全部相关 chunk，完整原文）
            w(f"\n【标准答案】qrels 相关 chunk（rel≥1）共 {len(rel_set)} 个:")
            for idx, cid in enumerate(sorted(rel_set), 1):
                meta = id_to_meta.get(cid, {})
                content = id_to_content.get(cid, "(未找到)")
                w(f"  [{idx}] {cid}  ({fmt_meta(meta)}/rel={rel.get(cid)})")
                w(f"      原文: {content}")

            # 三配置检索
            for cfg_id, cfg_label in [("D", "Dense-only"), ("S", "BM25-only"), ("F", "Fusion")]:
                results = run_config(d_retriever, s_bm25, f_retriever, cfg_id, pp)
                retrieved_ids = [r.get("chunk_id", "") for r in results]
                hits = [cid for cid in retrieved_ids if cid in rel_set]
                rec = recall_at_k(retrieved_ids, rel_set, 10)
                w(f"\n【{cfg_label} ({cfg_id})】 Recall@10 = {len(hits)}/{len(rel_set)} = {rec:.3f}")
                for rank, r in enumerate(results, 1):
                    cid = r.get("chunk_id", "")
                    mark = "HIT " if cid in rel_set else "MISS"
                    score = r.get("score", r.get("bm25_score", r.get("relevance_score", 0.0)))
                    meta = id_to_meta.get(cid, r.get("metadata", {}))
                    content = id_to_content.get(cid, r.get("content", "(未找到)"))
                    w(f"  #{rank:2d} [{mark}] {cid}  ({fmt_meta(meta)}/score={score:.4f})")
                    w(f"       原文: {content}")

            # 汇总
            missed = rel_set - set(retrieved_ids)  # 用最后一次 F 的
            w(f"\n【汇总】D/S/F 命中详情见上；qrels 相关 {len(rel_set)} 个")

            if (qi + 1) % 50 == 0:
                print(f"  [{ds}] {qi+1}/{len(queries)} 完成", flush=True)

    log.close()
    print(f"\n完成。共 {total_q} 条有标注 query，日志: {out_path}", flush=True)


if __name__ == "__main__":
    main()
