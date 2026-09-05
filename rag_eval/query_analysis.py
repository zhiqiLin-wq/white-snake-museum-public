"""Query-level 召回分析 —— 逐条输出所有查询的检索详情。

对每条查询运行 F (Fusion) 配置，输出: 查询文本 / 相关chunk / 检索到的chunk / 差异。
按 Recall@10 从低到高排序，全部输出到日志文件。

Usage:
    python rag_eval/query_analysis.py --config rag_eval/configs/test_config_daily_deepseek.yaml --output rag_eval/results/query_analysis.log
"""
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
for p in [str(PROJECT_ROOT), str(AGENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


class TeeLogger:
    """同时写入 stdout 和文件。"""
    def __init__(self, filepath: Path):
        self.file = open(str(filepath), "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, msg: str):
        self.file.write(msg)
        self.file.flush()
        self.stdout.write(msg)

    def close(self):
        self.file.close()


def load_data(queries_file: Path, qrels_file: Path, corpus_file: Path):
    from rag_eval.loader import load_corpus, load_queries, load_qrels, load_all_qrels

    corpus = load_corpus(corpus_file)
    queries = load_queries(queries_file)
    qrels = load_qrels(qrels_file)
    all_qrels = load_all_qrels(qrels_file)
    id_to_content = {doc_id: doc["text"] for doc_id, doc in corpus.items()}
    return corpus, queries, qrels, all_qrels, id_to_content


def build_retriever():
    from server.rag.config import rag_config
    from server.rag.embedder import Embedder
    from server.rag.vector_store import VectorStore
    from server.rag.bm25_retriever import BM25Retriever
    from server.rag.retriever import HybridRetriever, HybridRetrieverConfig
    from server.rag.query_expander import QueryExpander

    print("[INIT] Loading embedder...")
    embedder = Embedder(rag_config.embedding_model, local_path=rag_config.embedding_local_path or None)
    embedder.start_loading()
    waited = 0
    while not embedder.is_ready and waited < 300:
        time.sleep(1)
        waited += 1
    if not embedder.is_ready:
        raise RuntimeError(f"Embedder not ready after {waited}s")

    print("[INIT] Connecting to ChromaDB...")
    vector_store = VectorStore(rag_config.chroma_persist_path, embedder)
    if not vector_store.is_healthy() or vector_store.collection_count() == 0:
        raise RuntimeError("VectorStore is empty or unhealthy. Run full eval first to build index.")

    print("[INIT] Building BM25 from ChromaDB chunks...")
    bm25 = BM25Retriever()
    if vector_store._collection is not None:
        all_data = vector_store._collection.get()
        bm25_docs = []
        for i, content in enumerate(all_data.get("documents", [])):
            meta = all_data.get("metadatas", [{}])[i] if i < len(all_data.get("metadatas", [])) else {}
            chunk_id = all_data.get("ids", [""])[i] if i < len(all_data.get("ids", [])) else ""
            bm25_docs.append({"chunk_id": chunk_id, "content": content, "metadata": meta})
        bm25.index(bm25_docs)
        print(f"  BM25 indexed: {bm25._total_docs} docs")

    expander = QueryExpander()

    cfg = HybridRetrieverConfig(
        bm25_enabled=True, reranker_enabled=False,
        fusion_method=rag_config.fusion_method_validated,
        fusion_k=rag_config.fusion_k,
        fusion_dense_weight=rag_config.fusion_dense_weight,
        dense_candidate_k=rag_config.dense_candidate_k,
        sparse_candidate_k=rag_config.sparse_candidate_k,
        reranker_output_k=rag_config.reranker_output_k,
    )
    print(f"[INIT] Config: method={cfg.fusion_method}, dense_w={cfg.fusion_dense_weight}, "
          f"cand_k={cfg.dense_candidate_k}/{cfg.sparse_candidate_k}")

    retriever = HybridRetriever(vector_store, bm25_retriever=bm25, reranker=None, config=cfg)
    return retriever, expander


def content_snippet(content: str, max_len: int = 200) -> str:
    """截取文本前 max_len 字符，换行转义。"""
    return content[:max_len].replace("\n", "\\n").replace("\r", "")


def run_analysis(retriever, expander, queries, qrels, id_to_content, log):
    import asyncio

    from server.rag.query_intent import research_penalty_for

    all_records = []
    total = len(queries)
    log.write(f"\n[RUN] Evaluating {total} queries with F (Fusion) config...\n\n")

    for idx, q in enumerate(queries):
        query_id = q.get("query_id", f"q_{idx:04d}")
        query_text = q.get("text", "")
        relevant_docs = qrels.get(query_id, {})
        relevant_ids = set(relevant_docs.keys())

        if not relevant_ids:
            all_records.append({
                "query_id": query_id, "text": query_text,
                "category": q.get("category", ""), "difficulty": q.get("difficulty", ""),
                "recall": None, "relevant_count": 0, "hit_count": 0,
                "relevant_ids": [], "hits": [], "retrieved_ids": [],
            })
            continue

        expand_result = expander.expand_weighted(query_text)
        eq_dict = {
            "dense_query": query_text,
            "sparse_query": expand_result["expanded_query"],
            "sparse_weights": expand_result["token_weights"],
            "research_penalty": research_penalty_for(query_text),
        }

        try:
            results = asyncio.run(retriever.retrieve(eq_dict, top_k=10))
        except Exception as e:
            log.write(f"  [ERR] {query_id}: {e}\n")
            continue

        retrieved_ids = [r.get("chunk_id", "") for r in results]
        retrieved_set = set(retrieved_ids)
        hits = retrieved_set & relevant_ids
        recall = len(hits) / len(relevant_ids)

        all_records.append({
            "query_id": query_id, "text": query_text,
            "category": q.get("category", ""), "difficulty": q.get("difficulty", ""),
            "recall": round(recall, 4),
            "relevant_ids": list(relevant_ids), "relevant_count": len(relevant_ids),
            "retrieved_ids": retrieved_ids, "hits": list(hits), "hit_count": len(hits),
        })

        if (idx + 1) % 100 == 0:
            log.write(f"  [{idx+1}/{total}] done\n")

    # 按 recall 从低到高排序（None 排最前面）
    all_records.sort(key=lambda r: (r["recall"] is None, r["recall"] or 0))

    # ---- 分布统计 ----
    buckets = {"zero": 0, "very_low": 0, "low": 0, "medium": 0, "high": 0, "perfect": 0, "no_annotation": 0}
    for r in all_records:
        if r["recall"] is None:
            buckets["no_annotation"] += 1
        elif r["recall"] == 0:
            buckets["zero"] += 1
        elif r["recall"] < 0.2:
            buckets["very_low"] += 1
        elif r["recall"] < 0.4:
            buckets["low"] += 1
        elif r["recall"] < 0.6:
            buckets["medium"] += 1
        elif r["recall"] < 0.8:
            buckets["high"] += 1
        else:
            buckets["perfect"] += 1

    total_with_annot = sum(v for k, v in buckets.items() if k != "no_annotation")
    log.write(f"\n{'='*60}\n")
    log.write(f"  Recall@10 Distribution ({total_with_annot} queries with annotations)\n")
    log.write(f"{'='*60}\n")
    for name, label in [("zero", "= 0"), ("very_low", "0~0.2"), ("low", "0.2~0.4"),
                          ("medium", "0.4~0.6"), ("high", "0.6~0.8"), ("perfect", "0.8~1.0")]:
        n = buckets[name]
        pct = n / max(total_with_annot, 1) * 100
        bar = "#" * int(pct)
        log.write(f"  {label:>10}: {n:>4} ({pct:>5.1f}%) {bar}\n")
    if buckets["no_annotation"]:
        log.write(f"  (no qrels): {buckets['no_annotation']} queries skipped\n")
    log.write(f"\n{'='*60}\n")

    # ---- 逐查询详情 ----
    log.write(f"\n{'='*80}\n")
    log.write(f"  PER-QUERY DETAILS (sorted by Recall@10, worst first)\n")
    log.write(f"{'='*80}\n")

    for i, r in enumerate(all_records):
        if r["recall"] is None:
            log.write(f"\n{'─'*80}\n")
            log.write(f"[{i+1}/{len(all_records)}] {r['query_id']}  |  NO QRELS ANNOTATION  "
                      f"|  {r['category']}  |  {r['difficulty']}\n")
            log.write(f"Query: {r['text'][:300]}\n")
            continue

        log.write(f"\n{'─'*80}\n")
        log.write(f"[{i+1}/{len(all_records)}] {r['query_id']}  |  recall={r['recall']:.2f}  "
                  f"|  hits={r['hit_count']}/{r['relevant_count']}  "
                  f"|  {r['category']}  |  {r['difficulty']}\n")
        log.write(f"Query: {r['text'][:300]}\n")

        # 稀疏查询（别名扩展后）
        expand_result = expander.expand_weighted(r["text"])
        if expand_result["added_aliases"]:
            log.write(f"Aliases added: {expand_result['added_aliases'][:10]}\n")

        # 所有相关 chunk（MISS 的排前面）
        missed = [rid for rid in r["relevant_ids"] if rid not in r["hits"]]
        hit_rel = [rid for rid in r["relevant_ids"] if rid in r["hits"]]
        log.write(f"\n  Relevant chunks ({r['relevant_count']} total, {len(missed)} missed):\n")
        for rel_id in missed + hit_rel:
            content = id_to_content.get(rel_id, "(NOT FOUND)")
            snippet = content_snippet(content, 150)
            mark = " [MISS]" if rel_id in missed else " [HIT]"
            log.write(f"  {rel_id}{mark}: {snippet}...\n")

        # Top-5 检索结果
        log.write(f"\n  Retrieved Top-5:\n")
        for rank, rid in enumerate(r["retrieved_ids"][:5], 1):
            content = id_to_content.get(rid, "")
            snippet = content_snippet(content, 120)
            rel_mark = " [REL]" if rid in r["relevant_ids"] else ""
            log.write(f"  #{rank} {rid}{rel_mark}: {snippet}...\n")

    # ---- 汇总 ----
    log.write(f"\n{'='*80}\n")
    n_zero = buckets["zero"]
    n_vlow = buckets["very_low"]
    log.write(f"  SUMMARY: {n_zero + n_vlow}/{total_with_annot} queries with recall < 0.2 "
              f"({(n_zero + n_vlow)/max(total_with_annot,1)*100:.1f}%)\n")
    log.write(f"  zero (recall=0): {n_zero}\n")
    log.write(f"  very_low (0<r<0.2): {n_vlow}\n")
    log.write(f"{'='*80}\n")

    return all_records


def main():
    parser = argparse.ArgumentParser(description="Query-level recall analysis — full dump to log file")
    parser.add_argument("--config", required=True, help="Path to YAML test config")
    parser.add_argument("--output", required=True, help="Output log file path")
    args = parser.parse_args()

    import yaml
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        test_config = yaml.safe_load(f)

    gt_dir = Path(test_config.get("evaluation", {}).get("dataset", {}).get("path", "rag_eval/ground_truth"))
    corpus_file = gt_dir / test_config.get("evaluation", {}).get("dataset", {}).get("corpus_file", "corpus.jsonl")
    queries_file = gt_dir / test_config.get("evaluation", {}).get("dataset", {}).get("queries_file", "queries.jsonl")
    qrels_file = gt_dir / test_config.get("evaluation", {}).get("dataset", {}).get("qrels_file", "qrels.tsv")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log = TeeLogger(output_path)

    log.write(f"Query-Level Recall Analysis\n")
    log.write(f"Config: {args.config}\n")
    log.write(f"Dataset: {queries_file.name} / {qrels_file.name}\n")
    log.write(f"Output: {output_path}\n")
    log.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    log.write(f"\nLoading: {queries_file} / {qrels_file} / {corpus_file}\n")
    corpus, queries, qrels, all_qrels, id_to_content = load_data(queries_file, qrels_file, corpus_file)
    log.write(f"  corpus: {len(corpus)} docs, queries: {len(queries)}, qrels: {len(qrels)} groups\n")

    retriever, expander = build_retriever()
    all_records = run_analysis(retriever, expander, queries, qrels, id_to_content, log)

    log.write(f"\nDone. Full log saved to: {output_path}\n")
    log.close()


if __name__ == "__main__":
    main()
