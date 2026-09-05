"""RAG 检索评估 — 数据集生成工具 (Phase 1)。

CLI 接口:
    python -m rag_eval.generate_dataset --export-corpus
    python -m rag_eval.generate_dataset --sample-ci
    python -m rag_eval.generate_dataset --generate-config
    python -m rag_eval.generate_dataset --stats
    python -m rag_eval.generate_dataset --validate-qrels
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# 动态添加 agent 目录
AGENT_DIR = Path(__file__).resolve().parent.parent / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))


# ==============================================================================
# T01.02: cmd_export_corpus
# ==============================================================================
def cmd_export_corpus(output_dir: Path) -> None:
    """从 Chunker 导出 corpus.jsonl。

    Steps:
        1. 初始化 Embedder (等待就绪，最多 300s)
        2. 初始化 Chunker + chunk_all()
        3. 调用 loader.export_corpus() 导出
    """
    from server.rag.config import rag_config
    from server.rag.embedder import Embedder
    from server.rag.chunker import Chunker
    from rag_eval.loader import export_corpus

    print("[1/3] 加载 Embedder...")
    embedder = Embedder(
        rag_config.embedding_model,
        local_path=rag_config.embedding_local_path or None,
    )
    embedder.start_loading()
    waited = 0
    while not embedder.is_ready and waited < 300:
        time.sleep(1)
        waited += 1
        if waited % 30 == 0:
            print(f"  等待 Embedder... ({waited}s)")
    if not embedder.is_ready:
        print(f"[ERROR] Embedder 加载失败: {embedder.load_error}")
        sys.exit(1)

    print("[2/3] 执行分块 (Chunker.chunk_all)...")
    chunker = Chunker(data_dir=rag_config.data_dir_path, embedder=embedder)
    chunks = chunker.chunk_all()
    print(f"  总 chunk 数: {len(chunks)}")

    # 统计
    genres = Counter()
    dynasties = Counter()
    for doc in chunks:
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        g = meta.get("genre", "unknown")
        d = meta.get("dynasty", "unknown")
        genres[g] += 1
        dynasties[d] += 1

    print(f"  体裁分布: {dict(genres)}")
    print(f"  朝代分布: {dict(dynasties)}")

    print(f"[3/3] 导出 corpus.jsonl → {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    export_corpus(chunks, output_dir / "corpus.jsonl")
    print("  [OK] 导出完成")


# ==============================================================================
# T01.03: cmd_sample_ci
# ==============================================================================
def cmd_sample_ci(queries_path: Path, output_dir: Path) -> None:
    """从全量 queries.jsonl 中采样 CI 子集。"""
    from rag_eval.loader import load_queries, sample_ci_subset

    print(f"[1/2] 加载全量查询: {queries_path}")
    queries = load_queries(queries_path)
    print(f"  总查询数: {len(queries)}")

    print("[2/2] 采样 CI 子集...")
    ci_queries = sample_ci_subset(queries)
    print(f"  CI 子集: {len(ci_queries)} 条")

    # 统计
    by_cat = Counter(q.get("category") for q in ci_queries)
    by_diff = Counter(q.get("difficulty") for q in ci_queries)
    print(f"  按类别: {dict(by_cat)}")
    print(f"  按难度: {dict(by_diff)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "queries_ci.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for q in ci_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"  [OK] → {output_path}")


# ==============================================================================
# T01.04: cmd_generate_config
# ==============================================================================
def cmd_generate_config(corpus_path: Path, output_dir: Path) -> None:
    """扫描 corpus.jsonl 自动生成 dataset_config.yaml。"""
    from rag_eval.loader import load_corpus

    print(f"[1/2] 加载语料库: {corpus_path}")
    corpus = load_corpus(corpus_path)

    genres = set()
    dynasties = set()
    for doc_id, doc in corpus.items():
        meta = doc.get("metadata", {})
        g = meta.get("genre", "")
        d = meta.get("dynasty", "")
        if g:
            genres.add(g)
        if d:
            dynasties.add(d)

    print(f"  总文档数: {len(corpus)}")
    print(f"  体裁: {sorted(genres)}")
    print(f"  朝代: {sorted(dynasties)}")

    config = {
        "dataset": {
            "name": "white-snake-literature-rag-eval",
            "version": "1.0.0",
            "created_at": "2026-08-10",
            "language": "zh",
            "domain": "classical_chinese_literature",
            "corpus": {
                "file": "corpus.jsonl",
                "encoding": "utf-8",
                "total_docs": len(corpus),
            },
            "queries": {
                "file": "queries.jsonl",
                "encoding": "utf-8",
                "total_queries": 200,
                "categories": [
                    "entity", "location", "plot", "comparison", "motif",
                    "filtered", "alias", "intertext", "robustness",
                ],
                "difficulty_levels": ["easy", "medium", "hard"],
                "ci_subset_file": "queries_ci.jsonl",
                "ci_subset_queries": 30,
            },
            "qrels": {
                "file": "qrels.tsv",
                "encoding": "utf-8",
                "relevance_levels": {
                    0: "not_relevant",
                    1: "partially_relevant",
                    2: "highly_relevant",
                },
                "min_annotators": 2,
                "target_kappa": 0.70,
            },
        },
    }

    import yaml
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "dataset_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    print(f"\n[OK] → {config_path}")


# ==============================================================================
# T01.05: cmd_corpus_stats
# ==============================================================================
def cmd_corpus_stats(corpus_path: Path) -> None:
    """语料库统计摘要。"""
    from rag_eval.loader import load_corpus

    corpus = load_corpus(corpus_path)

    # 按体裁统计
    genre_stats: dict = {}
    for doc_id, doc in corpus.items():
        meta = doc.get("metadata", {})
        genre = meta.get("genre", "unknown")
        text_len = len(doc.get("text", ""))

        if genre not in genre_stats:
            genre_stats[genre] = {"count": 0, "total_chars": 0, "min_chars": 99999, "max_chars": 0}
        gs = genre_stats[genre]
        gs["count"] += 1
        gs["total_chars"] += text_len
        gs["min_chars"] = min(gs["min_chars"], text_len)
        gs["max_chars"] = max(gs["max_chars"], text_len)

    print("# Corpus 统计")
    print(f"\n总文档数: {len(corpus)}")
    print(f"\n## 按体裁")
    print(f"{'Genre':<25} {'Count':<8} {'Avg Chars':<12} {'Min':<8} {'Max':<8}")
    print("-" * 65)
    for genre, stats in sorted(genre_stats.items()):
        avg = stats["total_chars"] / max(stats["count"], 1)
        print(f"{genre:<25} {stats['count']:<8} {avg:<12.0f} {stats['min_chars']:<8} {stats['max_chars']:<8}")


# ==============================================================================
# T01.06: cmd_validate_qrels
# ==============================================================================
def cmd_validate_qrels(queries_path: Path, qrels_path: Path, corpus_path: Path) -> None:
    """qrels 格式校验。"""
    from rag_eval.loader import load_queries, load_qrels, load_corpus

    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)
    corpus = load_corpus(corpus_path)

    query_ids = {q["query_id"] for q in queries}
    corpus_ids = set(corpus.keys())

    passed = 0
    failed = 0

    # 1. query_id 存在
    for qid in qrels:
        if qid not in query_ids:
            print(f"[FAIL] qrels 中的 query_id '{qid}' 不在 queries.jsonl 中")
            failed += 1

    # 2. doc_id 存在
    for qid, docs in qrels.items():
        for did in docs:
            if did not in corpus_ids:
                print(f"[FAIL] qrels 中的 doc_id '{did}' (query={qid}) 不在 corpus.jsonl 中")
                failed += 1

    # 3. relevance 值检查
    for qid, docs in qrels.items():
        for did, rel in docs.items():
            if rel not in (1, 2):
                print(f"[FAIL] relevance={rel} (query={qid}, doc={did}) — 期望 1 或 2")
                failed += 1

    if failed == 0:
        print("[PASS] qrels 校验通过")
    else:
        print(f"\n[FAIL] {failed} 个校验项失败")
        sys.exit(1)


# ==============================================================================
# CLI
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="数据集生成工具")
    parser.add_argument("--export-corpus", action="store_true", help="导出 corpus.jsonl")
    parser.add_argument("--sample-ci", action="store_true", help="采样 CI 子集")
    parser.add_argument("--generate-config", action="store_true", help="生成 dataset_config.yaml")
    parser.add_argument("--stats", action="store_true", help="corpus 统计摘要")
    parser.add_argument("--validate-qrels", action="store_true", help="校验 qrels")
    parser.add_argument("--queries", type=str, default="rag_eval/ground_truth/queries.jsonl")
    parser.add_argument("--qrels", type=str, default="rag_eval/ground_truth/qrels.tsv")
    parser.add_argument("--corpus", type=str, default="rag_eval/ground_truth/corpus.jsonl")
    parser.add_argument("--output", type=str, default="rag_eval/ground_truth/")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.export_corpus:
        cmd_export_corpus(output_dir)
    elif args.sample_ci:
        cmd_sample_ci(Path(args.queries), output_dir)
    elif args.generate_config:
        cmd_generate_config(Path(args.corpus), output_dir)
    elif args.stats:
        cmd_corpus_stats(Path(args.corpus))
    elif args.validate_qrels:
        cmd_validate_qrels(Path(args.queries), Path(args.qrels), Path(args.corpus))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
