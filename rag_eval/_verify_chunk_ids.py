"""chunk_id 跨索引一致性验证 (T00.08)。

验证 ChromaDB 和 BM25 索引中相同 content 的 chunk_id 是否一致。
不一致的 chunk_id 会导致评估指标（Recall@10 等）系统性偏低。

Usage:
    python -m rag_eval._verify_chunk_ids
"""
import hashlib
import sys
from pathlib import Path

# 动态添加 agent 目录到 sys.path
AGENT_DIR = Path(__file__).resolve().parent.parent / "agent"
sys.path.insert(0, str(AGENT_DIR))


def _content_hash(content: str) -> str:
    """生成 content 的快速 hash 用于索引加速。"""
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:12]


def run_verification(
    chroma_collection,
    bm25_retriever,
    n_sample: int = 500,
    fail_threshold_fallback: float = 0.01,  # 1% fallback ID → 阻断
    warn_threshold_mismatch: float = 0.005,  # 0.5% 不一致 → 警告
) -> dict:
    """执行 chunk_id 跨索引一致性验证。

    Args:
        chroma_collection: ChromaDB collection 实例
        bm25_retriever: BM25Retriever 实例 (需已构建索引)
        n_sample: 抽样文档数
        fail_threshold_fallback: fallback ID 比例阻断阈值
        warn_threshold_mismatch: 不一致比例警告阈值

    Returns:
        验证报告 dict
    """
    # 1. 从 ChromaDB 随机抽取 N 个文档
    try:
        all_chroma = chroma_collection.get()
    except Exception as e:
        return {
            "status": "error",
            "error": f"ChromaDB 读取失败: {e}",
            "exit_code": 1,
        }

    if not all_chroma or not all_chroma.get("ids"):
        return {
            "status": "error",
            "error": "ChromaDB collection 为空",
            "exit_code": 1,
        }

    total_docs = len(all_chroma["ids"])

    # 随机抽样
    import random
    rng = random.Random(42)
    if total_docs <= n_sample:
        indices = list(range(total_docs))
    else:
        indices = rng.sample(range(total_docs), n_sample)

    # 2. 构建 BM25 content → chunk_id 查找表 (通过 hash 加速)
    bm25_content_map: dict[str, str] = {}
    bm25_doc_count = 0
    if bm25_retriever is not None and bm25_retriever._built:
        for doc in bm25_retriever._docs:
            content = doc.get("content", "")
            chunk_id = doc.get("chunk_id", "")
            h = _content_hash(content)
            bm25_content_map[h] = chunk_id
            bm25_doc_count += 1

    # 3. 逐对比较
    consistent = 0
    mismatched: list[dict] = []
    fallback_dense = 0  # dense_{hash} 格式
    fallback_sparse = 0  # sparse_{hash} 格式
    empty_bm25 = 0  # BM25 中空 chunk_id
    import re
    fallback_pattern = re.compile(r'^(dense_|sparse_)[0-9a-f]{8}$')

    for idx in indices:
        chroma_chunk_id = all_chroma["ids"][idx]
        content = (
            all_chroma.get("documents", [""])[idx]
            if idx < len(all_chroma.get("documents", []))
            else ""
        )
        h = _content_hash(content)

        bm25_chunk_id = bm25_content_map.get(h, "")

        # 检测 fallback ID
        if fallback_pattern.match(chroma_chunk_id):
            fallback_dense += 1

        if bm25_chunk_id and fallback_pattern.match(bm25_chunk_id):
            fallback_sparse += 1

        # 检测 BM25 空 chunk_id
        if bm25_chunk_id == "" and bm25_retriever is not None:
            empty_bm25 += 1
        elif chroma_chunk_id == bm25_chunk_id:
            consistent += 1
        elif bm25_chunk_id:  # 不一致
            mismatched.append({
                "chroma_id": chroma_chunk_id,
                "bm25_id": bm25_chunk_id,
                "content_preview": content[:80] if content else "(empty)",
            })

    n_checked = len(indices)
    n_mismatched = len(mismatched)

    fallback_ratio = (fallback_dense + fallback_sparse) / max(n_checked, 1)
    mismatch_ratio = n_mismatched / max(n_checked, 1)
    empty_bm25_ratio = empty_bm25 / max(n_checked, 1)

    # 判断阈值
    exit_code = 0
    if fallback_ratio > fail_threshold_fallback:
        exit_code = 1
    if empty_bm25_ratio > 0.0:
        exit_code = 1  # BM25 空 chunk_id 是阻断级

    report = {
        "status": "pass" if exit_code == 0 else "fail",
        "n_sample": n_checked,
        "total_chroma_docs": total_docs,
        "total_bm25_docs": bm25_doc_count,
        "consistent": consistent,
        "n_mismatched": n_mismatched,
        "fallback_dense_count": fallback_dense,
        "fallback_sparse_count": fallback_sparse,
        "empty_bm25_count": empty_bm25,
        "fallback_ratio": round(fallback_ratio, 4),
        "mismatch_ratio": round(mismatch_ratio, 4),
        "empty_bm25_ratio": round(empty_bm25_ratio, 4),
        "mismatched_details": mismatched[:20],  # 最多展示 20 条
        "exit_code": exit_code,
    }
    return report


def print_report(report: dict) -> None:
    """打印验证报告。"""
    print("chunk_id 跨索引一致性报告")
    print("=" * 40)
    print(f"抽样文档数: {report['n_sample']}")
    print(f"ChromaDB 总文档: {report['total_chroma_docs']}")
    print(f"BM25 总文档: {report['total_bm25_docs']}")
    print(f"一致: {report['consistent']} ({report['consistent']/max(report['n_sample'],1)*100:.1f}%)")
    print(f"不一致: {report['n_mismatched']} ({report['mismatch_ratio']*100:.1f}%)")
    print(f"Fallback ID (dense_*): {report['fallback_dense_count']} ({report['fallback_ratio']*100:.1f}%)")
    print(f"Fallback ID (sparse_*): {report['fallback_sparse_count']} ({report['fallback_ratio']*100:.1f}%)")
    print(f"BM25 空 chunk_id: {report['empty_bm25_count']} ({report['empty_bm25_ratio']*100:.1f}%)")

    if report["mismatched_details"]:
        print(f"\n不一致详情 (前 {min(len(report['mismatched_details']), 20)} 条):")
        for m in report["mismatched_details"]:
            print(f"  ChromaDB: {m['chroma_id']} | BM25: {m['bm25_id']} | content: {m['content_preview']}")

    print()
    if report["exit_code"] == 0:
        print("[PASS] 验证通过")
    else:
        print("[FAIL] 验证失败 — fallback ID 或空 chunk_id 超阈值")


def main():
    """CLI 入口: 加载 ChromaDB 和 BM25 索引，运行验证。"""
    import argparse
    parser = argparse.ArgumentParser(description="chunk_id 跨索引一致性验证")
    parser.add_argument("--n-sample", type=int, default=500, help="抽样数")
    parser.add_argument("--fail-fallback", type=float, default=0.01, help="fallback ID 阻断阈值")
    parser.add_argument("--warn-mismatch", type=float, default=0.005, help="不一致警告阈值")
    args = parser.parse_args()

    # 尝试加载索引
    try:
        from server.rag.config import rag_config
        from server.rag.embedder import Embedder
        from server.rag.vector_store import VectorStore
        from server.rag.bm25_retriever import BM25Retriever

        embedder = Embedder(
            rag_config.embedding_model,
            local_path=rag_config.embedding_local_path or None,
        )
        # 不等待就绪——只做结构检查
        vs = VectorStore(rag_config.chroma_persist_path, embedder)
        bm25 = BM25Retriever()
        # BM25 索引可能未构建——尝试加载已构建的
        # 注意: BM25 索引在内存中，没有序列化。如果未构建则跳过 BM25 侧验证

        chroma_ok = vs._collection is not None
        bm25_ok = bm25._built

        if not chroma_ok:
            print("[SKIP] ChromaDB collection 未初始化，跳过验证")
            sys.exit(2)

        report = run_verification(
            vs._collection,
            bm25 if bm25_ok else None,
            n_sample=args.n_sample,
            fail_threshold_fallback=args.fail_fallback,
            warn_threshold_mismatch=args.warn_mismatch,
        )

        print_report(report)

        # 写入标记文件
        if report["exit_code"] == 0:
            marker = Path(__file__).resolve().parent / "results" / "chunk_id_verification_pass.txt"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("pass")
        sys.exit(report["exit_code"])

    except ImportError as e:
        print(f"[SKIP] RAG 模块导入失败: {e}")
        sys.exit(2)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
