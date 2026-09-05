"""验证评估脚本可正确导入项目 RAG 模块。

Usage: python _verify_imports.py
"""
import sys
from pathlib import Path

# 动态添加 agent 目录到 sys.path
AGENT_DIR = Path(__file__).resolve().parent.parent / "agent"
sys.path.insert(0, str(AGENT_DIR))


def test_imports():
    """验证所有 RAG 核心模块可导入。"""
    errors: list[str] = []

    # 1. RAG 配置
    try:
        from server.rag.config import rag_config, RAGConfig
        assert rag_config is not None, "rag_config 单例为空"
        print(f"  [OK] server.rag.config (default_top_k={rag_config.default_top_k})")
    except Exception as e:
        errors.append(f"server.rag.config: {e}")

    # 2. HybridRetriever + Config
    try:
        from server.rag.retriever import HybridRetriever, HybridRetrieverConfig
        assert HybridRetrieverConfig is not None
        print("  [OK] server.rag.retriever (HybridRetriever + HybridRetrieverConfig)")
    except Exception as e:
        errors.append(f"server.rag.retriever: {e}")

    # 3. BM25Retriever
    try:
        from server.rag.bm25_retriever import BM25Retriever
        assert BM25Retriever is not None
        print("  [OK] server.rag.bm25_retriever")
    except Exception as e:
        errors.append(f"server.rag.bm25_retriever: {e}")

    # 4. Reranker
    try:
        from server.rag.reranker import Reranker
        assert Reranker is not None
        print("  [OK] server.rag.reranker")
    except Exception as e:
        errors.append(f"server.rag.reranker: {e}")

    # 5. Embedder
    try:
        from server.rag.embedder import Embedder
        assert Embedder is not None
        print("  [OK] server.rag.embedder")
    except Exception as e:
        errors.append(f"server.rag.embedder: {e}")

    # 6. VectorStore
    try:
        from server.rag.vector_store import VectorStore
        assert VectorStore is not None
        print("  [OK] server.rag.vector_store")
    except Exception as e:
        errors.append(f"server.rag.vector_store: {e}")

    # 7. Chunker + Genre + GENRE_CHUNK_DEFAULTS
    try:
        from server.rag.chunker import Chunker, GENRE_CHUNK_DEFAULTS, Genre
        assert Chunker is not None
        assert GENRE_CHUNK_DEFAULTS is not None
        assert len(GENRE_CHUNK_DEFAULTS) > 0, "GENRE_CHUNK_DEFAULTS 为空"
        print(f"  [OK] server.rag.chunker (Chunker + Genre + GENRE_CHUNK_DEFAULTS: {len(GENRE_CHUNK_DEFAULTS)} genres)")
    except Exception as e:
        errors.append(f"server.rag.chunker: {e}")

    # 汇总
    if errors:
        print("\n[FAIL] Import verification failed:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("\n[PASS] All imports OK")


if __name__ == "__main__":
    test_imports()
