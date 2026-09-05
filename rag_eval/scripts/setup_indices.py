"""评估系统索引构建 —— ChromaDB + BM25。

运行条件:
    1. corpus.jsonl 已生成 (python -m rag_eval.generate_dataset --export-corpus)
    2. Embedding 模型已缓存

运行方式:
    cd white-snake-museum-public
    python -m rag_eval.scripts.setup_indices

输出:
    agent/chroma_db/     — ChromaDB 持久化向量库
    BM25 索引 (内存中构建，通过 VectorStore 的 bm25_retriever 同步)
"""
import sys
import time
from pathlib import Path

# 添加项目路径
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AGENT_DIR = _PROJECT_ROOT / "agent"
sys.path.insert(0, str(AGENT_DIR))

from server.rag.config import rag_config
from server.rag.embedder import Embedder
from server.rag.vector_store import VectorStore
from server.rag.bm25_retriever import BM25Retriever, BM25Config
from rag_eval.loader import load_corpus


def main():
    print("=" * 60)
    print("RAG 评估索引构建 (ChromaDB + BM25)")
    print("=" * 60)

    # ---- Step 1: 加载 Embedder ----
    print("\n[1/5] 加载 Embedder...")
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
            print(f"  等待中... ({waited}s)")
    if not embedder.is_ready:
        print(f"[ERROR] Embedder 加载失败: {embedder.load_error}")
        sys.exit(1)
    print(f"  Embedder 就绪 (等待 {waited}s)")

    # ---- Step 2: 加载 corpus.jsonl ----
    print("\n[2/5] 加载 corpus.jsonl...")
    corpus_path = _PROJECT_ROOT / "rag_eval" / "ground_truth" / "corpus.jsonl"
    if not corpus_path.exists():
        print(f"[ERROR] corpus.jsonl 不存在: {corpus_path}")
        print("  请先运行: python -m rag_eval.generate_dataset --export-corpus")
        sys.exit(1)

    corpus = load_corpus(corpus_path)
    print(f"  已加载 {len(corpus)} 个文档")

    # ---- Step 3: 构建 BM25 索引 ----
    print("\n[3/5] 构建 BM25 索引...")
    bm25_config = BM25Config(
        k1=rag_config.bm25_k1,
        b=rag_config.bm25_b,
        tokenizer=rag_config.bm25_tokenizer,
        top_k_multiplier=rag_config.bm25_top_k_multiplier,
    )
    bm25 = BM25Retriever(config=bm25_config)

    docs_for_index = []
    for doc_id, doc in corpus.items():
        docs_for_index.append({
            "chunk_id": doc_id,
            "content": doc.get("text", ""),
            "metadata": doc.get("metadata", {}),
        })

    bm25.index(docs_for_index)
    print(f"  BM25 索引: {len(docs_for_index)} 文档")

    # ---- Step 4: 构建 ChromaDB ----
    print("\n[4/5] 构建 ChromaDB 向量库...")
    vs = VectorStore(
        persist_dir=rag_config.chroma_persist_path,
        embedder=embedder,
        bm25_retriever=bm25,
    )

    # 将 corpus 转换为 Document-like 对象列表
    class CorpusDoc:
        def __init__(self, doc_id, text, metadata):
            self.content = text
            self.metadata = metadata
            self.metadata["chunk_id"] = doc_id

    chunk_docs = []
    for doc_id, doc in corpus.items():
        chunk_docs.append(CorpusDoc(
            doc_id=doc_id,
            text=doc.get("text", ""),
            metadata=dict(doc.get("metadata", {})),
        ))

    vs.build_from_chunks(chunk_docs)
    print(f"  ChromaDB: {vs.collection_count()} 文档")

    # ---- Step 5: 验证 ----
    print("\n[5/5] 验证索引...")
    chroma_ok = vs.is_healthy()
    bm25_ok = bm25._built and bm25._total_docs > 0
    print(f"  ChromaDB: {'OK' if chroma_ok else 'FAIL'}")
    print(f"  BM25:     {'OK' if bm25_ok else 'FAIL'} (docs={bm25._total_docs})")

    if chroma_ok and bm25_ok:
        print(f"\n[DONE] 索引构建完成，可以运行评估")
        print(f"  python -m rag_eval.evaluate --config rag_eval/configs/test_config_deepseek.yaml --verbose")
    else:
        print(f"\n[FAIL] 索引构建失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
