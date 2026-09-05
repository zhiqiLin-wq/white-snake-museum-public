"""诊断：模拟 setup_indices 流程（embedder -> BM25 -> embed），定位 bad allocation。"""
import sys
import time
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "agent")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from server.rag.config import rag_config
from server.rag.embedder import Embedder
from server.rag.bm25_retriever import BM25Retriever, BM25Config

print("[1] loading embedder ...", flush=True)
emb = Embedder(rag_config.embedding_model, local_path=None)
emb.start_loading()
while not emb.is_ready:
    time.sleep(1)
print("[1] ready", flush=True)

# 读全部 427 chunk
corpus = []
with open(PROJECT_ROOT / "rag_eval" / "ground_truth" / "corpus.jsonl", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        corpus.append(d)

print(f"[2] loaded {len(corpus)} chunks", flush=True)

# 构建 BM25（模拟 setup_indices [3/5]）
print("[3] building BM25 ...", flush=True)
bm25 = BM25Retriever(config=BM25Config(k1=rag_config.bm25_k1, b=rag_config.bm25_b, tokenizer=rag_config.bm25_tokenizer))
docs = [{"chunk_id": d.get("_id", str(i)), "content": d.get("text", ""), "metadata": d.get("metadata", {})} for i, d in enumerate(corpus)]
bm25.index(docs)
print(f"[3] BM25 built: {bm25._total_docs} docs", flush=True)

# 批量 embed（模拟 build_from_chunks [4/5]）
print("[4] embedding in batches of 32 ...", flush=True)
texts = [d.get("text", "") for d in corpus]
try:
    n = 0
    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        emb.embed(batch)
        n += len(batch)
        if i % 128 == 0:
            print(f"  embedded {n}/{len(texts)}", flush=True)
    print(f"[4] embed OK, total {n}", flush=True)
except Exception as e:
    print(f"[4] FAIL at {n}: {type(e).__name__}: {str(e)[:120]}", flush=True)

print("DONE", flush=True)
