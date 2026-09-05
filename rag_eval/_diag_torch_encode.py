"""诊断：torch (sentence_transformers) 直接 encode 是否可行 + 性能/内存。"""
import sys
import time

print("loading bge-large via torch ...", flush=True)
t0 = time.time()
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-large-zh-v1.5", cache_folder="models_cache", device="cpu")
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

print("single encode ...", flush=True)
t0 = time.time()
emb = model.encode(["白娘子的丈夫是谁"], normalize_embeddings=True)
print(f"single: {time.time()-t0:.2f}s dim={len(emb[0])}", flush=True)

print("batch 32 encode ...", flush=True)
texts = ["白蛇传故事白娘子许仙法海雷峰塔西湖断桥" * 20] * 32
t0 = time.time()
emb32 = model.encode(texts, normalize_embeddings=True, batch_size=32)
print(f"batch32: {time.time()-t0:.2f}s dim={len(emb32[0])}", flush=True)

print("DONE", flush=True)
