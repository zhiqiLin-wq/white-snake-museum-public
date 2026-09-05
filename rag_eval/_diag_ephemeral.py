"""对照测试：新建临时内存 ChromaDB，判断 query segfault 是索引损坏还是版本 bug。"""
import numpy as np
import chromadb

print("[1] creating ephemeral client ...", flush=True)
client = chromadb.EphemeralClient()
col = client.create_collection("test", metadata={"hnsw:space": "cosine"})
print("[1] created", flush=True)

print("[2] adding 10 x 1024-dim vectors ...", flush=True)
vecs = np.random.rand(10, 1024).astype(np.float32).tolist()
col.add(embeddings=vecs, documents=[f"d{i}" for i in range(10)], ids=[f"i{i}" for i in range(10)])
print("[2] added, count =", col.count(), flush=True)

print("[3] testing query ...", flush=True)
r = col.query(query_embeddings=[vecs[0]], n_results=3)
print("[3] query ok, ids =", r["ids"][0], flush=True)

print("[4] DONE", flush=True)
