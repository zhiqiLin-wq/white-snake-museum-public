"""最小诊断：分离 ONNX embed 与 ChromaDB query，定位 segfault 位置。"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "agent")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from server.rag.config import rag_config
from server.rag.embedder import Embedder

print("[1] loading embedder ...", flush=True)
emb = Embedder(rag_config.embedding_model, local_path=None)
emb.start_loading()
waited = 0
while not emb.is_ready and waited < 300:
    time.sleep(1)
    waited += 1
if not emb.is_ready:
    print("embedder not ready:", emb.load_error)
    sys.exit(1)
print("[1] embedder ready", flush=True)

print("[2] testing embed_query ...", flush=True)
v = emb.embed_query("白娘子的丈夫是谁")
print("[2] embed ok, dim =", len(v), flush=True)

print("[3] loading vector_store ...", flush=True)
from server.rag.vector_store import VectorStore
vs = VectorStore(rag_config.chroma_persist_path, emb)
print("[3] collection count =", vs.collection_count(), flush=True)

print("[4] testing chroma query ...", flush=True)
r = vs.query(v, top_k=5)
print("[4] query ok, n =", len(r), flush=True)

print("[5] DONE", flush=True)
