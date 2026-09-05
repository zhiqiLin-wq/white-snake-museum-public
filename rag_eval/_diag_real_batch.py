"""诊断：批量 embed 真实 chunk，定位 bad allocation 根因。"""
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

print("loading embedder ...", flush=True)
emb = Embedder(rag_config.embedding_model, local_path=None)
emb.start_loading()
while not emb.is_ready:
    time.sleep(1)
print("ready", flush=True)

texts = []
with open(PROJECT_ROOT / "rag_eval" / "ground_truth" / "corpus.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 32:
            break
        texts.append(json.loads(line)["text"])

print(f"embedding {len(texts)} real chunks ...", flush=True)
for n in [1, 4, 8, 16, 32]:
    try:
        r = emb.embed(texts[:n])
        print(f"batch={n:2d}: OK dim={len(r[0])}", flush=True)
    except Exception as e:
        print(f"batch={n:2d}: FAIL {type(e).__name__}: {str(e)[:100]}", flush=True)
        break
print("DONE", flush=True)
