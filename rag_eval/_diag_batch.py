"""诊断 onnxruntime 批量 embed 的 batch 大小阈值。"""
import sys
import time
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

# 模拟 chunk 长度（约 200 字，重复填充到接近真实 chunk）
base = "白娘子为救许仙去昆仑山盗取灵芝仙草，途中遇险，幸得南极仙翁相助，才得脱身回到人间。" * 6
texts = [base] * 32

for n in [1, 2, 4, 8, 16, 32]:
    try:
        r = emb.embed(texts[:n])
        print(f"batch={n:2d}: OK dim={len(r[0])}", flush=True)
    except Exception as e:
        print(f"batch={n:2d}: FAIL {type(e).__name__}: {str(e)[:80]}", flush=True)
        break
print("DONE", flush=True)
