"""诊断：直接调 deepseek API，打印完整响应（含 finish_reason / reasoning_content）。"""
import json
import sys
import httpx
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
for p in [str(PROJECT_ROOT), str(AGENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rag_eval.loader import load_corpus
from rag_eval.scripts.generate_eval_dataset import PROVIDER_CONFIGS
from rag_eval.label_chunks import LIT_CHUNK_SYSTEM, GENRE_META

corpus = load_corpus(Path("rag_eval/ground_truth/corpus.jsonl"))
text = corpus["primary_opera_libretto_四_00079"]["text"][:900]
system = LIT_CHUNK_SYSTEM.format(genre_info=GENRE_META["opera_libretto"])

cfg = PROVIDER_CONFIGS["deepseek"]
resp = httpx.post(
    cfg["base_url"] + "/chat/completions",
    json={
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        "max_tokens": 8192,
        "temperature": 0.1,
    },
    headers={"Authorization": f"Bearer {cfg['api_key']}"},
    timeout=90,
)
print("status:", resp.status_code)
data = resp.json()
print("usage:", data.get("usage"))
choice = data.get("choices", [{}])[0]
print("finish_reason:", choice.get("finish_reason"))
msg = choice.get("message", {})
print("message keys:", list(msg.keys()))
for k, v in msg.items():
    if isinstance(v, str):
        print(f"  {k}: len={len(v)} repr={v[:200]!r}")
    else:
        print(f"  {k}: {v!r}")
