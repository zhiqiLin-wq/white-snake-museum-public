"""诊断：persons 合并（280 个原始值）在 max_tokens=8192 下的 LLM 返回。"""
import json
import sys
import httpx
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
for p in [str(PROJECT_ROOT), str(AGENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rag_eval.scripts.generate_eval_dataset import PROVIDER_CONFIGS
from rag_eval.merge_vocab2 import MERGE_SYSTEM, collect

values = collect()["persons"]
items = "\n".join(f"- {v}" for v in values)
user = f"待归并的「persons」原始值（去重后，共 {len(values)} 个）:\n{items}\n\n请归并，输出 JSON 数组。"

cfg = PROVIDER_CONFIGS["deepseek"]
resp = httpx.post(
    cfg["base_url"] + "/chat/completions",
    json={
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": MERGE_SYSTEM.format(kind="persons")},
            {"role": "user", "content": user},
        ],
        "max_tokens": 8192,
        "temperature": 0.1,
    },
    headers={"Authorization": f"Bearer {cfg['api_key']}"},
    timeout=180,
)
print("status:", resp.status_code)
data = resp.json()
print("usage:", data.get("usage"))
choice = data.get("choices", [{}])[0]
print("finish_reason:", choice.get("finish_reason"))
msg = choice.get("message", {})
print("content len:", len(msg.get("content") or ""))
print("content 前 200:", repr((msg.get("content") or "")[:200]))
print("reasoning len:", len(msg.get("reasoning_content") or ""))
