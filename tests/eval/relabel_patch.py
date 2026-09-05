"""补重标 q001-q010（relabel_d1.py 第一批因 LLM 返回空而失败）。

用 temperature=0.1 + 更多重试，只处理前 10 条查询，结果合并回 d1_retrieval.json。
"""

import asyncio
import sys
from pathlib import Path

_eval_dir = Path(__file__).resolve().parent
_project_root = _eval_dir.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_eval_dir))

from common import read_json, write_json
from agent.server.llm.deepseek_provider import DeepSeekProvider
from agent.server.config import settings
from relabel_d1 import RELABEL_SYSTEM, RELABEL_USER, extract_json

MODEL = settings.memory_extraction_model or settings.effective_fast_model


async def main():
    data = read_json("d1_retrieval.json")
    memories = data["memories"]
    queries = data["queries"]
    mem_by_id = {m["id"]: m for m in memories}

    mem_lines = "\n".join(
        f'- id={m["id"]} type={m["memory_type"]} content={m["content"]}' for m in memories
    )

    target_ids = {f"q{i:03d}" for i in range(1, 11)}
    batch = [q for q in queries if q["id"] in target_ids]
    q_lines = "\n".join(f'- query_id={q["id"]} query={q["query"]}' for q in batch)
    user = RELABEL_USER.format(memories=mem_lines, queries=q_lines)

    llm = DeepSeekProvider()
    parsed = None
    for attempt in range(8):
        resp = await llm.generate(
            system=RELABEL_SYSTEM, user=user, model=MODEL,
            max_tokens=4096, temperature=0.1,
        )
        parsed = extract_json(resp.content)
        if isinstance(parsed, list) and parsed:
            break
        print(f"  [重试 {attempt + 1}/8] 返回: {(resp.content or '')[:120]!r}")
        await asyncio.sleep(5)

    if not isinstance(parsed, list) or not parsed:
        print("补重标失败，未写入")
        return

    applied = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("query_id", ""))
        rel = item.get("relevant_ids", [])
        if not isinstance(rel, list):
            rel = []
        valid = []
        for r in rel:
            r = str(r)
            if r in mem_by_id and r not in valid:
                valid.append(r)
        applied[qid] = valid[:3]

    for q in queries:
        if q["id"] in applied:
            q["relevant_ids"] = applied[q["id"]]

    write_json(data, "d1_retrieval.json")
    for q in batch:
        print(f"  {q['id']}: {q['relevant_ids']}")


if __name__ == "__main__":
    asyncio.run(main())
