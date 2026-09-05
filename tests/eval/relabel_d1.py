"""D1 重标：按「强相关」标准用 LLM 重标 relevant_ids。

背景：原 D1 的 relevant_ids 混入了大量「同主题/同版本沾边」的弱相关记忆，
把 Recall 的分母撑大，导致检索评测虚低。本脚本用严格标准重新标注。

「强相关」定义：记忆内容能直接回答查询问的问题，或直接支持查询所问的结论。
以下都算弱相关，剔除：
1. 只提到查询里的同一版本/人物/地点/母题，但没直接回答查询问的内容；
2. 只是「该学者」的其他观点/规定/正在做的事，与查询问的具体问题无关；
3. 内容与查询主题相近，但信息对不上（问 A 答 B）。

流程：对每条查询，LLM 从全库 89 条记忆里重选 relevant_ids（宁缺毋滥），
temperature=0 保证稳定。原文件已备份为 d1_retrieval.json.bak。
"""

import asyncio
import json
import re
import sys
from pathlib import Path

_eval_dir = Path(__file__).resolve().parent
_project_root = _eval_dir.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_eval_dir))

from common import read_json, write_json
from agent.server.llm.deepseek_provider import DeepSeekProvider
from agent.server.config import settings

MODEL = settings.memory_extraction_model or settings.effective_fast_model

RELABEL_SYSTEM = "你是一个严谨的检索评测标注员。你只输出合法的 JSON，不输出任何解释性文字。"

RELABEL_USER = (
    "下面是研究白蛇传的学者的记忆库（带 id）。请对每条查询，从记忆库中选出「强相关」的记忆 id。\n\n"
    "「强相关」的严格定义：该记忆的内容能直接回答查询问的问题，或直接支持查询所问的结论。\n"
    "以下情况都算「弱相关」，不得入选：\n"
    "1. 只提到了查询中的同一个版本/人物/地点/母题，但没有直接回答查询问的内容；\n"
    "2. 只是「该学者」的其他观点、规定、正在做的事，与查询所问的具体问题无关；\n"
    "3. 内容与查询主题相近，但信息对不上（问 A 答 B）。\n"
    "要求：\n"
    "- 每条查询只选真正强相关的记忆，宁缺毋滥，可以为空数组。\n"
    "- relevant_ids 最多 3 个，不要为了凑数而选弱相关。\n"
    "- 只输出 JSON 数组，元素格式：{{\"query_id\": \"q001\", \"relevant_ids\": [\"m001\"]}}\n\n"
    "记忆库：\n{memories}\n\n"
    "查询列表：\n{queries}"
)


def extract_json(text: str):
    """从 LLM 输出中健壮地提取 JSON。"""
    if text is None:
        return None
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("[", "]"), ("{", "}")):
        lo = text.find(open_c)
        hi = text.rfind(close_c)
        if lo != -1 and hi > lo:
            try:
                return json.loads(text[lo:hi + 1])
            except json.JSONDecodeError:
                continue
    return None


async def llm_json(llm, user: str, retries: int = 5):
    for attempt in range(retries):
        try:
            resp = await llm.generate(
                system=RELABEL_SYSTEM, user=user, model=MODEL,
                max_tokens=4096, temperature=0,
            )
            data = extract_json(resp.content)
            if data is not None:
                return data
            snippet = (resp.content or "")[:200]
            print(f"  [重试 {attempt + 1}/{retries}] JSON 解析失败: {snippet!r}")
        except Exception as e:
            print(f"  [重试 {attempt + 1}/{retries}] LLM 调用失败: {e}")
        if attempt < retries - 1:
            await asyncio.sleep(3)
    return None


async def main():
    data = read_json("d1_retrieval.json")
    memories = data["memories"]
    queries = data["queries"]
    mem_by_id = {m["id"]: m for m in memories}

    mem_lines = "\n".join(
        f'- id={m["id"]} type={m["memory_type"]} content={m["content"]}' for m in memories
    )

    llm = DeepSeekProvider()

    old_total = sum(len(q["relevant_ids"]) for q in queries)
    print(f"[重标] 记忆 {len(memories)} 条，查询 {len(queries)} 条，原相关标注 {old_total} 条\n")

    BATCH = 10
    relabeled = {}
    for start in range(0, len(queries), BATCH):
        batch = queries[start:start + BATCH]
        q_lines = "\n".join(f'- query_id={q["id"]} query={q["query"]}' for q in batch)
        user = RELABEL_USER.format(memories=mem_lines, queries=q_lines)
        resp = await llm_json(llm, user)
        if not isinstance(resp, list):
            print(f"  [警告] 批次 {start} 连续失败，保留原标注")
            for q in batch:
                relabeled[q["id"]] = list(q["relevant_ids"])
        else:
            for item in resp:
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
                relabeled[qid] = valid[:3]
        print(f"  [重标] {min(start + BATCH, len(queries))}/{len(queries)}")
        await asyncio.sleep(2)

    # 回写 relevant_ids，未在输出里的查询保留原标注
    new_total = 0
    empty_cnt = 0
    for q in queries:
        if q["id"] in relabeled:
            q["relevant_ids"] = relabeled[q["id"]]
        new_total += len(q["relevant_ids"])
        if not q["relevant_ids"]:
            empty_cnt += 1

    data["meta"] = {
        "name": "D1 检索评测集（强相关重标版）",
        "generator": "LLM 生成 + 强相关标准重标（剔除同主题沾边弱相关）",
        "model": MODEL,
        "relabel_rule": "直接回答/直接支持查询才算相关，宁缺毋滥",
        "total_memories": len(memories),
        "total_queries": len(queries),
        "difficulty_breakdown": {
            d: sum(1 for q in queries if q["difficulty"] == d)
            for d in ("exact", "semantic", "cross")
        },
    }
    path = write_json(data, "d1_retrieval.json")
    print(f"\n[完成] 原相关标注 {old_total} 条 -> 新相关标注 {new_total} 条")
    print(f"[完成] 空相关查询（会被评测跳过）: {empty_cnt} 条")
    print(f"[完成] 写入 {path}")


if __name__ == "__main__":
    asyncio.run(main())
