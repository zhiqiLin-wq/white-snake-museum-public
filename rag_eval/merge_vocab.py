"""归并 plot_unit 和分析主题 成受控词表（先开放提取、后归并的第二步）。

把 relabel_v2 开放提取出的 plot_unit（文学文本+query）和 分析主题（研究文献+query）
喂给 LLM，归并成「标准名 + 别名」词表，输出 JSON。

归并原则（写进 prompt）:
- 同一情节/主题的不同叫法归并成一个标准名（选最通用简洁的）
- 细粒度保持：不确定是否同一情节就分开
- 太泛的（如"白蛇传故事"）标为待删除

Usage:
    python rag_eval/merge_vocab.py
"""
import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
for p in [str(PROJECT_ROOT), str(AGENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rag_eval.scripts.generate_eval_dataset import PROVIDER_CONFIGS
from server.llm.openai_compatible_provider import GenericOpenAIProvider

RESULTS_DIR = PROJECT_ROOT / "rag_eval" / "results"


def collect() -> tuple[Counter, Counter]:
    data = json.load(open(RESULTS_DIR / "relabel_v2.json", encoding="utf-8"))
    plot_units = Counter()
    topics = Counter()
    for r in data:
        ql = r["query_label"]
        if isinstance(ql, dict) and "_error" not in ql:
            if ql.get("plot_unit"):
                plot_units[ql["plot_unit"]] += 1
            if ql.get("分析主题"):
                topics[ql["分析主题"]] += 1
        for c in r["core_chunks"]:
            lab = c["label"]
            if not isinstance(lab, dict) or "_error" in lab:
                continue
            if c["genre"] == "research_literature":
                if lab.get("分析主题"):
                    topics[lab["分析主题"]] += 1
            else:
                if lab.get("plot_unit"):
                    plot_units[lab["plot_unit"]] += 1
    return plot_units, topics


MERGE_SYSTEM = """你是白蛇传文献研究专家。下面给你一组从语料中提取到的「{kind}」原始值（含出现频次），请归并成受控词表。

输出 JSON 数组，每个元素:
{{"标准名": "最通用简洁的名称", "别名": ["其他叫法1", "其他叫法2"]}}

归并原则:
1. 同一情节/主题的不同叫法归并成一个标准名（标准名选最通用、最简洁的）
2. 细粒度保持：如果两个值是"同一情节的不同阶段/侧面"，且 query 可能单独问，就保持分开（不要过度归并）
3. 不确定是否同一情节/主题的就分开
4. 太泛的（如"白蛇传故事"这种无法定位具体情节的）单独列为 {{"标准名": "待删除", "别名": ["白蛇传故事"]}}

只输出 JSON 数组，不要其他文字。"""


def _parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:]) if len(lines) > 1 else text[3:]
    if text.endswith("```"):
        text = text[:-3].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


async def merge_one(provider, values: Counter, kind: str) -> list:
    items = "\n".join(f"- {k}（{v}次）" for k, v in values.most_common())
    user = f"待归并的「{kind}」原始值:\n{items}\n\n请归并，输出 JSON 数组。"
    for attempt in range(3):
        try:
            resp = await provider.generate(
                system=MERGE_SYSTEM.format(kind=kind), user=user, model=None,
                max_tokens=4096, temperature=0.1,
            )
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2)
                continue
            return []
        parsed = _parse_json(resp.content)
        if isinstance(parsed, list):
            return parsed
        await asyncio.sleep(2)
    return []


async def main():
    plot_units, topics = collect()
    provider = GenericOpenAIProvider(
        api_key=PROVIDER_CONFIGS["deepseek"]["api_key"],
        base_url=PROVIDER_CONFIGS["deepseek"]["base_url"],
        default_model="deepseek-v4-flash",
    )
    plot_merged = await merge_one(provider, plot_units, "情节单元")
    topic_merged = await merge_one(provider, topics, "分析主题")
    await provider.close()

    result = {
        "plot_units": plot_merged,
        "topics": topic_merged,
    }
    out = RESULTS_DIR / "merged_vocab.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["=== 情节单元词表（归并后）==="]
    for x in plot_merged:
        lines.append(f"  {x.get('标准名')}  <- 别名: {x.get('别名', [])}")
    lines.append("")
    lines.append("=== 分析主题词表（归并后）===")
    for x in topic_merged:
        lines.append(f"  {x.get('标准名')}  <- 别名: {x.get('别名', [])}")
    readable = RESULTS_DIR / "merged_vocab_readable.txt"
    readable.write_text("\n".join(lines), encoding="utf-8")
    print(f"Done. plot={len(plot_merged)} topic={len(topic_merged)}")


if __name__ == "__main__":
    asyncio.run(main())
