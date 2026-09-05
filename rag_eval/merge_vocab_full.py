"""全量词表生成：对 427 个 chunk 的标签（chunk_labels_full.json）合并四类词表。

收集 persons/locations/plot_units/topics 原始值，每类 LLM 归并（deepseek-chat 非推理），
合并后的标准名 -> canonical，原始叫法 -> raw，输出 merged_vocab_full.json。
"""
import asyncio
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
for p in [str(PROJECT_ROOT), str(AGENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rag_eval.scripts.generate_eval_dataset import PROVIDER_CONFIGS
from server.llm.openai_compatible_provider import GenericOpenAIProvider

RESULTS = PROJECT_ROOT / "rag_eval" / "results"

MERGE_SYSTEM = """你是白蛇传文献研究专家。下面给你一组从语料中提取到的「{kind}」原始值（去重后），请归并成受控词表。

输出 JSON 数组，每个元素:
{{"canonical": "标准名", "raw": ["原始叫法1", "原始叫法2"]}}

归并原则:
1. 同一对象的多种叫法归并成一个标准名，canonical 选最通用、最简洁的
2. 该标准名对应的所有原始叫法都放进 raw 数组
3. 不确定是否同一对象的就分开（保持细粒度）
4. 太泛、无法定位具体对象的，canonical 标为"待删除"

只输出 JSON 数组，不要其他文字。"""


def _parse_json_array(text):
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


async def merge_one(provider, values, kind, sem):
    items = "\n".join(f"- {v}" for v in sorted(values))
    user = f"待归并的「{kind}」原始值（去重后，共 {len(values)} 个）:\n{items}\n\n请归并，输出 JSON 数组。"
    async with sem:
        # 值太多时分批归并（每批 200 个），避免 LLM 输出截断
        if len(values) > 200:
            batch_size = 200
            sorted_vals = sorted(values)
            all_merged = []
            for i in range(0, len(sorted_vals), batch_size):
                batch = sorted_vals[i:i + batch_size]
                batch_items = "\n".join(f"- {v}" for v in batch)
                batch_user = f"待归并的「{kind}」原始值（本批 {len(batch)} 个）:\n{batch_items}\n\n请归并，输出 JSON 数组。"
                for attempt in range(3):
                    try:
                        resp = await provider.generate(
                            system=MERGE_SYSTEM.format(kind=kind), user=batch_user, model=None,
                            max_tokens=16384, temperature=0.1,
                        )
                    except Exception:
                        if attempt < 2:
                            await asyncio.sleep(2)
                            continue
                        # 降级：原始值直接作 canonical
                        all_merged.extend([{"canonical": v, "raw": [v]} for v in batch])
                        break
                    parsed = _parse_json_array(resp.content)
                    if isinstance(parsed, list):
                        all_merged.extend(parsed)
                        break
                    await asyncio.sleep(2)
                else:
                    all_merged.extend([{"canonical": v, "raw": [v]} for v in batch])
            return kind, all_merged

        for attempt in range(3):
            try:
                resp = await provider.generate(
                    system=MERGE_SYSTEM.format(kind=kind), user=user, model=None,
                    max_tokens=16384, temperature=0.1,
                )
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2)
                    continue
                # 降级：原始值直接作 canonical
                return kind, [{"canonical": v, "raw": [v]} for v in sorted(values)]
            parsed = _parse_json_array(resp.content)
            if isinstance(parsed, list):
                return kind, parsed
            await asyncio.sleep(2)
        # 降级：原始值直接作 canonical
        return kind, [{"canonical": v, "raw": [v]} for v in sorted(values)]


def collect():
    data = json.load(open(RESULTS / "chunk_labels_full.json", encoding="utf-8"))
    labels = data["chunk_labels"]
    persons, locations, plot_units, topics = set(), set(), set(), set()
    for cid, v in labels.items():
        lab = v.get("label") or {}
        if "_error" in lab:
            continue
        g = v["genre"]
        if g == "research_literature":
            topics.update(lab.get("分析主题") or [])
            ent = lab.get("引用实体") or {}
            persons.update(ent.get("persons") or [])
            locations.update(ent.get("locations") or [])
            plot_units.update(lab.get("引用情节") or [])
        else:
            persons.update(lab.get("persons") or [])
            locations.update(lab.get("locations") or [])
            plot_units.update(lab.get("plot_unit") or [])
    return {
        "persons": sorted(persons),
        "locations": sorted(locations),
        "plot_units": sorted(plot_units),
        "topics": sorted(topics),
    }


async def main():
    values = collect()
    for k, v in values.items():
        print(f"[INIT] {k}: {len(v)} 个原始值", flush=True)

    provider = GenericOpenAIProvider(
        api_key=PROVIDER_CONFIGS["deepseek"]["api_key"],
        base_url=PROVIDER_CONFIGS["deepseek"]["base_url"],
        default_model="deepseek-chat",
    )
    sem = asyncio.Semaphore(2)

    results = await asyncio.gather(*[
        merge_one(provider, v, k, sem) for k, v in values.items()
    ])
    await provider.close()

    vocab = {}
    for kind, merged in results:
        vocab[kind] = merged
        n_del = sum(1 for e in merged if e.get("canonical") == "待删除")
        print(f"[DONE] {kind}: {len(merged)} 个词条（待删除 {n_del}）", flush=True)

    out_path = RESULTS / "merged_vocab_full.json"
    out_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OUT] {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
