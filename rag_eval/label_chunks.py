"""阶段 3：chunk 双轨打标（开放提取原始值）。

对候选集去重后的 chunk，按 genre 分两套 prompt 打标。
persons / locations / plot_unit（及研究文献的分析主题 / 引用实体.persons / 引用实体.locations /
引用情节）输出原始值字符串列表，canonical/raw 留到阶段 4 合并时填。
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

from rag_eval.loader import load_corpus
from rag_eval.scripts.generate_eval_dataset import PROVIDER_CONFIGS
from server.llm.openai_compatible_provider import GenericOpenAIProvider

RESULTS = PROJECT_ROOT / "rag_eval" / "results"
GT = PROJECT_ROOT / "rag_eval" / "ground_truth"
CHUNK_TRUNC = 900

GENRE_META = {
    "classical_chinese": "唐文言小说《李黄》",
    "huaben": "宋话本《西湖三塔记》",
    "imitation_huaben": "明拟话本《白娘子永镇雷峰塔》",
    "opera_libretto": "清戏曲《雷峰塔传奇》",
    "chapter_vernacular": "清白话神魔小说《雷峰塔传奇》",
    "tanci": "清苏州弹词《义妖传》",
    "script": "现当代京剧《白蛇传》",
}

LIT_CHUNK_SYSTEM = """你是白蛇传文献研究专家。给下面这段文学文本（{genre_info}）提取结构化信息。

输出 JSON（提取不出的字段返回空数组或空字符串）:
{{"persons": ["原文人物称呼"], "locations": ["原文地名"], "plot_unit": ["主干情节名"], "plot_detail": "情节细节一句话", "summary": "一句话摘要"}}

字段说明:
- persons: 文本出现的人物，用原文实际写的称呼（如"汉文""白氏""许宣""小青"），同一人只记一条，无则 []
- locations: 文本出现的地名，用原文称呼（如"临安""钱塘""西湖"），无则 []
- plot_unit: 这段对应的白蛇传主干情节名（如 游湖借伞/盗仙草/水漫金山/合钵镇妖/祭塔），无则 []
- plot_detail: 这段具体在讲的情节细节，一句话，无则 ""
- summary: 这段内容的一句话白话摘要

只输出 JSON，不要其他文字。"""

RESEARCH_CHUNK_SYSTEM = """你是白蛇传文献研究专家。给下面这段研究文献提取结构化信息。

输出 JSON（提取不出的字段返回空数组或空字符串）:
{{"涉及版本": ["genre代码"], "分析主题": ["分析主题"], "引用实体": {{"persons": [], "locations": []}}, "引用情节": ["情节名"], "summary": "一句话摘要"}}

字段说明:
- 涉及版本: 这段分析涉及的白蛇传版本，用 genre 代码（classical_chinese=文言小说《李黄》/ huaben=话本《西湖三塔记》/ imitation_huaben=拟话本冯梦龙《白娘子永镇雷峰塔》/ opera_libretto=戏曲方成培《雷峰塔传奇》/ chapter_vernacular=白话神魔小说玉山堂主人《雷峰塔传奇》/ tanci=苏州弹词《义妖传》/ script=京剧田汉《白蛇传》），无则 []
- 分析主题: 这段在分析什么（如 称谓演变/法海形象流变/文学地图/景观互构/版本谱系），开放提取，无则 []
- 引用实体: 这段学术转述提到的人物/地点，persons/locations 用原文称呼，无则空
- 引用情节: 这段讨论到的白蛇传情节（用主干情节名），无则 []
- summary: 一句话摘要

只输出 JSON，不要其他文字。"""


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
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


async def extract(provider, system, content, sem) -> dict:
    async with sem:
        last_raw = ""
        for attempt in range(3):
            try:
                resp = await provider.generate(
                    system=system, user=content, model=None,
                    max_tokens=8192, temperature=0.1,
                )
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                return {"_error": f"{type(e).__name__}: {e}"}
            last_raw = resp.content
            parsed = _parse_json(resp.content)
            if isinstance(parsed, dict):
                return parsed
            if attempt < 2:
                await asyncio.sleep(1.0)
        return {"_error": "parse_failed", "_raw": last_raw[:200]}


async def main():
    data = json.load(open(RESULTS / "retrieved_candidates.json", encoding="utf-8"))
    corpus = load_corpus(GT / "corpus.jsonl")

    chunk_map = {}
    for r in data:
        for c in r["candidates"]:
            cid = c["chunk_id"]
            if cid not in chunk_map:
                chunk_map[cid] = {
                    "genre": c["genre"],
                    "text": corpus.get(cid, {}).get("text", "")[:CHUNK_TRUNC],
                }

    print(f"[INIT] 去重后 chunk 数: {len(chunk_map)}", flush=True)

    provider = GenericOpenAIProvider(
        api_key=PROVIDER_CONFIGS["deepseek"]["api_key"],
        base_url=PROVIDER_CONFIGS["deepseek"]["base_url"],
        default_model="deepseek-v4-flash",
    )
    sem = asyncio.Semaphore(6)

    items = list(chunk_map.items())
    counter = {"done": 0}

    async def label_one(cid, info):
        genre = info["genre"]
        if genre == "research_literature":
            system = RESEARCH_CHUNK_SYSTEM
        else:
            system = LIT_CHUNK_SYSTEM.format(genre_info=GENRE_META.get(genre, genre))
        lab = await extract(provider, system, info["text"], sem)
        counter["done"] += 1
        if counter["done"] % 40 == 0:
            print(f"  [progress] {counter['done']}/{len(items)}", flush=True)
        return cid, genre, lab

    results = await asyncio.gather(*[label_one(cid, info) for cid, info in items])
    await provider.close()

    labels = {}
    for cid, genre, lab in results:
        labels[cid] = {"genre": genre, "label": lab}

    out_path = RESULTS / "chunk_labels.json"
    out_path.write_text(json.dumps({"chunk_labels": labels}, ensure_ascii=False, indent=2), encoding="utf-8")
    n_err = sum(1 for v in labels.values() if "_error" in (v["label"] or {}))
    print(f"\n[OUT] {out_path}（{len(labels)} chunks, {n_err} errors）", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
