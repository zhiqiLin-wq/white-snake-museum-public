"""标签体系重构版：chunk 双轨打标（文学文本 vs 研究文献）+ query 分类型解析。

重构要点:
1. 版本映射表 GENRE_META: genre -> 朝代/文本名/作者/体裁（映射，零 LLM）
2. 文学文本 chunk: persons/locations 用 canonical+raw 双名结构，plot_unit/plot_detail/summary
3. 研究文献 chunk: 涉及版本/分析主题/引用实体/引用情节/summary（信息形态不同，标签隔离）
4. query: query_type 细分 + need_examples/need_analysis + 对比维度 + 分析主题

Usage:
    python rag_eval/relabel_v2.py
"""
import asyncio
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
for p in [str(PROJECT_ROOT), str(AGENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rag_eval.loader import load_corpus, load_queries, load_qrels
from rag_eval.scripts.generate_eval_dataset import PROVIDER_CONFIGS
from server.llm.openai_compatible_provider import GenericOpenAIProvider

RESULTS_DIR = PROJECT_ROOT / "rag_eval" / "results"
GT = PROJECT_ROOT / "rag_eval" / "ground_truth"
DATASETS = ["academic_deepseek", "academic_qwen", "daily_deepseek", "daily_qwen"]

CHUNK_TRUNC = 900
N_PICK = 10

GENRE_META = {
    "classical_chinese": {"dynasty": "唐", "text": "李黄", "author": "佚名", "name": "文言小说"},
    "huaben": {"dynasty": "宋", "text": "西湖三塔记", "author": "佚名", "name": "话本"},
    "imitation_huaben": {"dynasty": "明", "text": "白娘子永镇雷峰塔", "author": "冯梦龙", "name": "拟话本"},
    "opera_libretto": {"dynasty": "清", "text": "雷峰塔传奇", "author": "方成培", "name": "戏曲"},
    "chapter_vernacular": {"dynasty": "清", "text": "雷峰塔传奇", "author": "玉山堂主人", "name": "白话神魔小说"},
    "tanci": {"dynasty": "清", "text": "义妖传", "author": "陈遇乾", "name": "苏州弹词"},
    "script": {"dynasty": "现当代", "text": "白蛇传", "author": "田汉", "name": "京剧"},
    "research_literature": {"dynasty": None, "text": "研究文献", "author": "研究团队", "name": "学术"},
}

# 文学文本 chunk 打标 prompt
LIT_CHUNK_SYSTEM = """你是白蛇传文献研究专家。给下面这段文学文本（{genre_info}）提取结构化信息。

输出 JSON（提取不出的返回空）:
{{"persons": [{{"canonical": "通用名", "raw": "原文名"}}], "locations": [{{"canonical": "通用名", "raw": "原文名"}}], "plot_unit": "主干情节名", "plot_detail": "细情节描述", "summary": "一句话摘要"}}

字段说明:
- persons: 文本出现的人物。canonical 用通用名（白娘子/许仙/法海/青儿/许氏/李公甫等），raw 用这段原文实际写的称呼（如"汉文""白氏""许宣""小青"）。同一人只记一条。
- locations: 同理，canonical 用通用名（杭州/苏州/金山寺/雷峰塔/峨眉山等），raw 用原文称呼（如"临安""钱塘"）。
- plot_unit: 这段对应的白蛇传主干情节（如 游湖借伞/盗库银/盗仙草/水漫金山/合钵镇妖/祭塔/端阳现形/法海点破），无则 ""
- plot_detail: 这段具体在讲的情节细节，一句话，无则 ""
- summary: 这段内容的一句话白话摘要

只输出 JSON，不要其他文字。"""

# 研究文献 chunk 打标 prompt
RESEARCH_CHUNK_SYSTEM = """你是白蛇传文献研究专家。给下面这段研究文献（学术论文段落）提取结构化信息。

输出 JSON（提取不出的返回空）:
{{"涉及版本": ["genre代码"], "分析主题": "主题", "引用实体": {{"persons": [], "locations": []}}, "引用情节": ["情节名"], "summary": "一句话摘要"}}

字段说明:
- 涉及版本: 这段分析/讨论涉及的白蛇传版本，用 genre 代码（classical_chinese=文言小说《李黄》/ huaben=话本《西湖三塔记》/ imitation_huaben=拟话本冯梦龙《白娘子永镇雷峰塔》/ opera_libretto=戏曲方成培《雷峰塔传奇》/ chapter_vernacular=白话神魔小说玉山堂主人《雷峰塔传奇》/ tanci=苏州弹词《义妖传》/ script=京剧田汉《白蛇传》），无则 []
- 分析主题: 这段在分析什么（如 称谓演变/法海形象流变/文学地图/景观互构/版本谱系/时空分布），开放提取，无则 ""
- 引用实体: 这段学术转述提到的人物/地点（不是情节叙述），无则空
- 引用情节: 这段讨论到的白蛇传情节（用主干情节名如 盗仙草/水漫金山/合钵镇妖），无则 []
- summary: 一句话摘要

只输出 JSON，不要其他文字。"""

# query 分类型解析 prompt
QUERY_SYSTEM = """你是白蛇传文献研究专家。解析下面这个问题的结构化要素。问题 category 是「{category}」。

输出 JSON（提取不出的返回空）:
{{"query_type": "类型", "need_examples": true, "need_analysis": false, "persons": [{{"canonical": "通用名", "raw": "原文名"}}], "locations": [{{"canonical": "通用名", "raw": "原文名"}}], "plot_unit": "主干情节名", "plot_detail": "细情节描述", "分析主题": "主题", "对比维度": {{"维度名": "具体值"}}, "summary": "一句话"}}

字段说明:
- query_type: 在 category 基础上细分，从 情节题/人物题/地点题/comparison题/研究分析题/其他 中选
- need_examples: 是否需要文学文本的情节/人物/地点例子来支撑答案（true/false）
- need_analysis: 是否需要研究文献的分析结论来支撑答案（true/false）
- 对比维度: 仅 comparison 题填，用对象列出对比的维度和具体值，如 {{"朝代": "清", "体裁": "话本", "作者": "冯梦龙", "称谓": ["汉文", "许宣"]}}，无具体值的维度省略
- 分析主题: 仅研究分析题或 comparison 题填（如 称谓演变/法海形象/景观互构）
- persons/locations 的 canonical 用通用名、raw 用问题原文里的称呼，其余同 chunk 打标

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
                    max_tokens=2048, temperature=0.1,
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


def genre_info(g: str) -> str:
    m = GENRE_META.get(g, {})
    return f"{m.get('dynasty','')}{m.get('name','')}《{m.get('text','')}》作者{m.get('author','')}"


async def main():
    rank = json.load(open(RESULTS_DIR / "recall_rank_attribution.json", encoding="utf-8"))
    by_query = defaultdict(list)
    for x in rank["detail"]:
        if x["category"] == "unretrievable":
            by_query[(x["dataset"], x["query_id"])].append(x)
    picked = []
    per_ds = defaultdict(int)
    for (ds, qid), chunks in sorted(by_query.items(), key=lambda kv: -len(kv[1])):
        if per_ds[ds] >= 3:
            continue
        picked.append((ds, qid))
        per_ds[ds] += 1
        if len(picked) >= N_PICK:
            break

    corpus = load_corpus(GT / "corpus.jsonl")
    qtext_by_ds = {}
    qcat_by_ds = {}
    qrels_by_ds = {}
    for ds in DATASETS:
        qs = load_queries(GT / f"queries_{ds}.jsonl")
        qtext_by_ds[ds] = {q["query_id"]: q["text"] for q in qs}
        qcat_by_ds[ds] = {q["query_id"]: q.get("category", "") for q in qs}
        qrels_by_ds[ds] = load_qrels(GT / f"qrels_{ds}_augmented.tsv")

    provider = GenericOpenAIProvider(
        api_key=PROVIDER_CONFIGS["deepseek"]["api_key"],
        base_url=PROVIDER_CONFIGS["deepseek"]["base_url"],
        default_model="deepseek-v4-flash",
    )
    sem = asyncio.Semaphore(6)

    result = []
    for ds, qid in picked:
        qtext = qtext_by_ds[ds].get(qid, "")
        qcat = qcat_by_ds[ds].get(qid, "")
        qlabel = await extract(provider, QUERY_SYSTEM.format(category=qcat), qtext, sem)

        core_ids = [c for c, r in qrels_by_ds[ds].get(qid, {}).items() if r == 2]
        chunks = []
        for cid in core_ids:
            doc = corpus.get(cid, {})
            g = doc.get("metadata", {}).get("genre", "?")
            txt = doc.get("text", "")[:CHUNK_TRUNC]
            if g == "research_literature":
                clabel = await extract(provider, RESEARCH_CHUNK_SYSTEM, txt, sem)
            else:
                clabel = await extract(provider, LIT_CHUNK_SYSTEM.format(genre_info=genre_info(g)), txt, sem)
            chunks.append({
                "chunk_id": cid,
                "genre": g,
                "text": txt,
                "label": clabel,
            })
        result.append({
            "dataset": ds,
            "query_id": qid,
            "category": qcat,
            "query_text": qtext,
            "query_label": qlabel,
            "core_chunks": chunks,
        })
        print(f"[done] {ds}/{qid} ({qcat}) {len(chunks)} chunks", flush=True)

    await provider.close()

    out = RESULTS_DIR / "relabel_v2.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    for r in result:
        lines.append("=" * 70)
        lines.append(f"[{r['dataset']}] {r['query_id']}  category={r['category']}")
        lines.append(f"Q: {r['query_text']}")
        lines.append(f"query: {json.dumps(r['query_label'], ensure_ascii=False)}")
        lines.append("")
        for c in r["core_chunks"]:
            lines.append(f"  [{c['chunk_id']} / {c['genre']}]")
            lines.append(f"    {json.dumps(c['label'], ensure_ascii=False)}")
            lines.append("")
    readable = RESULTS_DIR / "relabel_v2_readable.txt"
    readable.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. json={out.name} readable={readable.name}")


if __name__ == "__main__":
    asyncio.run(main())
