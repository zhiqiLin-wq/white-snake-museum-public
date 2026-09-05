"""阶段 5：query 打标。

用阶段 4 生成的全局词表给 10 个 query 打标。persons/locations/plot_unit/分析主题
为 [{"canonical","raw"}] 双名列表，canonical 从词表 canonical 选、raw 从词表 raw 选。
过滤 canonical=="待删除" 的词条。
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

from rag_eval.loader import load_queries
from rag_eval.scripts.generate_eval_dataset import PROVIDER_CONFIGS
from server.llm.openai_compatible_provider import GenericOpenAIProvider

RESULTS = PROJECT_ROOT / "rag_eval" / "results"
GT = PROJECT_ROOT / "rag_eval" / "ground_truth"

GENRE_HINT = (
    "classical_chinese=唐文言小说佚名《李黄》/ huaben=宋话本佚名《西湖三塔记》/ "
    "imitation_huaben=明拟话本冯梦龙《白娘子永镇雷峰塔》/ opera_libretto=清戏曲方成培《雷峰塔传奇》/ "
    "chapter_vernacular=清白话神魔小说玉山堂主人《雷峰塔传奇》/ tanci=清苏州弹词陈遇乾《义妖传》/ "
    "script=现当代京剧田汉《白蛇传》"
)


def format_vocab(vocab, kind, label):
    """把词表格式化成 prompt 文本，过滤待删除。"""
    entries = [e for e in vocab.get(kind, []) if e.get("canonical") != "待删除"]
    lines = [f"{label}词表:"]
    for e in entries:
        raw = "、".join(e.get("raw", []))
        lines.append(f"- {e['canonical']} <- {raw}")
    return "\n".join(lines)


QUERY_SYSTEM = """你是白蛇传文献研究专家。请解析下面这个问题的结构化要素。

版本 genre 映射: {genre_hint}

下面是从语料中归纳的受控词表:

{vocab_text}

输出 JSON:
{{"query_type": "类型", "need_examples": true, "need_analysis": false, "persons": [{{"canonical": "标准名", "raw": "原文称呼"}}], "locations": [{{"canonical": "标准名", "raw": "原文称呼"}}], "plot_unit": [{{"canonical": "标准名", "raw": "原文称呼"}}], "plot_detail": "细情节描述", "分析主题": [{{"canonical": "标准名", "raw": "原文称呼"}}], "对比维度": {{}}, "summary": "一句话", "genre": "genre代码或空"}}

字段说明:
- query_type: 严格从 情节题/人物题/地点题/comparison题/研究分析题 五类中选，不要用"其他"。
  判断标准:
  - 情节题: 问某个情节的具体内容、细节或文本（含唱词、台词、对白、发生的事情），例如"盗仙草前托付青儿的唱词为何"、"赠银多少两"
  - 人物题: 问某个人物的身份、关系或行为
  - 地点题: 问某个地点
  - comparison题: 跨版本或跨文本的比较、承袭、源流（如"A 与 B 的差异"、"承袭自哪个版本"）
  - 研究分析题: 问学术分析（形象流变、景观互构、主题演变、版本谱系等）
- need_examples: 是否需要文学文本的情节/人物/地点例子支撑答案（true/false）
- need_analysis: 是否需要研究文献的分析结论支撑答案（true/false）
- persons/locations/plot_unit/分析主题: canonical 必须从上面词表的 canonical 里选，raw 从该 canonical 的 raw 里选；若问题原文的称呼不在词表里，raw 就用问题原文里实际出现的称呼
- plot_detail: 问题问的具体情节，一句话，无则 ""
- 对比维度: 仅 comparison 题填。字段名严格用 朝代/文本/体裁/作者/称谓，不要用其他字段名（如"版本""典故"）:
  - 朝代: 唐/宋/明/清/现当代
  - 文本: 李黄/西湖三塔记/白娘子永镇雷峰塔/雷峰塔传奇/义妖传/白蛇传
  - 体裁: 文言小说/话本/拟话本/戏曲/白话神魔小说/苏州弹词/京剧
  - 作者: 佚名/冯梦龙/方成培/玉山堂主人/陈遇乾/田汉
  - 称谓: 具体称呼列表，如 ["汉文","许宣"]
  - 无具体值的维度省略
- 分析主题: 仅研究分析题或 comparison 题填
- summary: 这个问题问的是什么事，一句话
- genre: 问题明确指向的版本 genre 代码（从上面映射里选），不明确则 ""

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


async def extract(provider, system, content, sem):
    async with sem:
        for attempt in range(3):
            try:
                resp = await provider.generate(
                    system=system, user=content, model=None,
                    max_tokens=4096, temperature=0.1,
                )
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                return {"_error": f"{type(e).__name__}: {e}"}
            parsed = _parse_json(resp.content)
            if isinstance(parsed, dict):
                return parsed
            await asyncio.sleep(1.0)
        return {"_error": "parse_failed", "_raw": resp.content[:200]}


async def main():
    vocab = json.load(open(RESULTS / "merged_vocab2.json", encoding="utf-8"))
    vocab_text = "\n\n".join([
        format_vocab(vocab, "persons", "persons(人物)"),
        format_vocab(vocab, "locations", "locations(地点)"),
        format_vocab(vocab, "plot_units", "plot_unit(情节单元)"),
        format_vocab(vocab, "topics", "分析主题"),
    ])
    system = QUERY_SYSTEM.format(genre_hint=GENRE_HINT, vocab_text=vocab_text)

    picked = json.load(open(RESULTS / "picked_sortable_queries.json", encoding="utf-8"))
    qtext_by_ds = {}
    for ds in sorted({x["dataset"] for x in picked}):
        qs = load_queries(GT / f"queries_{ds}.jsonl")
        qtext_by_ds[ds] = {q["query_id"]: q["text"] for q in qs}

    provider = GenericOpenAIProvider(
        api_key=PROVIDER_CONFIGS["deepseek"]["api_key"],
        base_url=PROVIDER_CONFIGS["deepseek"]["base_url"],
        default_model="deepseek-chat",
    )
    sem = asyncio.Semaphore(3)

    results = []
    for item in picked:
        ds, qid = item["dataset"], item["query_id"]
        qtext = qtext_by_ds[ds].get(qid, "")
        user = f"问题是（category={item.get('category','')}）: {qtext}"
        lab = await extract(provider, system, user, sem)
        results.append({
            "dataset": ds,
            "query_id": qid,
            "category": item.get("category", ""),
            "query_text": qtext,
            "query_label": lab,
        })
        print(f"  [done] {ds}/{qid} {'error' if '_error' in lab else 'ok'}", flush=True)

    await provider.close()

    out_path = RESULTS / "query_labels.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    n_err = sum(1 for r in results if "_error" in r["query_label"])
    print(f"\n[OUT] {out_path}（{len(results)} queries, {n_err} errors）", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
