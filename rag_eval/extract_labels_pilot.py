"""结构化标签提取试点 — 10 条 unretrievable query 的核心 chunk 打标 + query 解析。

不做全量，只抽 10 条最难(unretrievable，核心在 D/S top-50 都找不到)的 query，
用 LLM 给它们的 rel=2 核心 chunk 提取结构化标签，同时解析 query 的相同字段，
输出结果供人工判断提取质量。

字段（提取不出的返回空）:
    persons     : 人物列表 []
    locations   : 地点列表 []
    plot_unit   : 主干情节 ""
    plot_detail : 情节细节 ""
    summary     : 一句话摘要 ""

Usage:
    python rag_eval/extract_labels_pilot.py
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

CHUNK_TRUNC = 800
N_PICK = 10

CHUNK_SYSTEM = """你是白蛇传文献研究专家。给下面这段文本提取结构化信息。

输出 JSON（提取不出的返回空数组或空字符串）:
{
  "persons": ["人物通用名", ...],
  "locations": ["地点", ...],
  "plot_unit": "主干情节",
  "plot_detail": "情节细节",
  "summary": "一句话白话摘要"
}

字段说明:
- persons: 文本出现的人物，用通用名（如 白娘子/许仙/法海/青儿/许氏/徐乾），无则 []
- locations: 文本提到的地点（如 西湖/雷峰塔/金山寺/净慈寺/镇江/峨眉山），无则 []
- plot_unit: 这段对应的白蛇传主干情节（如 游湖借伞/盗仙草/水漫金山/合钵镇妖/端阳现形/法海点破），无则 ""
- plot_detail: 这段具体在讲的情节细节，一句话，无则 ""
- summary: 这段内容的一句话白话摘要

只输出 JSON，不要输出其他文字。"""

QUERY_SYSTEM = """你是白蛇传文献研究专家。解析下面这个问题的结构化要素。

输出 JSON（提取不出的返回空数组或空字符串）:
{
  "persons": ["人物通用名", ...],
  "locations": ["地点", ...],
  "plot_unit": "主干情节",
  "plot_detail": "情节细节",
  "summary": "问题问的是什么事"
}

字段说明与 chunk 打标一致: persons=人物, locations=地点, plot_unit=主干情节, plot_detail=情节细节, summary=问题一句话改写。

只输出 JSON，不要输出其他文字。"""


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


async def main():
    # 抽 10 条 unretrievable query（分层，每数据集最多 3 条，核心丢失多的优先）
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
    qrels_by_ds = {}
    for ds in DATASETS:
        # query_id 跨数据集重复，必须按 dataset 分开加载，避免覆盖
        qtext_by_ds[ds] = {q["query_id"]: q["text"] for q in load_queries(GT / f"queries_{ds}.jsonl")}
        qrels_by_ds[ds] = load_qrels(GT / f"qrels_{ds}_augmented.tsv")

    provider = GenericOpenAIProvider(
        api_key=PROVIDER_CONFIGS["deepseek"]["api_key"],
        base_url=PROVIDER_CONFIGS["deepseek"]["base_url"],
        default_model="deepseek-v4-flash",
    )
    sem = asyncio.Semaphore(6)

    result = []
    for ds, qid in picked:
        core_ids = [c for c, r in qrels_by_ds[ds].get(qid, {}).items() if r == 2]
        qlabel = await extract(provider, QUERY_SYSTEM, qtext_by_ds[ds].get(qid, ""), sem)
        chunks = []
        for cid in core_ids:
            doc = corpus.get(cid, {})
            txt = doc.get("text", "")[:CHUNK_TRUNC]
            clabel = await extract(provider, CHUNK_SYSTEM, txt, sem)
            chunks.append({
                "chunk_id": cid,
                "genre": doc.get("metadata", {}).get("genre", "?"),
                "text": txt,
                "label": clabel,
            })
        result.append({
            "dataset": ds,
            "query_id": qid,
            "query_text": qtext_by_ds[ds].get(qid, ""),
            "query_label": qlabel,
            "core_chunks": chunks,
        })
        print(f"[done] {ds}/{qid}  {len(chunks)} chunks", flush=True)

    await provider.close()

    out = RESULTS_DIR / "label_pilot_10q.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    readable = []
    for r in result:
        readable.append("=" * 70)
        readable.append(f"[{r['dataset']}] {r['query_id']}")
        readable.append(f"Q: {r['query_text']}")
        readable.append(f"query 解析: {json.dumps(r['query_label'], ensure_ascii=False)}")
        readable.append("")
        for c in r["core_chunks"]:
            readable.append(f"  [{c['chunk_id']} / {c['genre']}]")
            readable.append(f"    文本: {c['text'][:150].replace(chr(10), ' ')}")
            readable.append(f"    标签: {json.dumps(c['label'], ensure_ascii=False)}")
            readable.append("")
    readable_path = RESULTS_DIR / "label_pilot_10q_readable.txt"
    readable_path.write_text("\n".join(readable), encoding="utf-8")
    print(f"\nDone. json={out.name}  readable={readable_path.name}")


if __name__ == "__main__":
    asyncio.run(main())
