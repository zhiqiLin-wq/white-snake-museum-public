"""用规范表约束重新解析 10 条试点 query，对比自由发挥版。

把 label_vocab.json 的规范表嵌入 prompt，要求 LLM 解析 query 时
persons/locations/plot_unit 从规范表选标准名，输出新旧对比。

Usage:
    python rag_eval/relabel_query_vocab.py
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

RESULTS_DIR = PROJECT_ROOT / "rag_eval" / "results"
GT = PROJECT_ROOT / "rag_eval" / "ground_truth"
DATASETS = ["academic_deepseek", "academic_qwen", "daily_deepseek", "daily_qwen"]


def build_vocab_text(vocab: dict) -> str:
    lines = ["【人物标准名表】(persons 从下面选标准名)"]
    for std, aliases in vocab["person_aliases"].items():
        lines.append(f"- {std}" + (f"（别名：{'、'.join(aliases)}）" if aliases else ""))
    lines.append("")
    lines.append("【地点标准名表】(locations 从下面选标准名)")
    for std, aliases in vocab["location_aliases"].items():
        lines.append(f"- {std}" + (f"（别名：{'、'.join(aliases)}）" if aliases else ""))
    lines.append("")
    lines.append("【情节标准名表】(plot_unit 从下面选标准名)")
    for std, aliases in vocab["plot_units"].items():
        lines.append(f"- {std}" + (f"（别名：{'、'.join(aliases)}）" if aliases else ""))
    return "\n".join(lines)


QUERY_SYSTEM_TEMPLATE = """你是白蛇传文献研究专家。解析下面这个问题的结构化要素。

输出 JSON（提取不出的返回空数组或空字符串）:
{{"persons": ["人物标准名", ...], "locations": ["地点标准名", ...], "plot_unit": "情节标准名", "plot_detail": "情节细节", "summary": "问题问的是什么事"}}

{vocab_text}

规则:
- persons / locations / plot_unit 三个字段必须从上面的标准名表里选标准名，不要用别名或原文叫法（例如"白素贞"要写成"白娘子"，"许宣"写成"许仙"，"许梦蛟"写成"许士麟"）
- 表里没有的实体才允许用原文，但尽量归到已有标准名
- plot_detail / summary 不受表限制，自由写

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


async def main():
    vocab = json.load(open(RESULTS_DIR / "label_vocab.json", encoding="utf-8"))
    vocab_text = build_vocab_text(vocab)
    system = QUERY_SYSTEM_TEMPLATE.format(vocab_text=vocab_text)

    # 读旧的 query_label（自由发挥版）
    old = json.load(open(RESULTS_DIR / "label_pilot_10q.json", encoding="utf-8"))
    qtext_by_ds = {}
    for ds in DATASETS:
        # query_id 跨数据集重复，必须按 dataset 分开加载，避免覆盖
        qtext_by_ds[ds] = {q["query_id"]: q["text"] for q in load_queries(GT / f"queries_{ds}.jsonl")}

    provider = GenericOpenAIProvider(
        api_key=PROVIDER_CONFIGS["deepseek"]["api_key"],
        base_url=PROVIDER_CONFIGS["deepseek"]["base_url"],
        default_model="deepseek-v4-flash",
    )
    sem = asyncio.Semaphore(5)

    result = []
    for r in old:
        ds, qid = r["dataset"], r["query_id"]
        content = qtext_by_ds[ds].get(qid, r["query_text"])
        last_raw = ""
        parsed = None
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
                parsed = {"_error": f"{type(e).__name__}: {e}"}
                break
            last_raw = resp.content
            parsed = _parse_json(resp.content)
            if isinstance(parsed, dict):
                break
            if attempt < 2:
                await asyncio.sleep(1.0)
        if parsed is None:
            parsed = {"_error": "parse_failed", "_raw": last_raw[:200]}
        result.append({
            "dataset": ds,
            "query_id": qid,
            "query_text": content,
            "old_label": r["query_label"],
            "new_label": parsed,
        })
        print(f"[done] {ds}/{qid}", flush=True)

    await provider.close()

    out = RESULTS_DIR / "relabel_query_vocab.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    for r in result:
        lines.append("=" * 70)
        lines.append(f"[{r['dataset']}] {r['query_id']}")
        lines.append(f"Q: {r['query_text']}")
        lines.append(f"旧(自由发挥) persons={r['old_label'].get('persons')} plot_unit={r['old_label'].get('plot_unit')}")
        lines.append(f"新(规范表)   persons={r['new_label'].get('persons')} plot_unit={r['new_label'].get('plot_unit')}")
        lines.append("")
    readable = RESULTS_DIR / "relabel_query_vocab_readable.txt"
    readable.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. json={out.name}  readable={readable.name}")


if __name__ == "__main__":
    asyncio.run(main())
