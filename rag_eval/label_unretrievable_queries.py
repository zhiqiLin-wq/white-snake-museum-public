"""给 40 unretrievable query 打标签，合并进 query_labels.json。"""
import asyncio
import json
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
from server.rag.query_tagger import tag_query

RESULTS = PROJECT_ROOT / "rag_eval" / "results"
GT = PROJECT_ROOT / "rag_eval" / "ground_truth"


async def main():
    picked = json.load(open(RESULTS / "picked_unretrievable_queries.json", encoding="utf-8"))
    qtext_by_ds = {}
    for ds in sorted({x["dataset"] for x in picked}):
        qs = load_queries(GT / f"queries_{ds}.jsonl")
        qtext_by_ds[ds] = {q["query_id"]: q["text"] for q in qs}

    provider = GenericOpenAIProvider(
        api_key=PROVIDER_CONFIGS["deepseek"]["api_key"],
        base_url=PROVIDER_CONFIGS["deepseek"]["base_url"],
        default_model="deepseek-chat",
    )
    sem = asyncio.Semaphore(4)

    new_labels = []
    for item in picked:
        ds, qid = item["dataset"], item["query_id"]
        qtext = qtext_by_ds[ds].get(qid, "")
        lab = await tag_query(provider, qtext, sem)
        new_labels.append({
            "dataset": ds,
            "query_id": qid,
            "category": item.get("category", ""),
            "query_text": qtext,
            "query_label": lab,
        })
        print(f"  [done] {ds}/{qid} {'error' if '_error' in lab else 'ok'}", flush=True)

    await provider.close()

    # 合并进 query_labels.json
    existing = json.load(open(RESULTS / "query_labels.json", encoding="utf-8"))
    seen = {(r["dataset"], r["query_id"]) for r in existing}
    for r in new_labels:
        if (r["dataset"], r["query_id"]) not in seen:
            existing.append(r)
            seen.add((r["dataset"], r["query_id"]))

    out = RESULTS / "query_labels.json"
    out.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    n_err = sum(1 for r in existing if "_error" in r["query_label"])
    print(f"\n[OUT] {out}（共 {len(existing)} queries, {n_err} errors）", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
