"""打标全量有 rel=2 核心的 query（复用已有标签，只打剩下的）。"""
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
for p in [str(PROJECT_ROOT), str(AGENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rag_eval.loader import load_queries, load_qrels
from rag_eval.scripts.generate_eval_dataset import PROVIDER_CONFIGS
from server.llm.openai_compatible_provider import GenericOpenAIProvider
from server.rag.query_tagger import tag_query

RESULTS = PROJECT_ROOT / "rag_eval" / "results"
GT = PROJECT_ROOT / "rag_eval" / "ground_truth"
DATASETS = ["academic_deepseek", "academic_qwen", "daily_deepseek", "daily_qwen"]


async def main():
    # 现有标签（100 条）
    existing = json.load(open(RESULTS / "query_labels.json", encoding="utf-8"))
    seen = {(r["dataset"], r["query_id"]) for r in existing}

    provider = GenericOpenAIProvider(
        api_key=PROVIDER_CONFIGS["deepseek"]["api_key"],
        base_url=PROVIDER_CONFIGS["deepseek"]["base_url"],
        default_model="deepseek-chat",
    )
    sem = asyncio.Semaphore(6)

    total_new = 0
    out = list(existing)
    for ds in DATASETS:
        qs = load_queries(GT / f"queries_{ds}.jsonl")
        qrels = load_qrels(GT / f"qrels_{ds}_src_only.tsv")
        for q in qs:
            qid = q["query_id"]
            rels = qrels.get(qid, {})
            if not any(r == 2 for r in rels.values()):
                continue
            if (ds, qid) in seen:
                continue
            lab = await tag_query(provider, q["text"], sem)
            out.append({
                "dataset": ds, "query_id": qid,
                "category": q.get("category", ""),
                "query_text": q["text"],
                "query_label": lab,
            })
            total_new += 1
            if total_new % 100 == 0:
                print(f"  [progress] 新增 {total_new}", flush=True)

    await provider.close()

    out_path = RESULTS / "query_labels_full.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    n_err = sum(1 for r in out if "_error" in r["query_label"])
    print(f"\n[OUT] {out_path}（共 {len(out)} queries, 新增 {total_new}, {n_err} errors）", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
