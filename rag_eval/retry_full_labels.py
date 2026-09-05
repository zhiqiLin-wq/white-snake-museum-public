"""重试 chunk_labels_full.json 里的 error chunk。"""
import asyncio
import json
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
from rag_eval.label_all_chunks import (
    LIT_CHUNK_SYSTEM, RESEARCH_CHUNK_SYSTEM, GENRE_META, extract, CHUNK_TRUNC,
)

RESULTS = PROJECT_ROOT / "rag_eval" / "results"
GT = PROJECT_ROOT / "rag_eval" / "ground_truth"


async def main():
    data = json.load(open(RESULTS / "chunk_labels_full.json", encoding="utf-8"))
    labels = data["chunk_labels"]
    corpus = load_corpus(GT / "corpus.jsonl")

    retry_ids = [cid for cid, v in labels.items() if "_error" in (v.get("label") or {})]
    print(f"[INIT] 重试 chunk 数: {len(retry_ids)}", flush=True)

    provider = GenericOpenAIProvider(
        api_key=PROVIDER_CONFIGS["deepseek"]["api_key"],
        base_url=PROVIDER_CONFIGS["deepseek"]["base_url"],
        default_model="deepseek-v4-flash",
    )
    sem = asyncio.Semaphore(3)

    async def label_one(cid):
        genre = labels[cid]["genre"]
        text = corpus.get(cid, {}).get("text", "")[:CHUNK_TRUNC]
        if genre == "research_literature":
            system = RESEARCH_CHUNK_SYSTEM
        else:
            system = LIT_CHUNK_SYSTEM.format(genre_info=GENRE_META.get(genre, genre))
        return cid, await extract(provider, system, text, sem)

    results = await asyncio.gather(*[label_one(cid) for cid in retry_ids])
    await provider.close()

    n_ok = 0
    for cid, lab in results:
        if "_error" not in lab:
            labels[cid]["label"] = lab
            n_ok += 1

    out_path = RESULTS / "chunk_labels_full.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    n_err = sum(1 for v in labels.values() if "_error" in (v["label"] or {}))
    print(f"[OUT] 重试成功 {n_ok}/{len(retry_ids)}，剩余 error {n_err}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
