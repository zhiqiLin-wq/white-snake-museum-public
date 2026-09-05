"""诊断：用 label_chunks.extract 对 00079 打标，打印返回。"""
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
from rag_eval.label_chunks import LIT_CHUNK_SYSTEM, GENRE_META, extract


async def main():
    corpus = load_corpus(PROJECT_ROOT / "rag_eval" / "ground_truth" / "corpus.jsonl")
    cid = "primary_opera_libretto_四_00079"
    text = corpus[cid]["text"][:900]

    provider = GenericOpenAIProvider(
        api_key=PROVIDER_CONFIGS["deepseek"]["api_key"],
        base_url=PROVIDER_CONFIGS["deepseek"]["base_url"],
        default_model="deepseek-v4-flash",
    )
    sem = asyncio.Semaphore(3)
    system = LIT_CHUNK_SYSTEM.format(genre_info=GENRE_META["opera_libretto"])
    result = await extract(provider, system, text, sem)
    print("result:", json.dumps(result, ensure_ascii=False)[:500], flush=True)
    await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
