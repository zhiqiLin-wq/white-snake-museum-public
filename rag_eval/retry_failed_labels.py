"""重打 Phase 2 里 parse_failed 的 query_label 和核心 chunk 标签。

只重打「影响 boost 验证」的失败项:
- query_label 为 _error 的 query（否则该 query 整体 boost 失效）
- 核心 chunk（is_core=true）标签为 _error 的（否则核心捞不回）

非核心 chunk 的 parse_failed 不重打（它们 boost=0 本来就是对，重打浪费 token）。

Usage:
    python rag_eval/retry_failed_labels.py
"""
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
for p in [str(PROJECT_ROOT), str(AGENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rag_eval.relabel_v2 import (
    LIT_CHUNK_SYSTEM, RESEARCH_CHUNK_SYSTEM, QUERY_SYSTEM,
    extract, genre_info,
)
from rag_eval.scripts.generate_eval_dataset import PROVIDER_CONFIGS
from server.llm.openai_compatible_provider import GenericOpenAIProvider

RESULTS_DIR = PROJECT_ROOT / "rag_eval" / "results"
CANDIDATES_FILE = RESULTS_DIR / "boost_candidates.json"
LABELS_FILE = RESULTS_DIR / "boost_labels.json"


async def main():
    candidates = json.load(open(CANDIDATES_FILE, encoding="utf-8"))
    labels = json.load(open(LABELS_FILE, encoding="utf-8"))

    cand_by = {(r["dataset"], r["query_id"]): r for r in candidates}
    lab_by = {(r["dataset"], r["query_id"]): r for r in labels}

    # 找需要重打的
    failed_queries = []
    failed_core_chunks = []  # (key, chunk_id, genre, text)
    for r in candidates:
        key = (r["dataset"], r["query_id"])
        lab = lab_by.get(key)
        if not lab:
            continue
        ql = lab["query_label"]
        if not isinstance(ql, dict) or "_error" in ql:
            failed_queries.append(key)
        core_set = set(r["core"])
        for c in r["candidates"]:
            if not c["is_core"]:
                continue
            clab = lab["chunk_labels"].get(c["chunk_id"])
            if not isinstance(clab, dict) or "_error" in clab:
                failed_core_chunks.append((key, c["chunk_id"], c["genre"], c["text"]))

    print(f"重打: {len(failed_queries)} 条 query + {len(failed_core_chunks)} 个核心 chunk", flush=True)

    provider = GenericOpenAIProvider(
        api_key=PROVIDER_CONFIGS["deepseek"]["api_key"],
        base_url=PROVIDER_CONFIGS["deepseek"]["base_url"],
        default_model="deepseek-v4-flash",
    )
    sem = asyncio.Semaphore(6)

    # 重打 query
    for key in failed_queries:
        r = cand_by[key]
        ql = await extract(provider, QUERY_SYSTEM.format(category=r["category"]), r["query_text"], sem)
        lab_by[key]["query_label"] = ql
        ok = "OK" if (isinstance(ql, dict) and "_error" not in ql) else "仍失败"
        print(f"  [query] {key[0]}/{key[1]} -> {ok}", flush=True)

    # 重打核心 chunk
    for key, cid, genre, text in failed_core_chunks:
        if genre == "research_literature":
            clab = await extract(provider, RESEARCH_CHUNK_SYSTEM, text, sem)
        else:
            clab = await extract(provider, LIT_CHUNK_SYSTEM.format(genre_info=genre_info(genre)), text, sem)
        lab_by[key]["chunk_labels"][cid] = clab
        ok = "OK" if (isinstance(clab, dict) and "_error" not in clab) else "仍失败"
        print(f"  [chunk] {key[0]}/{key[1]}/{cid} -> {ok}", flush=True)

    await provider.close()

    # 保持 labels 顺序写回
    out = []
    for r in labels:
        key = (r["dataset"], r["query_id"])
        out.append(lab_by[key])
    LABELS_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OUT] 已更新: {LABELS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
