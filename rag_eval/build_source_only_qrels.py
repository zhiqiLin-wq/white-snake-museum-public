"""生成「只保留源 chunk 核心」的 qrels：把 B 类 rel=2 降级为 rel=1。

基于源 chunk 溯源（source_trace 的结论），对每个 query:
  rel=2 的 chunk 若不在 description 记录的 source_chunk_ids 里（B 类，批量标注补的），
  降级为 rel=1（保留辅助相关，不再算核心）。

不调 LLM，纯解析 description + 改写 qrels。原 qrels 不动，另存一份 *_src_only.tsv。

Usage:
    python rag_eval/build_source_only_qrels.py
"""
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "agent")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rag_eval.loader import load_queries

RESULTS_DIR = PROJECT_ROOT / "rag_eval" / "results"
GT = PROJECT_ROOT / "rag_eval" / "ground_truth"
DATASETS = ["academic_deepseek", "academic_qwen", "daily_deepseek", "daily_qwen"]

RE_CHUNKS = re.compile(r"chunks\s*\[(.*?)\]")


def parse_source_chunks(description: str) -> set[str]:
    if not description:
        return set()
    m = RE_CHUNKS.search(description)
    if not m:
        return set()
    return set(re.findall(r"'([^']+)'", m.group(1)))


def main():
    total_rel2_before = 0
    total_rel2_after = 0
    total_downgraded = 0

    for ds in DATASETS:
        queries = load_queries(GT / f"queries_{ds}.jsonl")
        source_map = {q["query_id"]: parse_source_chunks(q.get("description", "")) for q in queries}

        src_path = GT / f"qrels_{ds}_augmented.tsv"
        out_path = GT / f"qrels_{ds}_src_only.tsv"

        header = []
        body = []
        rel2_before = 0
        rel2_after = 0
        downgraded = 0

        for line in src_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                header.append("")
                continue
            if line.startswith("#"):
                header.append(line)
                continue
            qid, doc_id, rel = line.split("\t")
            rel = int(rel)
            if rel == 2:
                rel2_before += 1
                if doc_id not in source_map.get(qid, set()):
                    # B 类：非源 chunk，降级为 rel=1
                    rel = 1
                    downgraded += 1
                else:
                    rel2_after += 1
                body.append(f"{qid}\t{doc_id}\t{rel}")
            else:
                body.append(f"{qid}\t{doc_id}\t{rel}")

        out_lines = [
            f"# qrels (source-only) — 从 {src_path.name} 排除 B 类 rel=2",
            "# B 类 = rel=2 但不在 query 生成时的 source_chunk_ids 里（批量标注补的核心），降级为 rel=1",
            "# 格式: query_id\tdoc_id\trelevance",
            "",
        ] + body
        out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

        total_rel2_before += rel2_before
        total_rel2_after += rel2_after
        total_downgraded += downgraded
        print(f"[{ds}] rel=2: {rel2_before} -> {rel2_after} (降级 {downgraded})")

    print(f"\n汇总: rel=2 共 {total_rel2_before} -> {total_rel2_after}，降级 {total_downgraded}")


if __name__ == "__main__":
    main()
