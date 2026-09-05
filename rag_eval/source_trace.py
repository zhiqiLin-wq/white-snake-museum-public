"""源 chunk 溯源：对比 rel=2 核心 vs 生成时的 source_chunk_ids。

纯解析 description + 对比 qrels，零 LLM 判断。
把 rel=2 核心客观分成:
  A 类: group 成员（生成 query 时就参与构造，rel 来自 source_relevance 或默认 2）
  B 类: 非 group 成员（生成后、候选池批量标注阶段补标成 rel=2 的）

只回答「这个 rel=2 是怎么来的」，不判定「它是否该是 rel=2」。
判定真伪需要人工看 B 类原文，本脚本只负责客观定位 + 提供 B 类明细供抽检。

Usage:
    python rag_eval/source_trace.py
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "agent")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rag_eval.loader import load_queries, load_qrels

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
    lines = []
    total_core = 0
    total_a = 0
    total_b = 0
    total_no_desc = 0
    b_details = []

    for ds in DATASETS:
        queries = load_queries(GT / f"queries_{ds}.jsonl")
        qrels = load_qrels(GT / f"qrels_{ds}_augmented.tsv")

        ds_core = 0
        ds_a = 0
        ds_b = 0
        ds_no_desc = 0

        for q in queries:
            qid = q.get("query_id", "")
            desc = q.get("description", "")
            source_ids = parse_source_chunks(desc)

            core_ids = {c for c, r in qrels.get(qid, {}).items() if r == 2}
            if not core_ids:
                continue
            ds_core += len(core_ids)

            if not source_ids:
                ds_no_desc += len(core_ids)
                continue

            for cid in core_ids:
                if cid in source_ids:
                    ds_a += 1
                else:
                    ds_b += 1
                    b_details.append({
                        "dataset": ds,
                        "query_id": qid,
                        "chunk_id": cid,
                        "query_text": q.get("text", ""),
                    })

        lines.append("=" * 70)
        lines.append(f"[{ds}]")
        lines.append(f"  rel=2 核心总数: {ds_core}")
        lines.append(f"  A 类(group成员): {ds_a}  ({ds_a/ds_core:.1%})" if ds_core else "  A 类: 0")
        lines.append(f"  B 类(批量标注补): {ds_b}  ({ds_b/ds_core:.1%})" if ds_core else "  B 类: 0")
        lines.append(f"  无 description 无法溯源: {ds_no_desc}")

        total_core += ds_core
        total_a += ds_a
        total_b += ds_b
        total_no_desc += ds_no_desc

    lines.append("")
    lines.append("=" * 70)
    lines.append("汇总:")
    lines.append(f"  rel=2 核心总数: {total_core}")
    lines.append(f"  A 类(group成员): {total_a}  ({total_a/total_core:.1%})" if total_core else "")
    lines.append(f"  B 类(批量标注补): {total_b}  ({total_b/total_core:.1%})" if total_core else "")
    lines.append(f"  无法溯源: {total_no_desc}")

    out = RESULTS_DIR / "source_trace.txt"
    out.write_text("\n".join(lines), encoding="utf-8")

    # B 类明细（供人工抽检）
    detail_lines = ["B 类（批量标注补的 rel=2）明细，供人工抽检原文:", ""]
    for d in b_details:
        detail_lines.append(f"[{d['dataset']}] {d['query_id']}  {d['chunk_id']}")
        detail_lines.append(f"    query: {d['query_text']}")
    detail_out = RESULTS_DIR / "source_trace_B_details.txt"
    detail_out.write_text("\n".join(detail_lines), encoding="utf-8")

    print(f"done -> {out.name} / {detail_out.name}")


if __name__ == "__main__":
    main()
