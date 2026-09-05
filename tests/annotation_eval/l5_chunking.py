"""L5 — 超长段落块级切分 + 坐标映射测试。

覆盖：
1. _split_paragraph_into_chunks 切块函数（纯函数）：
   - 短段落不切；有标点按句切（落刀句号、overlap 3 句）；无标点按字数切（overlap 150 字）；覆盖完整不丢字符。
2. annotate_passage.handler 块级标注 + 坐标映射（mock LLM）：
   - 超长段落被切成多块（total_batches > 1）
   - 每个标注 span 自洽（段落text[start:end]==entity，坐标映射正确）
   - 重叠区无重复 span（去重生效）

产出：指标 dict，接入 run_eval。
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from agent.server.mcp.tools.annotate_passage import (
    handler as annotate_passage_handler,
    _split_paragraph_into_chunks,
)

from .mocks import MockLLM, MockNodeClient

logger = logging.getLogger("annotation_eval.l5")


def _test_chunk_splitter() -> dict:
    """切块函数纯函数单测。"""
    checks = []

    # 1. 短段落不切
    short = "短段落。" * 10
    chunks = _split_paragraph_into_chunks(short)
    checks.append(("短段落不切", chunks == [(0, len(short))]))

    # 2. 超长有标点：每句 50 字 + 句号，40 句
    sent = "甲" * 49 + "。"
    long_punct = sent * 40  # 40 * 50 = 2000 字
    chunks = _split_paragraph_into_chunks(long_punct)
    # 覆盖完整
    cover_ok = chunks[0][0] == 0 and chunks[-1][1] == len(long_punct)
    # 相邻无 gap（有 overlap：next_start <= prev_end）
    gap_ok = all(chunks[i + 1][0] <= chunks[i][1] for i in range(len(chunks) - 1))
    # 落刀在句号后（非最后一块的 end 处是句号）
    cut_ok = all(
        chunks[i][1] == len(long_punct) or long_punct[chunks[i][1] - 1] == "。"
        for i in range(len(chunks))
    )
    # overlap 约 3 句：每块(除最后)end 到下一块 start 之间约 3 个句号
    overlap_ok = all(
        chunks[i][1] - chunks[i + 1][0] >= 3 * 50 - 1
        for i in range(len(chunks) - 1)
    )
    checks.append(("有标点切成多块", len(chunks) > 1))
    checks.append(("有标点覆盖完整", cover_ok))
    checks.append(("有标点无 gap（有 overlap）", gap_ok))
    checks.append(("有标点落刀句号", cut_ok))
    checks.append(("有标点 overlap≈3句", overlap_ok))

    # 3. 超长无标点：连续无句号
    long_no_punct = "乙" * 2000
    chunks = _split_paragraph_into_chunks(long_no_punct)
    cover_ok = chunks[0][0] == 0 and chunks[-1][1] == len(long_no_punct)
    gap_ok = all(chunks[i + 1][0] <= chunks[i][1] for i in range(len(chunks) - 1))
    # overlap 150 字：下一块 start = 上一块 end - 150
    overlap_ok = all(
        chunks[i][1] - chunks[i + 1][0] == 150
        for i in range(len(chunks) - 1)
    )
    checks.append(("无标点切成多块", len(chunks) > 1))
    checks.append(("无标点覆盖完整", cover_ok))
    checks.append(("无标点无 gap", gap_ok))
    checks.append(("无标点 overlap=150字", overlap_ok))

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        logger.info("[L5][chunk] %s: %s", name, "OK" if ok else "FAIL")

    return {
        "total": len(checks),
        "passed": passed,
        "accuracy": passed / len(checks) if checks else 0.0,
        "details": [{"name": n, "ok": o} for n, o in checks],
    }


async def _test_block_annotation() -> dict:
    """块级标注 + 坐标映射（mock LLM + 超长段落）。"""
    # 超长段落：约 1035 字，含已知实体，句号分隔
    sent_a = "许宣出涌金门，到西湖边。"
    sent_b = "白娘子与青青在船上等候。"
    text = (sent_a + sent_b) * 45  # 23 * 45 ≈ 1035 字

    node = MockNodeClient({"chapterNumber": 1, "title": "块级测试章", "content": text})
    llm = MockLLM()

    result = await annotate_passage_handler(
        chapter_number=1,
        paragraph_range=[0, 0],
        categories=["person", "location"],
        node_client=node,
        llm=llm,
    )
    if "error" in result:
        return {"error": result["error"]}

    annotations = result.get("annotations", [])
    stats = result.get("batch_stats", {})

    # 1. 切块生效
    chunked = stats.get("total_batches", 0) > 1

    # 2. 坐标映射正确：每个 span 自洽
    sc_ok = sum(1 for a in annotations if text[a["start_char"]:a["end_char"]] == a["entity"])
    sc_total = len(annotations)
    self_consistency = sc_ok / sc_total if sc_total else 1.0

    # 3. 无重复 span
    keys = [(a["entity"], a["start_char"], a["end_char"]) for a in annotations]
    no_dup = len(keys) == len(set(keys))

    logger.info(
        "[L5][block] total_batches=%s annotations=%d self_consistency=%.3f no_dup=%s",
        stats.get("total_batches"), len(annotations), self_consistency, no_dup,
    )

    return {
        "total_batches": stats.get("total_batches"),
        "annotations": len(annotations),
        "chunked": chunked,
        "span_self_consistency": self_consistency,
        "no_duplicate_span": no_dup,
        "passed": chunked and self_consistency == 1.0 and no_dup,
    }


def run() -> dict:
    logger.info("[L5] 开始块级切分 + 坐标映射测试")
    chunker = _test_chunk_splitter()
    block = asyncio.run(_test_block_annotation())
    return {
        "layer": "L5_chunking",
        "chunk_splitter": chunker,
        "block_annotation": block,
    }


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(message)s")
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
