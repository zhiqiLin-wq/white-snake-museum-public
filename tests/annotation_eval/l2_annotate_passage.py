"""L2 — annotate_passage 全流程测试（mock node_client + mock LLM）。

覆盖:
- 规则预扫描（person/location 已知实体）
- 行式 tuple 解析 + 代码定位
- gleaning 查漏二遍（mock 返回重复实体，靠去重吸收）
- event/motif 默认旁注（概括词去重）
- batch_stats 质量指标

用两路 mock 跑：
1. clean：正常路径，期望 F1=1.0 且 motif 产 1 条旁注。
2. corrupt（历史遗留 corrupt_offset_delta 参数）：坐标已由代码定位、LLM 不返回坐标，
   结果应与 clean 完全一致，证明坐标隔离。

产出: 预测标注列表 + 旁注 + 指标 dict。
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from agent.server.mcp.tools.annotate_passage import handler as annotate_passage_handler

from .ann_common import compute_metrics, map_handler_annotations
from .fixtures import build_chapter, build_golden, paragraph_texts, chapter_content
from .mocks import MockLLM, MockNodeClient

logger = logging.getLogger("annotation_eval.l2")


async def _run_once(corrupt: bool) -> dict:
    llm = MockLLM(corrupt_offset_delta=(2 if corrupt else 0))
    node = MockNodeClient(build_chapter())

    result = await annotate_passage_handler(
        chapter_number=1,
        paragraph_range=[0, len(build_golden()) and 5],
        categories=["person", "location", "term", "motif"],
        node_client=node,
        llm=llm,
    )

    if "error" in result:
        logger.error("[L2] handler 返回 error: %s", result["error"])
        return {"error": result["error"]}

    predictions = map_handler_annotations(result, chapter_content())
    golden = build_golden()
    metrics = compute_metrics(predictions, golden, paragraph_texts())

    batch_stats = result.get("batch_stats", {})
    marginalia = result.get("marginalia", [])

    conf_dist = {}
    for p in predictions:
        c = p.get("confidence", "unknown")
        conf_dist[c] = conf_dist.get(c, 0) + 1

    for p in predictions:
        logger.info(
            "[L2][pred] ch=1 para=%s cat=%s entity=%r span=[%s,%s] conf=%s source=%s",
            p["paragraph_index"], p["category"], p["entity"],
            p["start_char"], p["end_char"], p["confidence"], p["source"],
        )
    for m in marginalia:
        logger.info("[L2][marg] para=%s anchor=%s content=%r",
                    m["paragraph_index"], m["anchor_char_offset"], m["content"])

    return {
        "predictions": predictions,
        "metrics": metrics,
        "batch_stats": batch_stats,
        "marginalia": marginalia,
        "confidence_distribution": conf_dist,
        "llm_calls": len(llm.calls),
    }


def run() -> dict:
    logger.info("[L2] 开始 annotate_passage 全流程测试")

    clean = asyncio.run(_run_once(corrupt=False))
    corrupt = asyncio.run(_run_once(corrupt=True))

    # 断言 1：clean 路径 F1 必须 1.0
    clean_metrics = clean.get("metrics") or {}
    clean_f1 = (clean_metrics.get("delta", {}) or {}).get("delta_0", {}).get("f1", 0.0)
    assert clean_f1 == 1.0, f"clean 标注 F1 应为 1.0，实际 {clean_f1}"

    # 断言 2：event/motif 默认旁注生成正确（golden 里 1 个 motif=报恩）
    clean_marg = clean.get("marginalia", [])
    assert len(clean_marg) == 1, f"应生成 1 条 motif 旁注，实际 {len(clean_marg)}"
    assert "报恩" in clean_marg[0]["content"], f"旁注 content 应含概括词'报恩'，实际 {clean_marg[0]['content']}"

    # 断言 3：坐标隔离——corrupt（历史遗留参数）结果必须与 clean 一致
    corrupt_metrics = corrupt.get("metrics") or {}
    corrupt_f1 = (corrupt_metrics.get("delta", {}) or {}).get("delta_0", {}).get("f1", 0.0)
    assert corrupt_f1 == clean_f1, f"corrupt 与 clean 的 F1 应一致：{corrupt_f1} vs {clean_f1}"

    return {
        "layer": "L2_annotate_passage",
        "clean_offset": {
            "error": clean.get("error"),
            "metrics": clean_metrics,
            "batch_stats": clean.get("batch_stats"),
            "marginalia": clean_marg,
            "confidence_distribution": clean.get("confidence_distribution"),
            "llm_calls": clean.get("llm_calls"),
        },
        "corrupt_offset": {
            "error": corrupt.get("error"),
            "metrics": corrupt_metrics,
            "batch_stats": corrupt.get("batch_stats"),
            "marginalia": corrupt.get("marginalia", []),
            "confidence_distribution": corrupt.get("confidence_distribution"),
            "llm_calls": corrupt.get("llm_calls"),
        },
    }


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(message)s")
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
