"""L3 — 两轮标注流水线测试（Discovery → Precise Resolution → Coverage）。

依赖 mock prompt_registry + mock LLM，确定性、无网络。

覆盖:
- discovery_pass 滑动窗口 → 候选实体
- precise_resolution_pass 规则预匹配 + LLM 补充 + span 校验 + 去重
- CoverageTracker 覆盖率快照

对金标准计算 P/R/F1、自洽率、段落覆盖率。
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from agent.server.mcp.tools.annotation_pipeline import (
    discovery_pass, precise_resolution_pass
)
from agent.server.mcp.tools.coverage_tracker import CoverageTracker

from .ann_common import compute_metrics
from .fixtures import build_paragraph_list, build_golden, paragraph_texts
from .mocks import MockLLM, MockPromptRegistry

logger = logging.getLogger("annotation_eval.l3")


async def _run_pipeline() -> dict:
    paragraphs = build_paragraph_list()
    categories = ["person", "location", "term", "motif"]
    llm = MockLLM()
    registry = MockPromptRegistry()

    discoveries = await discovery_pass(
        paragraphs, categories, llm, registry,
        concurrency=1, window_size=3, overlap=1,
    )
    logger.info("[L3] discovery 返回 %d 个候选实体", len(discoveries))
    for d in discoveries:
        logger.info("[L3][discovery] para=%s cat=%s entity=%r hint=%r",
                    d["paragraph_index"], d["category"], d["entity"], d.get("context_hint", ""))

    tracker = CoverageTracker(len(paragraphs), categories)
    collected = []
    async for r in precise_resolution_pass(
        paragraphs, discoveries, llm, registry,
        concurrency=1, coverage_tracker=tracker,
    ):
        p_idx = r["paragraph_index"]
        for ann in r["annotations"]:
            logger.info(
                "[L3][precise] para=%s cat=%s entity=%r span=[%s,%s] conf=%s source=%s",
                p_idx, ann["category"], ann["entity"],
                ann["start_char"], ann["end_char"],
                ann.get("confidence", ""), ann.get("source", ""),
            )
            collected.append({
                "paragraph_index": p_idx,
                "entity": ann["entity"],
                "category": ann["category"],
                "start_char": ann["start_char"],
                "end_char": ann["end_char"],
                "confidence": ann.get("confidence", ""),
                "source": ann.get("source", ""),
            })

    snapshot = tracker.snapshot()
    logger.info("[L3] coverage snapshot: %s", json.dumps(snapshot, ensure_ascii=False))
    return {"predictions": collected, "snapshot": snapshot,
            "discovery_count": len(discoveries), "llm_calls": len(llm.calls)}


def run() -> dict:
    logger.info("[L3] 开始两轮流水线测试")
    out = asyncio.run(_run_pipeline())
    golden = build_golden()
    metrics = compute_metrics(out["predictions"], golden, paragraph_texts())
    return {
        "layer": "L3_pipeline",
        "metrics": metrics,
        "coverage": out["snapshot"],
        "discovery_count": out["discovery_count"],
        "llm_calls": out["llm_calls"],
    }


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(message)s")
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
