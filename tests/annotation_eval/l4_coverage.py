"""L4 — 标注覆盖率（召回率）专项测试。

针对"LLM 找不全"这一真实痛点，验证：
1. 规则引擎能 100% 覆盖已知人名（词表内实体，代码精确匹配，不会漏）。
2. 词表外实体只能靠 LLM，若 LLM 漏标，召回率会真实下降——测试能测出这个缺口。
3. 报告"规则引擎覆盖率 / LLM 覆盖率 / 总召回率"三个数字。

fixture: 单段章节，5 个人名，其中 3 个在规则词表内、2 个在词表外。
  词表内(规则引擎兜底): 许宣 / 白娘子 / 法海
  词表外(仅靠 LLM):   钱塘君 / 聂龙

用两路 mock LLM 跑：
  full_mock    → 返回全部词表外实体，期望总召回率 1.0
  missing_mock → 漏掉"聂龙"，期望总召回率降到 0.8（暴露 LLM 漏标）
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
from agent.server.mcp.tools.annotation_knowledge_base import _ENTITY_KB

from .ann_common import compute_metrics, map_handler_annotations
from .mocks import MockNodeClient

logger = logging.getLogger("annotation_eval.l4")

# 单段章节，包含 5 个人名（3 词表内 + 2 词表外）
COVERAGE_PARAGRAPH = "许宣与钱塘君同游西湖，白娘子与法海相遇，聂龙亦在。"
COVERAGE_CONTENT = COVERAGE_PARAGRAPH

# 金标准人名（按原文出现顺序）
GOLDEN_PERSONS = ["许宣", "钱塘君", "白娘子", "法海", "聂龙"]

# 词表外实体（规则引擎覆盖不到，只能靠 LLM）
UNKNOWN_PERSONS = ["钱塘君", "聂龙"]


def _find_all(text: str, entity: str) -> list[int]:
    positions = []
    search_start = 0
    while True:
        pos = text.find(entity, search_start)
        if pos < 0:
            break
        positions.append(pos)
        search_start = pos + 1
    return positions


def _build_golden() -> list[dict]:
    golden = []
    for entity in GOLDEN_PERSONS:
        for start in _find_all(COVERAGE_PARAGRAPH, entity):
            golden.append({
                "paragraph_index": 0,
                "entity": entity,
                "category": "person",
                "start_char": start,
                "end_char": start + len(entity),
            })
    return golden


def _known_person_names() -> set[str]:
    return set(_ENTITY_KB.get("person", {}).get("known", []))


class SubsetMockLLM:
    """只返回指定实体列表的 mock LLM（模拟 LLM 找到/漏掉词表外实体）。"""

    def __init__(self, entities: list[str]):
        self.entities = entities
        self.calls = []

    async def generate(self, system: str, user: str, model: str = "",
                       max_tokens: int = 2048, temperature: float = 0.1,
                       thinking_disabled: bool = False):
        self.calls.append({"system_head": system[:40], "user_head": user[:40]})
        annotations = [
            {"entity": e, "category": "person", "explanation": "mock"}
            for e in self.entities
        ]
        from .mocks import MockLLMResponse
        return MockLLMResponse(json.dumps({"annotations": annotations}, ensure_ascii=False))


async def _run(mock_entities: list[str]) -> dict:
    llm = SubsetMockLLM(mock_entities)
    node = MockNodeClient({
        "chapterNumber": 1,
        "title": "覆盖率样例章",
        "content": COVERAGE_CONTENT,
    })

    result = await annotate_passage_handler(
        chapter_number=1,
        paragraph_range=[0, 0],
        categories=["person"],
        node_client=node,
        llm=llm,
    )
    if "error" in result:
        return {"error": result["error"]}

    predictions = map_handler_annotations(result, COVERAGE_CONTENT)
    golden = _build_golden()
    metrics = compute_metrics(predictions, golden, {0: COVERAGE_PARAGRAPH})

    # 规则引擎与 LLM 分别命中了多少个金标准人名
    known = _known_person_names()
    rule_hit = sum(1 for p in predictions
                   if p.get("source") == "rule" and p.get("category") == "person")
    llm_hit = sum(1 for p in predictions
                  if p.get("source") == "agent" and p.get("category") == "person")

    for p in predictions:
        logger.info(
            "[L4][pred] entity=%r source=%s span=[%s,%s]",
            p["entity"], p["source"], p["start_char"], p["end_char"],
        )

    return {
        "predictions": predictions,
        "metrics": metrics,
        "rule_hit": rule_hit,
        "llm_hit": llm_hit,
        "total_golden": len(golden),
        "llm_calls": len(llm.calls),
    }


def run() -> dict:
    logger.info("[L4] 开始标注覆盖率专项测试")

    full = asyncio.run(_run(UNKNOWN_PERSONS))
    missing = asyncio.run(_run(["钱塘君"]))  # 故意漏掉"聂龙"

    full_metrics = full.get("metrics") or {}
    missing_metrics = missing.get("metrics") or {}
    full_recall = (full_metrics.get("delta", {}) or {}).get("delta_0", {}).get("recall", 0.0)
    missing_recall = (missing_metrics.get("delta", {}) or {}).get("delta_0", {}).get("recall", 0.0)

    # 断言：全量 mock 应召回 1.0；漏标 mock 应 < 1.0（证明测试能测出漏标）
    assert full_recall == 1.0, f"全量 mock 召回率应为 1.0，实际 {full_recall}"
    assert missing_recall < 1.0, f"漏标 mock 召回率应下降，实际 {missing_recall}"

    return {
        "layer": "L4_coverage",
        "fixture": {
            "paragraph": COVERAGE_PARAGRAPH,
            "total_golden_persons": len(_build_golden()),
            "known_persons": sorted(_known_person_names() & set(GOLDEN_PERSONS)),
            "unknown_persons": UNKNOWN_PERSONS,
        },
        "full_mock": {
            "recall": full_recall,
            "metrics": full_metrics,
            "rule_hit": full.get("rule_hit"),
            "llm_hit": full.get("llm_hit"),
            "llm_calls": full.get("llm_calls"),
        },
        "missing_mock": {
            "recall": missing_recall,
            "metrics": missing_metrics,
            "rule_hit": missing.get("rule_hit"),
            "llm_hit": missing.get("llm_hit"),
            "llm_calls": missing.get("llm_calls"),
        },
    }


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(message)s")
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
