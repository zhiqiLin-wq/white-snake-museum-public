"""L1 — 纯函数测试：span 偏移校验器 + 意图解析器 + 段落切分对齐。

不依赖网络 / 不依赖 LLM，全部确定性输入。
覆盖 span_validator.validate_span 的 6 级回退链，以及
annotate_user_request.parse_annotation_intent 的关键词意图解析。
"""

import json
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from agent.server.mcp.tools.span_validator import validate_span
from agent.server.mcp.tools.annotate_user_request import parse_annotation_intent
from agent.server.mcp.tools.annotate_passage import _split_paragraphs

logger = logging.getLogger("annotation_eval.l1")


# 每个 case: (段落, entity, start, end, 期望 status, 期望修正后 (start,end) 或 None)
VALIDATOR_CASES = [
    # Level 1 精确匹配
    ("许宣出涌金门，到西湖边。", "许宣", 0, 2, "valid", (0, 2)),
    # Level 2 窗口搜索：offset 偏了但在 ±50 内
    ("许宣出涌金门，到西湖边。", "涌金门", 5, 8, "corrected", (3, 6)),
    # Level 3 全文搜索：offset 完全错误
    ("许宣出涌金门，到西湖边。", "西湖", 0, 2, "corrected", (8, 10)),
    # Level 4 compact 匹配：entity 带空格，原文有空格
    ("许 宣出涌金门。", "许宣", 0, 2, "corrected", (0, 2)),
    # Level 5 模糊匹配：6 字实体改 1 字，ratio=0.833 >= 0.7，且去空白后原文无此串
    ("过军桥黑珠巷", "过军桥黑球巷", 100, 106, "corrected", (0, 6)),
    # Level 6 丢弃：完全不存在的实体
    ("许宣出涌金门。", "完全不存在的实体", 0, 7, "discarded", None),
]


def run() -> dict:
    logger.info("[L1] 开始 span_validator + 意图解析 纯函数测试")

    # ---- span_validator 回退链 ----
    level_counts = {"valid": 0, "window": 0, "global": 0, "compact": 0, "fuzzy": 0, "discarded": 0}
    recovered = 0
    discarded = 0
    for text, entity, start, end, expect_status, expect_span in VALIDATOR_CASES:
        vr = validate_span(text, entity, start, end)
        method = vr.correction_method or ""
        if expect_span is not None:
            ok = (vr.status == expect_status
                  and vr.start_char == expect_span[0]
                  and vr.end_char == expect_span[1])
        else:
            ok = (vr.status == expect_status)

        if vr.status in ("valid", "corrected"):
            recovered += 1
        else:
            discarded += 1

        if vr.status == "valid":
            level_counts["valid"] += 1
        elif method.startswith("window_search"):
            level_counts["window"] += 1
        elif method == "global_search":
            level_counts["global"] += 1
        elif method.startswith("compact_match"):
            level_counts["compact"] += 1
        elif method.startswith("fuzzy_match"):
            level_counts["fuzzy"] += 1
        else:
            level_counts["discarded"] += 1

        logger.info(
            "[L1][validate_span] entity=%r llm=[%d,%d] -> status=%s method=%r final=[%d,%d] %s",
            entity, start, end, vr.status, method, vr.start_char, vr.end_char,
            "OK" if ok else "FAIL(expected=%s %s)" % (expect_status, expect_span),
        )
        if not ok:
            logger.error("[L1][validate_span] 断言失败: %r", entity)

    total = len(VALIDATOR_CASES)
    recovery_rate = recovered / total if total else 0.0

    # ---- 意图解析器 ----
    intent_cases = [
        ("把人物标蓝", "annotate", ["person"], "#4A90D9"),
        ("标出所有地点", "annotate", ["location"], None),
        ("清除地点", "delete", ["location"], None),
        ("继续标注剩余", "continue", None, None),
        ("把人物和地点都标出来", "annotate", ["person", "location"], None),
    ]
    # 注意: 纯旁批（如"给第3段写批注"）在此函数返回 None（置信度 0.5<0.8，
    # 交由阶段2 LLM 兜底），不在 L1 纯关键词层断言；其真实路径由 supervisor 的
    # _MARGINALIA_KEYWORDS 直检 + annotate_user_request 的纯批注分支覆盖。
    intent_ok = 0
    intent_total = len(intent_cases)
    for instr, expect_action, expect_cats, expect_color in intent_cases:
        r = parse_annotation_intent(instr)
        action_ok = r is not None and r.get("action") == expect_action
        cats_ok = True
        if expect_cats:
            cats_ok = r is not None and set(r.get("categories") or []) == set(expect_cats)
        color_ok = True
        if expect_color:
            color_ok = r is not None and r.get("color_overrides", {}).get("person") == expect_color
        ok = action_ok and cats_ok and color_ok
        intent_ok += 1 if ok else 0
        logger.info(
            "[L1][intent] %r -> %s %s", instr,
            json.dumps(r, ensure_ascii=False) if r else "None",
            "OK" if ok else "FAIL",
        )

    intent_accuracy = intent_ok / intent_total if intent_total else 0.0

    # ---- 段落切分对齐（\r\n / \r / \n\n）----
    split_cases = [
        "a\n\nb\n\nc",
        "a\r\n\r\nb\r\n\r\nc",
        "a\r\rb\r\rc",
    ]
    split_ok = 0
    for raw in split_cases:
        paras = _split_paragraphs(raw)
        ok = paras == ["a", "b", "c"]
        split_ok += 1 if ok else 0
        logger.info("[L1][split] %r -> %s %s", raw, paras, "OK" if ok else "FAIL")

    split_accuracy = split_ok / len(split_cases)

    return {
        "layer": "L1_span_validator",
        "validator": {
            "total": total,
            "recovered": recovered,
            "discarded": discarded,
            "recovery_rate": recovery_rate,
            "fallback_distribution": level_counts,
        },
        "intent_parser": {
            "total": intent_total,
            "accuracy": intent_accuracy,
        },
        "split_paragraphs": {
            "total": len(split_cases),
            "accuracy": split_accuracy,
        },
    }


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(message)s")
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
