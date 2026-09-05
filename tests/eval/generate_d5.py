"""D5 衰减评测集生成器（纯规则，不调 LLM）。

覆盖 should_tombstone 的全部决策分支：
- System 永不清理
- importance >= floor 豁免保留
- importance < floor 且 days > TTL -> tombstone
- importance < floor 且 days <= TTL -> 保留

expected_label 由 should_tombstone 的规则反推，零歧义，作为金标准。
输出 tests/eval/data/d5_decay.json
"""

import sys
from pathlib import Path

_eval_dir = Path(__file__).resolve().parent
_project_root = _eval_dir.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_eval_dir))

from agent.server.context.memory_scorer import IMPORTANCE_FLOOR, TTL_DAYS
from common import write_json

# 每条组合配一条具体内容，供将来实际写入记忆的评测脚本使用
_CONTENT_BY_TYPE = {
    "Entity": "用户偏好事项{}",
    "Archival": "宋代江南地区事实{}",
    "System": "用户约束：不要执行操作{}",
    "Summary": "对话摘要要点{}",
    "Working": "临时状态标记{}",
}


def _reason(memory_type: str, importance: float, days: int) -> str:
    if memory_type == "System":
        return "System 类型永不清理"
    floor = IMPORTANCE_FLOOR[memory_type]
    ttl = TTL_DAYS[memory_type]
    if importance >= floor:
        return f"importance {importance} >= floor {floor}，豁免保留"
    if days > ttl:
        return f"importance {importance} < floor {floor} 且 days {days} > TTL {ttl}，tombstone"
    return f"importance {importance} < floor {floor} 但 days {days} <= TTL {ttl}，保留"


def _expected_label(memory_type: str, importance: float, days: int) -> str:
    """独立规则生成 expected_label（不调 should_tombstone，避免自证）。

    金标准必须独立于被测函数，否则边界 bug（如 > vs >=）永远测不出。
    """
    if memory_type == "System":
        return "keep"
    floor = IMPORTANCE_FLOOR[memory_type]
    ttl = TTL_DAYS[memory_type]
    if importance >= floor:
        return "keep"
    if days > ttl:
        return "tombstone"
    return "keep"


def generate() -> dict:
    cases = []
    seq = 0
    for memory_type in ["System", "Entity", "Archival", "Summary", "Working"]:
        floor = IMPORTANCE_FLOOR[memory_type]
        ttl = TTL_DAYS[memory_type]

        if memory_type == "System":
            importance_values = [0.5, 0.9, 0.95]
            days_values = [1, 365, 999]
        else:
            importance_values = [0.1, round(floor - 0.1, 2), floor,
                                 round(floor + 0.1, 2), 0.9, 1.0]
            days_values = [1, ttl - 1, ttl, ttl + 1, ttl * 2, ttl * 3]

        importance_values = sorted(set(round(v, 2) for v in importance_values))
        days_values = sorted(set(days_values))

        for importance in importance_values:
            for days in days_values:
                expected = _expected_label(memory_type, importance, days)
                seq += 1
                cases.append({
                    "id": f"d5_{seq:03d}",
                    "memory_type": memory_type,
                    "importance": importance,
                    "days_unaccessed": days,
                    "content": _CONTENT_BY_TYPE[memory_type].format(seq),
                    "expected_label": expected,
                    "reason": _reason(memory_type, importance, days),
                })

    tombstone_count = sum(1 for c in cases if c["expected_label"] == "tombstone")
    keep_count = sum(1 for c in cases if c["expected_label"] == "keep")
    return {
        "meta": {
            "name": "D5 衰减评测集",
            "generator": "纯规则枚举（不调 LLM）",
            "total_cases": len(cases),
            "tombstone_cases": tombstone_count,
            "keep_cases": keep_count,
            "note": "expected_label 由 should_tombstone 规则反推，零歧义",
        },
        "cases": cases,
    }


if __name__ == "__main__":
    data = generate()
    path = write_json(data, "d5_decay.json")
    m = data["meta"]
    print(f"生成 {m['total_cases']} 组（tombstone {m['tombstone_cases']} / keep {m['keep_cases']}）")
    print(f"写入: {path}")
