"""D5 衰减评测：验证 should_tombstone 的分类型 importance 保底 + TTL 规则。

对 d5_decay.json 的 153 组，逐组调 should_tombstone 比对 expected_label。
指标：衰减正确率 / 误删率（必须=0）/ 漏删率（必须=0）。
纯规则，零 LLM。
"""

import sys
from pathlib import Path

_eval_dir = Path(__file__).resolve().parent
_project_root = _eval_dir.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_eval_dir))

from common import read_json
from agent.server.context.memory_scorer import MemoryScorer


def main():
    data = read_json("d5_decay.json")
    cases = data["cases"]

    tp = tn = fp = fn = 0
    errors = []
    for c in cases:
        predicted = ("tombstone" if MemoryScorer.should_tombstone(
            c["importance"], c["memory_type"], c["days_unaccessed"]) else "keep")
        expected = c["expected_label"]
        if predicted == expected:
            if expected == "tombstone":
                tp += 1
            else:
                tn += 1
        else:
            if expected == "tombstone":
                fn += 1  # 应删却保留（漏删）
            else:
                fp += 1  # 应留却删（误删）
            errors.append((c["id"], c["memory_type"], c["importance"],
                           c["days_unaccessed"], expected, predicted))

    total = len(cases)
    accuracy = (tp + tn) / total
    keep_total = tn + fp
    false_delete_rate = fp / keep_total if keep_total else 0.0
    tombstone_total = tp + fn
    miss_rate = fn / tombstone_total if tombstone_total else 0.0

    print("===== D5 衰减评测 =====")
    print(f"总组数: {total}")
    print(f"正确: {tp + tn}（tombstone 正确 {tp} / keep 正确 {tn}）")
    print(f"错误: {fp + fn}（误删 {fp} / 漏删 {fn}）")
    print(f"衰减正确率: {accuracy:.4f}")
    print(f"误删率: {false_delete_rate:.4f}（应保留却被删，期望=0）")
    print(f"漏删率: {miss_rate:.4f}（应删却保留，期望=0）")
    if errors:
        print("\n错误样例:")
        for e in errors[:10]:
            print(f"  {e}")


if __name__ == "__main__":
    main()
