"""把 golden_draft_review.md 里的人工标注（✗ 删除 + 补充表）解析回 golden_draft.json。

- 候选表「判定」列为 ✗ 的行 → 从 candidates 删除。
- 人工补充表里非空的「实体原文/类别」→ 用代码定位坐标（find 全部出现）补进 candidates，source=manual。
- 定位不到的补充实体 → 打印警告（需人工确认）。

运行：
  PYTHONIOENCODING=utf-8 python -m tests.annotation_eval.real_llm_eval.parse_review
"""

import json
import re
from pathlib import Path

REVIEW = Path(__file__).resolve().parent / "data" / "golden_draft_review.md"
DRAFT = Path(__file__).resolve().parent / "data" / "golden_draft.json"


def _parse_review():
    text = REVIEW.read_text(encoding="utf-8")
    # 按 "## N." 分段
    segs = re.split(r"(?m)^## \d+\. ", text)
    result = {}  # seg_index(1-based) -> {"delete": [entity], "add": [(entity, category)]}
    for seg in segs[1:]:
        lines = seg.splitlines()
        # 段标题行：如 "ch3 · imitation_huaben · para 0（50字）"
        head = lines[0] if lines else ""
        m = re.search(r"para (\d+)", head)
        para = int(m.group(1)) if m else None

        deletes = []
        adds = []
        in_candidates = False
        in_supplement = False
        for line in lines:
            if line.startswith("### 候选实体"):
                in_candidates = True
                in_supplement = False
                continue
            if line.startswith("### 人工补充"):
                in_supplement = True
                in_candidates = False
                continue
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            # 候选表：| # | 实体 | 类别 | start-end | 来源 | 判定 |
            if in_candidates and len(cells) >= 6 and cells[0].isdigit():
                if cells[5] in ("✗", "✘", "×", "x", "X"):
                    deletes.append(cells[1])
            # 补充表：| # | 实体原文 | 类别 |
            elif in_supplement and len(cells) >= 3 and cells[0].isdigit():
                if cells[1]:
                    adds.append((cells[1], cells[2] or "custom"))

        if para is not None:
            result.setdefault(para, {"delete": [], "add": []})
            result[para]["delete"].extend(deletes)
            result[para]["add"].extend(adds)
    return result


def _find_all(text, entity):
    positions = []
    s = 0
    while True:
        p = text.find(entity, s)
        if p < 0:
            break
        positions.append(p)
        s = p + 1
    return positions


def _find_spans_fuzzy(text, entity):
    """精确匹配失败时，去标点/空白后匹配，返回 (start, end) 列表（原文本坐标）。"""
    _PUNCT = set('，。！？；：、,.!?;:\'"“”‘’（）()[]【】《》')
    ec = "".join(ch for ch in entity if ch not in _PUNCT and not ch.isspace())
    if not ec:
        return []
    compact = []
    map_to_orig = []
    for i, ch in enumerate(text):
        if ch not in _PUNCT and not ch.isspace():
            compact.append(ch)
            map_to_orig.append(i)
    tc = "".join(compact)
    spans = []
    s = 0
    while True:
        p = tc.find(ec, s)
        if p < 0:
            break
        start = map_to_orig[p]
        end = map_to_orig[p + len(ec) - 1] + 1
        spans.append((start, end))
        s = p + 1
    return spans


def main():
    review = _parse_review()
    d = json.loads(DRAFT.read_text(encoding="utf-8"))
    g = d["golden_draft"]

    deleted = 0
    added = 0
    unfound = []

    for s in g:
        para = s["paragraph_index"]
        if para not in review:
            continue
        act = review[para]
        # 删除
        if act["delete"]:
            before = len(s["candidates"])
            s["candidates"] = [
                c for c in s["candidates"] if c.get("entity") not in act["delete"]
            ]
            deleted += before - len(s["candidates"])
        # 补充
        for entity, category in act["add"]:
            positions = _find_all(s["text"], entity)
            spans = [(p, p + len(entity)) for p in positions]
            if not spans:
                spans = _find_spans_fuzzy(s["text"], entity)
            if not spans:
                unfound.append((para, entity))
                continue
            for start, end in spans:
                s["candidates"].append({
                    "entity": entity,
                    "category": category,
                    "start_char": start,
                    "end_char": end,
                    "source": "manual",
                })
                added += 1

    DRAFT.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"删除 {deleted} 条，补充 {added} 条")
    if unfound:
        print("定位不到（需人工确认）：")
        for para, entity in unfound:
            print(f"  para {para}: {entity}")


if __name__ == "__main__":
    main()
