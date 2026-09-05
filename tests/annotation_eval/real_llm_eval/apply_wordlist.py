"""用 wordlist.json 替换 annotation_knowledge_base.py 的 known 词表 + event/motif 的 few-shot examples。

- known：按 _ENTITY_KB 类别顺序（person/location/event/term/motif）从后往前替换。
- examples：替换 event 和 motif 两个类别的 examples 为报告里的完整原文原句 few-shot。

运行：
  PYTHONIOENCODING=utf-8 python -m tests.annotation_eval.real_llm_eval.apply_wordlist
"""

import json
import re
from pathlib import Path

KB = Path("agent/server/mcp/tools/annotation_knowledge_base.py")
WL = Path("tests/annotation_eval/real_llm_eval/data/wordlist.json")

ORDER = ["person", "location", "event", "term", "motif"]


def _render_list(words, indent=12):
    pad = " " * indent
    lines = []
    for i in range(0, len(words), 6):
        chunk = words[i:i + 6]
        lines.append(pad + ", ".join(f'"{w}"' for w in chunk) + ",")
    return "[\n" + "\n".join(lines) + "\n" + " " * (indent - 4) + "]"


def _render_examples(examples, category, indent=8):
    pad = " " * indent
    pad2 = " " * (indent + 4)
    pad3 = " " * (indent + 8)
    lines = [pad + '"examples": [']
    for ex in examples:
        title = ex["title"]
        text = ex["text"]
        t = json.dumps(text, ensure_ascii=False)
        lines.append(pad2 + f'({t}, [')
        lines.append(pad3 + f'{{"entity": {t}, "category": "{category}", "explanation": {json.dumps(title, ensure_ascii=False)}}},')
        lines.append(pad2 + ']),')
    lines.append(pad + '],')
    return "\n".join(lines)


def _find_balanced(src, open_idx):
    """从 open_idx（指向 '['）找到配对的 ']' 之后的索引。"""
    depth = 0
    for i in range(open_idx, len(src)):
        if src[i] == "[":
            depth += 1
        elif src[i] == "]":
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError("未找到配对括号")


def main():
    wl = json.loads(WL.read_text(encoding="utf-8"))
    src = KB.read_text(encoding="utf-8")

    # ---- 1. 替换 known ----
    matches = list(re.finditer(r'("known":\s*)\[[^\]]*\]', src, re.S))
    assert len(matches) == 5, f"期望 5 个 known 列表，实际 {len(matches)}"
    for cat, m in zip(reversed(ORDER), reversed(matches)):
        new_list = _render_list(wl[cat])
        src = src[:m.start()] + m.group(1) + new_list + src[m.end():]

    # ---- 2. 替换 event / motif 的 examples ----
    # 定位所有 "examples": [，按文件顺序；类别顺序 person/location/event/term/motif
    ex_matches = list(re.finditer(r'"examples":\s*\[', src, re.S))
    assert len(ex_matches) == 5, f"期望 5 个 examples，实际 {len(ex_matches)}"
    # event 是第 3 个（index 2），motif 是第 5 个（index 4）
    for idx, cat in [(2, "event"), (4, "motif")]:
        m = ex_matches[idx]
        end = _find_balanced(src, m.end() - 1)  # m.end()-1 指向 '['
        key = "event_examples" if cat == "event" else "motif_examples"
        new_block = _render_examples(wl[key], cat)
        src = src[:m.start()] + new_block + src[end:]

    KB.write_text(src, encoding="utf-8")
    print("已写回:", KB)


if __name__ == "__main__":
    main()
