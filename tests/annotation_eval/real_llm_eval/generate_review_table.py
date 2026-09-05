"""把 golden_draft.json 转成易读的 Markdown 审校表，方便逐段人工核对。

每段：原文 + 候选实体表格（含 start/end/来源 + 判定空栏）+ 补充空表。

运行：
  PYTHONIOENCODING=utf-8 python -m tests.annotation_eval.real_llm_eval.generate_review_table
"""

import json
from pathlib import Path

DRAFT = Path(__file__).resolve().parent / "data" / "golden_draft.json"
OUT = Path(__file__).resolve().parent / "data" / "golden_draft_review.md"

CAT_LABEL = {
    "person": "人名", "location": "地点", "term": "术语",
    "event": "事件句子", "motif": "主题句子",
}


def main():
    d = json.loads(DRAFT.read_text(encoding="utf-8"))
    samples = d["golden_draft"]

    lines = []
    lines.append("# 《白蛇传文献选集》标注金标准审校表")
    lines.append("")
    lines.append("> 审校方法：")
    lines.append("> 1. 每段看「原文」，核对下方「候选实体」。")
    lines.append("> 2. **错的**：在「判定」列写 ✗（或直接删该行）；类别/起止错的直接在列里改。")
    lines.append("> 3. **漏掉的**：填进「人工补充」表，只填「实体原文」和「类别」，start/end 由程序定位（不用手数）。")
    lines.append("> 4. 事件/主题的「实体原文」要填原文里完整原句（一字不差）。")
    lines.append("")

    for idx, s in enumerate(samples, 1):
        ch = s["chapter_number"]
        genre = s["genre"]
        pi = s["paragraph_index"]
        text = s["text"]
        candidates = s["candidates"]

        lines.append(f"## {idx}. ch{ch} · {genre} · para {pi}（{len(text)}字）")
        lines.append("")
        lines.append("**原文**：")
        lines.append("")
        lines.append(f"> {text}")
        lines.append("")

        lines.append("### 候选实体（程序预标）")
        lines.append("")
        lines.append("| # | 实体 | 类别 | start-end | 来源 | 判定 |")
        lines.append("|---|------|------|-----------|------|------|")
        for i, c in enumerate(candidates, 1):
            cat = CAT_LABEL.get(c.get("category", "custom"), c.get("category", "custom"))
            lines.append(
                f"| {i} | {c.get('entity','')} | {cat} | {c.get('start_char','')}-{c.get('end_char','')} | {c.get('source','')} |  |"
            )
        if not candidates:
            lines.append("| — | （无候选） | | | | |")
        lines.append("")

        lines.append("### 人工补充（漏掉的实体）")
        lines.append("")
        lines.append("| # | 实体原文 | 类别 |")
        lines.append("|---|----------|------|")
        lines.append("| 1 |  |  |")
        lines.append("| 2 |  |  |")
        lines.append("| 3 |  |  |")
        lines.append("")

        lines.append("---")
        lines.append("")

    lines.append("## 定稿说明")
    lines.append("")
    lines.append("- 审校完这份表后，把你标注的结果（保留/删除/补充）告诉我，我帮你转成 `golden.json`。")
    lines.append("- 或者你直接改 `golden_draft.json` 的 `candidates` 数组，另存为 `golden.json`。")
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("已生成:", OUT)


if __name__ == "__main__":
    main()
