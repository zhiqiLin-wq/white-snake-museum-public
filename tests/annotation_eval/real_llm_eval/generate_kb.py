"""用 wordlist.json 直接生成完整的 annotation_knowledge_base.py。

避免正则替换的脆弱性，一次生成正确的 KB 文件（label/rules/known/examples 全量）。

运行：
  PYTHONIOENCODING=utf-8 python -m tests.annotation_eval.real_llm_eval.generate_kb
"""

import json
from pathlib import Path

KB = Path("agent/server/mcp/tools/annotation_knowledge_base.py")
WL = Path("tests/annotation_eval/real_llm_eval/data/wordlist.json")


def _py_list(words, per_line=6, indent=12):
    pad = " " * indent
    lines = []
    for i in range(0, len(words), per_line):
        chunk = words[i:i + per_line]
        lines.append(pad + ", ".join(json.dumps(w, ensure_ascii=False) for w in chunk) + ",")
    return "[\n" + "\n".join(lines) + "\n" + " " * (indent - 4) + "]"


def _py_examples(examples, category, indent=8):
    pad = " " * indent
    pad2 = " " * (indent + 4)
    pad3 = " " * (indent + 8)
    lines = [pad + '"examples": [']
    for ex in examples:
        t = json.dumps(ex["text"], ensure_ascii=False)
        entity = json.dumps(ex.get("entity", ex["text"]), ensure_ascii=False)
        ti = json.dumps(ex["title"], ensure_ascii=False)
        lines.append(pad2 + f'({t}, [')
        lines.append(pad3 + f'{{"entity": {entity}, "category": "{category}", "explanation": {ti}}},')
        lines.append(pad2 + ']),')
    lines.append(pad + '],')
    return "\n".join(lines)


RULES = {
    "person": (
        "提取文中出现的所有人物（包括姓名、字号、称谓、身份，如'侍者''使者''娘子''姨''二女奴'）。"
        "只要原文中出现人物就标注，不要遗漏。"
        "不标代词(他/她/其/吾/余)。不标单个姓氏字(许/白/李)。"
    ),
    "location": (
        "提取文中提及的地理位置、建筑或场所。"
        "不标泛指(山中/寺里/城门/街上/巷口)。"
        "不标单个通用字(山/寺/门/桥/塔/楼)，除非它是专名的一部分如'断桥'。"
    ),
    "term": (
        "提取文中涉及的古代职官、典章制度、特定称谓或文言特殊词汇。"
        "不标日常常用词(道/佛/经/庙/缘)。术语可能为2-4字的固定搭配。"
    ),
    "event": (
        "提取直接推动故事情节发展、包含明确动作或人物交互的客观叙事句子。"
        "事件句子的实体原文必须一字不差引用原文完整原句，禁止删改字词、禁止用自己的话概括、禁止加标点断句。"
        "并在概括词列给出该事件的概括名（如'端午惊变''水漫金山'）。"
    ),
    "motif": (
        "提取能够揭示文章核心意旨、人物特质、情感基调或作者议论抒情的句子。"
        "主题句子的实体原文必须一字不差引用原文完整原句，禁止删改字词、禁止用自己的话总结、禁止加标点断句。"
        "并在概括词列给出该主题的概括名（如'报恩''人妖恋'）。"
    ),
}

LABEL = {"person": "人名", "location": "地点", "term": "专业术语", "event": "事件句子", "motif": "主题句子"}

# 人物/地点/术语的 few-shot 示例（简单示例即可）
FIXED_EXAMPLES = {
    "person": [
        {"title": "白蛇传男主角", "text": "许宣道：小生住在过军桥黑珠巷。", "entity": "许宣"},
    ],
    "location": [
        {"title": "杭州城门", "text": "许宣出涌金门，到西湖边。", "entity": "涌金门"},
    ],
    "term": [
        {"title": "法海降妖法器", "text": "法海将钵盂一罩，白娘子现了原形。", "entity": "钵盂"},
    ],
}


def main():
    wl = json.loads(WL.read_text(encoding="utf-8"))

    # 按在文献全文中出现的频率排序（person/location/term 是字面词，频率=重要性）
    lit_full = Path("excel_data/白蛇传文献选集.txt").read_text(encoding="utf-8")

    def sort_by_freq(words):
        return sorted(words, key=lambda w: (-lit_full.count(w), w))

    for cat in ("person", "location", "term"):
        wl[cat] = sort_by_freq(wl[cat])

    # 复合地名补全：报告把"长安东市"拆成了"长安"+"东市"，补回整体，长实体优先整体匹配
    if "长安东市" not in wl["location"]:
        wl["location"].append("长安东市")
        wl["location"] = sort_by_freq(wl["location"])

    parts = []
    parts.append('"""标注领域知识库 — 白蛇传已知实体、类别规则、颜色映射。\n')
    parts.append('\n阶段 0 — P0-05.\n从 annotate_passage.py 提取，供新旧标注管线共享。\n')
    parts.append('\n词表来源：《白蛇传文献选集全面提取报告.docx》提取的人物/地点/专业术语/事件句子/主题句子。\n')
    parts.append('"""\n\n')
    parts.append('_CATEGORY_REVERSE_MAP: dict[str, str] = {\n')
    parts.append('    "人物名称": "person", "地点名称": "location",\n')
    parts.append('    "事件名称": "event", "专有术语": "term", "叙事母题": "motif",\n')
    parts.append('    "人物": "person", "地点": "location",\n')
    parts.append('    "事件": "event", "术语": "term", "母题": "motif",\n')
    parts.append('    "人名": "person", "专业术语": "term", "事件句子": "event", "主题句子": "motif",\n')
    parts.append('    "person": "person", "location": "location",\n')
    parts.append('    "event": "event", "term": "term", "motif": "motif",\n')
    parts.append('}\n\n')
    parts.append('DEFAULT_CATEGORY_COLORS = {\n')
    parts.append('    "person": "#68A0C8",\n    "location": "#78B898",\n    "event": "#68B0B8",\n')
    parts.append('    "term": "#9888C0",\n    "motif": "#C898B0",\n    "custom": "#B8B878",\n')
    parts.append('}\n\n')
    parts.append('_ENTITY_KB: dict[str, dict] = {\n')

    for cat in ["person", "location", "term", "event", "motif"]:
        parts.append(f'    "{cat}": {{\n')
        parts.append(f'        "label": "{LABEL[cat]}",\n')
        parts.append('        "known": ')
        parts.append(_py_list(wl[cat]))
        parts.append(',\n')
        parts.append(f'        "rules": (\n            "{RULES[cat]}"\n        ),\n')
        if cat in ("event", "motif"):
            key = "event_examples" if cat == "event" else "motif_examples"
            parts.append(_py_examples(wl[key], cat))
        else:
            parts.append(_py_examples(FIXED_EXAMPLES[cat], cat))
        parts.append('\n    },\n')

    parts.append('}\n')

    KB.write_text("".join(parts), encoding="utf-8")
    print("已生成:", KB)


if __name__ == "__main__":
    main()
