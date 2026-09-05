"""从《白蛇传文献选集全面提取报告.docx》提取五类词表 + 事件/主题 few-shot 示例。

- 人物 / 地点：每行「• 名称：说明」，取名称拆「/」别名，过滤单字词，并剔除明显非人名。
- 专业术语：每行「• 词、词、词」，拆「、」，去【】。
- 事件句子 / 主题句子：每行「【概括短句】 原文」，取【】里的概括名；并提取完整原文原句（无省略号）做 few-shot 示例（每类选最短 3 条）。

输出 data/wordlist.json。

运行：
  PYTHONIOENCODING=utf-8 python -m tests.annotation_eval.real_llm_eval.build_wordlist
"""

import json
import re
import zipfile
from pathlib import Path

DOCX = Path("E:/Flow/白蛇传文献选集全面提取报告.docx")
OUT = Path(__file__).resolve().parent / "data" / "wordlist.json"

# 明显非人名（植物/泛称/鬼物），从人物词表剔除
PERSON_REMOVE = {"皂荚树", "枯槐树", "二女奴", "地方邻右", "师父", "船夫", "五鬼"}


def _read_paragraphs() -> list[str]:
    z = zipfile.ZipFile(DOCX)
    xml = z.read("word/document.xml").decode("utf-8")
    paras = re.split(r"</w:p>", xml)
    out = []
    for para in paras:
        t = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", para, re.S))
        if t.strip():
            out.append(t.strip())
    return out


def _section_range(paras, start_kw, end_kw, start_from=0):
    start = None
    for i in range(start_from, len(paras)):
        if paras[i].startswith(start_kw):
            start = i
            break
    end = len(paras)
    if start is not None:
        for j in range(start + 1, len(paras)):
            if paras[j].startswith(end_kw):
                end = j
                break
    return (start + 1 if start is not None else 0, end)


def _extract_named(paras, lo, hi, remove=None) -> list[str]:
    items = []
    remove = remove or set()
    for line in paras[lo:hi]:
        line = line.strip()
        if not line.startswith("•"):
            continue
        body = line[1:].strip()
        name = re.split(r"[：:]", body, maxsplit=1)[0].strip()
        if not name:
            continue
        for part in re.split(r"[/／]", name):
            part = part.strip()
            if part and len(part) >= 2 and part not in items and part not in remove:
                items.append(part)
    return items


def _extract_terms(paras, lo, hi) -> list[str]:
    items = []
    for line in paras[lo:hi]:
        line = line.strip()
        if not line.startswith("•"):
            continue
        for w in re.split(r"[、，,]", line[1:].strip()):
            w = w.strip().strip("【】[]")
            if w and len(w) >= 2 and w not in items:
                items.append(w)
    return items


def _extract_summaries_and_fewshot(paras):
    """返回 (summaries_by_cat, fewshot_by_cat)，按事件/主题归类，去重。"""
    summaries = {"event": [], "theme": []}
    fewshot = {"event": [], "theme": []}  # (概括短句, 完整原文原句)
    cur = None
    for line in paras:
        if line.startswith("四、事件句子提取"):
            cur = "event"
            continue
        if line.startswith("五、主题句子提取"):
            cur = "theme"
            continue
        if line.startswith("三、专业术语提取"):
            cur = "term"
            continue
        if line.startswith("一、人物提取") or line.startswith("二、地点提取"):
            cur = "other"
            continue
        if cur not in ("event", "theme"):
            continue
        m = re.match(r"^【(.+?)】\s*(.*)$", line)
        if not m:
            continue
        title = m.group(1).strip()
        text = m.group(2).strip()
        if title not in summaries[cur]:
            summaries[cur].append(title)
        if text and "…" not in text:
            if not any(t == title for t, _ in fewshot[cur]):
                fewshot[cur].append((title, text))

    # 每类选最短 3 条完整原句做 few-shot
    for cat in ("event", "theme"):
        fewshot[cat].sort(key=lambda x: len(x[1]))
        fewshot[cat] = fewshot[cat][:3]
    return summaries, fewshot


def main():
    paras = _read_paragraphs()
    person_lo, person_hi = _section_range(paras, "一、人物提取", "二、地点提取", start_from=8)
    loc_lo, loc_hi = _section_range(paras, "二、地点提取", "三、专业术语提取", start_from=8)
    term_lo, term_hi = _section_range(paras, "三、专业术语提取", "四、事件句子提取", start_from=8)

    persons = _extract_named(paras, person_lo, person_hi, remove=PERSON_REMOVE)
    locations = _extract_named(paras, loc_lo, loc_hi)
    terms = _extract_terms(paras, term_lo, term_hi)
    summaries, fewshot = _extract_summaries_and_fewshot(paras)

    data = {
        "person": persons,
        "location": locations,
        "term": terms,
        "event": summaries["event"],
        "motif": summaries["theme"],
        "event_examples": [{"title": t, "text": s} for t, s in fewshot["event"]],
        "motif_examples": [{"title": t, "text": s} for t, s in fewshot["theme"]],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print("人物:", len(persons), "样例:", persons[:6])
    print("地点:", len(locations))
    print("术语:", len(terms))
    print("事件概括名:", len(summaries["event"]), "few-shot:", [(t, len(s)) for t, s in fewshot["event"]])
    print("主题概括名:", len(summaries["theme"]), "few-shot:", [(t, len(s)) for t, s in fewshot["theme"]])
    print("已写出:", OUT)


if __name__ == "__main__":
    main()
