"""一次性探查：报告 docx 里 event/motif 的【概括名】原文句 有多少能对齐回文献全文。

只读，不改任何生产代码/数据。用于评估「句级映射」方案的可行性。
"""
import json
import re
import zipfile
from pathlib import Path

DOCX = Path("E:/Flow/白蛇传文献选集全面提取报告.docx")
LIT = Path("excel_data/白蛇传文献选集.txt")


def read_paras(docx: Path) -> list[str]:
    z = zipfile.ZipFile(docx)
    xml = z.read("word/document.xml").decode("utf-8")
    paras = re.split(r"</w:p>", xml)
    out = []
    for p in paras:
        t = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, re.S))
        if t.strip():
            out.append(t.strip())
    return out


def extract_pairs(paras: list[str]) -> dict[str, list[tuple[str, str]]]:
    """返回 {cat: [(概括名, 原文句), ...]}，event 用 四、事件句子提取，theme 用 五、主题句子提取。"""
    result = {"event": [], "motif": []}
    cur = None
    for line in paras:
        if line.startswith("四、事件句子提取"):
            cur = "event"
            continue
        if line.startswith("五、主题句子提取"):
            cur = "motif"
            continue
        if line.startswith("三、专业术语提取"):
            cur = "term"
            continue
        if line.startswith("一、人物提取") or line.startswith("二、地点提取"):
            cur = "other"
            continue
        if cur not in ("event", "motif"):
            continue
        m = re.match(r"^【(.+?)】\s*(.*)$", line)
        if not m:
            continue
        title = m.group(1).strip()
        text = m.group(2).strip()
        result[cur].append((title, text))
    return result


def strip_punct(s: str) -> str:
    return re.sub(r"[\s，。！？、；：""''（）()【】《》…·—…,\.\!\?\:;]", "", s)


def main():
    paras = read_paras(DOCX)
    pairs = extract_pairs(paras)
    lit = LIT.read_text(encoding="utf-8")

    for cat in ("event", "motif"):
        arr = pairs[cat]
        full = [(t, s) for t, s in arr if s and "…" not in s and "…" not in s]
        ellipsis = [(t, s) for t, s in arr if s and ("…" in s or "…" in s)]
        empty = [(t, s) for t, s in arr if not s]

        exact = 0
        exact_set = set()
        strip_hit = 0
        for t, s in full:
            if s in lit:
                exact += 1
                exact_set.add((t, s))
            else:
                # 去标点模糊对齐（整句在文献中去标点后能找到）
                lit_c = strip_punct(lit)
                s_c = strip_punct(s)
                if s_c and s_c in lit_c:
                    strip_hit += 1

        print(f"\n===== {cat} =====")
        print(f"总条数: {len(arr)}")
        print(f"  完整句(无省略号): {len(full)}")
        print(f"  省略号截断句: {len(ellipsis)}")
        print(f"  空原文句: {len(empty)}")
        print(f"  完整句中【精确命中】文献全文: {exact}")
        print(f"  完整句中【去标点命中】文献全文: {strip_hit}")
        print(f"  完整句中【未能对齐】: {len(full) - exact - strip_hit}")

        # 打几条例句看格式
        print("  完整句样例(前3):")
        for t, s in full[:3]:
            print(f"    【{t}】 {s[:40]}")
        print("  省略号样例(前3):")
        for t, s in ellipsis[:3]:
            print(f"    【{t}】 {s[:40]}...")


if __name__ == "__main__":
    main()
