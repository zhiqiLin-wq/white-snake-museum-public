# -*- coding: utf-8 -*-
"""C2: 补算主文献 paragraph_index（dry-run，不落库）

口径对齐链（全部与生产一致）:
- 章节切分: 复现 server/src/services/literature.ts parseLiterature
  （行首 中文数字+"、" 期别行，content=标题后到下一标题前，trim）
- 段落切分: 复现 client splitParagraphs（\r\n→\n 规范化，按 \n\n 分段，trim，非空）
- 段落定位: char_start 有效 → 落点段落 + 文本交叉验证；
  失败/缺失 → chunk 首句文本规范化匹配
"""
import json
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"e:\Flow\white-snake-museum-public\agent")

from server.rag.tag_store import load_corpus, CORPUS_PATH  # noqa: E402

SRC = r"e:\Flow\white-snake-museum-public\excel_data\白蛇传文献选集.txt"
CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7}


def norm_text(s: str) -> str:
    """规范化: 去所有空白 + 统一引号变体，用于文本交叉验证/匹配"""
    return re.sub(r"[\s\u3000]+", "", s).replace("\u201c", "\"").replace("\u201d", "\"") \
        .replace("\u2018", "'").replace("\u2019", "'").replace("（", "(").replace("）", ")")


def parse_chapters(raw: str) -> dict[int, dict]:
    """复现 parseLiterature: 返回 {章号int: {"title", "content"(原始,trim后)}}"""
    regex = re.compile(r"^\s*([一二三四五六七])\s*[、，,.\s]\s*(.+)$", re.M)
    titles = [(m.start(), m.end(), m.group(1), m.group(2).strip()) for m in regex.finditer(raw)]
    chapters = {}
    for i, (s, e, num, title) in enumerate(titles):
        start = s
        end = titles[i + 1][0] if i + 1 < len(titles) else len(raw)
        content = raw[start + (e - s):end].strip()
        chapters[CN_NUM[num]] = {"title": title, "content": content}
    return chapters


def split_paragraphs_with_spans(content: str):
    """复现 splitParagraphs 并记录每段在规范化 content 中的字符区间"""
    norm = content.replace("\r\n", "\n").replace("\r", "\n")
    paras = []
    pos = 0
    for seg in norm.split("\n\n"):
        t = seg.strip()
        if t:
            start = pos + (len(seg) - len(seg.lstrip()))
            paras.append({"text": t, "start": start, "end": start + len(t)})
        pos += len(seg) + 2
    return paras


def main():
    raw = open(SRC, encoding="utf-8").read()
    print(f"原文: {len(raw)} 字符, 含\\r: {raw.count(chr(13)) > 0}")
    chapters = parse_chapters(raw)
    print(f"章节: {len(chapters)} 个 → {sorted(chapters.keys())}: {[c['title'][:18] for k, c in sorted(chapters.items())]}")

    # 预处理每章段落 + 规范化文本索引
    for ch in chapters.values():
        ch["paras"] = split_paragraphs_with_spans(ch["content"])
        ch["para_norms"] = [norm_text(p["text"]) for p in ch["paras"]]

    corpus = load_corpus()
    results = {}
    stats: Counter = Counter()
    for cid, doc in corpus.items():
        meta = doc.get("metadata", {}) or {}
        if meta.get("source_type") != "primary_literature":
            continue
        ch_num = CN_NUM.get(str(meta.get("chapter_number", "")).strip())
        text = doc.get("text", "") or ""
        para_idx, method = -1, "none"
        ch = chapters.get(ch_num)
        if ch is None:
            stats["章节未找到"] += 1
        elif not text:
            stats["空文本"] += 1
        else:
            # C2b: 逐行匹配 + 行序递增消歧
            # - chunk 可能含期别标题行（"七、流变期…"）→ 跳过
            # - chunk 从段中/短段开始 → 用第一个 >=6 字的正文行定位起点段落，
            #   再用后续行消歧（chunk 内行序 = 原文行序，段落号应递增）
            TITLE_RE = re.compile(r"^[一二三四五六七]\s*[、，,.\s]")
            heads = [norm_text(x) for x in text.split("\n")
                     if len(norm_text(x)) >= 6 and not TITLE_RE.match(x.strip())][:3]
            hit = None
            if heads:
                first_lists = [[i for i, pn in enumerate(ch["para_norms"]) if h in pn]
                               for h in heads]
                first_lists = [c for c in first_lists if c]
                if first_lists:
                    if len(first_lists) == 1:
                        hit = first_lists[0][0]
                    else:
                        # 多行都有命中：找第一个满足与下一行命中段落序号一致的起点
                        hit = None
                        for p1 in first_lists[0]:
                            for nxt in first_lists[1:]:
                                if any(p2 >= p1 for p2 in nxt):
                                    hit = p1
                                    break
                            if hit is not None:
                                break
                        if hit is None:
                            hit = first_lists[0][0]
            if hit is not None:
                para_idx, method = hit, "文本匹配"
            # 2) char_start 落点兜底
            elif isinstance(cs := meta.get("char_start"), int) and 0 <= cs < len(ch["content"]):
                hit2 = next((i for i, p in enumerate(ch["paras"]) if p["start"] <= cs < p["end"]), None)
                if hit2 is not None:
                    para_idx = hit2
                    method = "char_start+验证通过" if (
                        hit2 < len(ch["para_norms"]) and heads and heads[0][:12] in ch["para_norms"][hit2]
                    ) else "char_start落点(文本验证未过)"
                else:
                    method = "char_start未落任何段"
            if para_idx < 0:
                method = "未命中"
        stats[method] += 1
        old_pi = meta.get("paragraph_index")
        results[cid] = {"old": old_pi, "new": para_idx, "method": method,
                        "chapter": ch_num, "old_valid": isinstance(old_pi, int) and old_pi >= 0}

    print(f"\n=== 补算结果 ({len(results)} 块主文献) ===")
    for k, v in stats.most_common():
        print(f"  {k}: {v}")

    changed = sum(1 for r in results.values() if r["new"] != r["old"])
    oldneg_to_valid = sum(1 for r in results.values() if r["old"] == -1 and r["new"] >= 0)
    conflict = [cid for cid, r in results.items() if r["old_valid"] and r["new"] >= 0 and r["new"] != r["old"]]
    print(f"\n新值≠旧值: {changed} 块 | 旧-1→有效值: {oldneg_to_valid} 块 | 旧有效值与新值冲突: {len(conflict)} 块")

    # 分布
    combo = Counter()
    for cid, r in results.items():
        combo[(r["chapter"], r["new"])] += 1
    print(f"\n补算后 (章节,段落) 组合数: {len(combo)} (原 8 种)")
    print("样例(每章前 6 个组合):")
    seen: dict[int, int] = {}
    for (c, p), n in sorted(combo.items()):
        if seen.get(c, 0) < 6:
            print(f"  章{c} 段{p}: {n} 块")
            seen[c] = seen.get(c, 0) + 1

    # 未命中样本
    fails = [(cid, r) for cid, r in results.items() if r["new"] < 0]
    print(f"\n未命中样本 (前 5):")
    for cid, r in fails[:5]:
        print(f"  {cid} chapter={r['chapter']} 「{(corpus[cid]['text'] or '')[:40]}…」")

    # 冲突样本
    if conflict:
        print(f"\n新旧冲突样本 (前 5):")
        for cid in conflict[:5]:
            r = results[cid]
            print(f"  {cid} old={r['old']} new={r['new']} method={r['method']}")

    # 保存补算结果供 C3 落库
    out = CORPUS_PATH.parent / "_para_patch.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\n补算结果已存: {out}")


if __name__ == "__main__":
    main()
