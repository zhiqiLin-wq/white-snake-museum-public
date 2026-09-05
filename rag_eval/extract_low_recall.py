"""从 detail_log 提取 F 配置低召回 query 子日志。

用法:
    python rag_eval/extract_low_recall.py <源日志> <输出文件> [阈值=0.5]
"""
import re
import sys
from pathlib import Path


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    lines = src.read_text(encoding="utf-8").split("\n")

    # 按「90 个 -」分隔符切块（数据集头是「=」，不会误切）
    blocks = []
    cur = []
    for line in lines:
        s = line.strip()
        if s and set(s) == {"-"}:
            if cur:
                blocks.append(cur)
            cur = [line]
        else:
            cur.append(line)
    if cur:
        blocks.append(cur)

    kept = []
    summary = []
    total = 0
    pat_recall = re.compile(r"【Fusion \(F\)】 Recall@10 = \d+/\d+ = ([\d.]+)")
    pat_qid = re.compile(r"Query \d+/\d+\s+(\S+)")
    pat_qtext = re.compile(r"原始 query: (.*)")
    pat_ds = re.compile(r"\[([^\]]+)\] Query \d+/\d+")

    for b in blocks:
        text = "\n".join(b)
        m = pat_recall.search(text)
        if not m:
            continue
        total += 1
        rec = float(m.group(1))
        if rec < threshold:
            kept.append(text)
            qid = pat_qid.search(text)
            qtext = pat_qtext.search(text)
            ds = pat_ds.search(text)
            summary.append((
                ds.group(1) if ds else "?",
                qid.group(1) if qid else "?",
                rec,
                qtext.group(1)[:50] if qtext else "?",
            ))

    # 写子日志：加一个头部摘要
    with open(dst, "w", encoding="utf-8") as f:
        f.write(f"低召回子日志（F 配置 Recall@10 < {threshold}）— 共 {len(kept)}/{total} 条\n")
        f.write("=" * 90 + "\n")
        f.write(f"{'数据集':<20}{'query_id':<16}{'recall':>8}  query\n")
        f.write("-" * 90 + "\n")
        for ds, qid, rec, qtext in summary:
            f.write(f"{ds:<20}{qid:<16}{rec:>8.3f}  {qtext}\n")
        f.write("=" * 90 + "\n\n")
        for text in kept:
            f.write(text + "\n")

    print(f"总 query: {total}, 低召回(<{threshold}): {len(kept)}")
    print(f"输出: {dst}")


if __name__ == "__main__":
    main()
