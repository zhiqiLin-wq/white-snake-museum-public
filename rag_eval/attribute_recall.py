"""召回失败归因分析：把 recall 低拆解到「核心答案(rel=2) vs 跨版本串(rel=1)」。

背景:
    raw recall@10(rel>=1) = 0.476 被两个因素共同压低:
      1. qrels 把「跨版本同类情节」标成 rel=1 正例，分母灌水（数据集问题）
      2. 检索本身丢核心答案（算法问题，core recall 仅 0.537）

    本脚本不重跑检索，直接解析 detail_log 里已写出的 D/S/F 三路结果，
    对每条 query 精确分类「核心丢在哪」，回答一个关键问题:

      核心答案没进 top-10，到底是「D/S 本来就找不到」还是「D/S 找到了但被融合埋没」？

分类维度（针对每条 query 的 F 配置 top-10）:
    core_recall  = F 命中的 rel=2 数 / rel=2 总数
    rel1_recall  = F 命中的 rel=1 数 / rel=1 总数

    6 种失败模式:
      core_full              : 核心全找到（rel=2 一个不丢）
      core_partial_rel1_high : 丢部分核心，但串捞得多（串挤核心，融合/排序问题）
      core_partial_rel1_low  : 核心和串都丢部分（底层检索弱）
      core_miss_rel1_high    : 核心全丢但串捞一堆（严重串挤核心）
      core_miss_rel1_low     : 核心全丢且串也漏（检索彻底失败）

核心丢失定位（对 core_recall<1 的 query，逐 chunk 看 D/S 路）:
    d_or_s_found    : 核心 chunk 在 D 或 S 的 top-10 里，但被融合埋没 → 融合损失
    neither_found   : D/S 都找不到该核心 → 底层 dense/BM25 弱

Usage:
    python rag_eval/attribute_recall.py
    python rag_eval/attribute_recall.py --log results/detail_log_20260817T122003.txt
    python rag_eval/attribute_recall.py --out results/recall_attribution.json
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# detail_log 行格式锚点（与 detail_log.py 的 w() 输出严格对齐）
RE_QUERY = re.compile(
    r'\[(academic_deepseek|academic_qwen|daily_deepseek|daily_qwen)\] '
    r'Query \d+/\d+  (q_\d+)'
)
RE_GOLD_LINE = re.compile(r'^  \[\d+\] (\S+)  \(.*rel=(\d)\)$')
RE_RETR_LINE = re.compile(r'^  #\s*(\d+) \[(HIT|MISS)\s*\] (\S+)  \(')

CONFIG_LABELS = {
    "Dense-only": "D",
    "BM25-only": "S",
    "Fusion": "F",
}


def find_latest_log() -> Path | None:
    """返回最新的完整 detail_log（时间戳命名），排除 low_recall 等派生摘要文件。"""
    logs = [
        p for p in RESULTS_DIR.glob("detail_log_*.txt")
        if re.match(r"detail_log_\d{8}T\d{6}\.txt$", p.name)
    ]
    return max(logs, key=lambda p: p.name) if logs else None


def parse_log(path: Path) -> list[dict]:
    """逐行解析 detail_log，返回逐 query 的结构化记录。

    每条记录:
        {dataset, query_id, gold: {chunk_id: rel},
         top10: {"D": [chunk_id], "S": [...], "F": [...]}}
    """
    records: list[dict] = []
    cur: dict | None = None
    section: str | None = None  # 'gold' | 'D' | 'S' | 'F' | None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            m = RE_QUERY.search(line)
            if m:
                # 开启新 query，收尾上一条
                cur = {
                    "dataset": m.group(1),
                    "query_id": m.group(2),
                    "gold": {},
                    "top10": {"D": [], "S": [], "F": []},
                }
                records.append(cur)
                section = None
                continue

            if cur is None:
                continue

            if "【标准答案】" in line:
                section = "gold"
                continue

            for label, cfg in CONFIG_LABELS.items():
                if f"【{label}" in line:
                    section = cfg
                    break

            if section == "gold":
                m = RE_GOLD_LINE.match(line)
                if m:
                    cur["gold"][m.group(1)] = int(m.group(2))
            elif section in ("D", "S", "F"):
                m = RE_RETR_LINE.match(line)
                if m:
                    cur["top10"][section].append(m.group(3))

    return records


def _recall(hit_ids: list[str], rel_set: set[str]) -> float:
    if not rel_set:
        return 0.0
    return sum(1 for cid in hit_ids if cid in rel_set) / len(rel_set)


def attribute(records: list[dict]) -> dict:
    """对每条 query 计算归因，聚合出模式分布与核心丢失定位。"""
    per_query = []
    mode_counter = Counter()
    n_core_miss = 0
    n_core_partial = 0

    # 核心丢失定位的全局统计
    core_lost_by_fusion = 0      # 核心在 D/S 里但被融合埋没
    core_lost_by_retrieval = 0   # 核心 D/S 都找不到
    core_lost_total = 0

    core_recalls: list[float] = []
    rel1_recalls: list[float] = []
    raw_recalls: list[float] = []

    for rec in records:
        gold = rec["gold"]
        core_set = {cid for cid, rel in gold.items() if rel == 2}
        rel1_set = {cid for cid, rel in gold.items() if rel == 1}
        all_set = set(gold.keys())

        f_top = rec["top10"]["F"]
        d_top = rec["top10"]["D"]
        s_top = rec["top10"]["S"]

        core_hit = [cid for cid in f_top if cid in core_set]
        rel1_hit = [cid for cid in f_top if cid in rel1_set]

        core_total = len(core_set)
        rel1_total = len(rel1_set)

        core_recall = _recall(core_hit, core_set)
        rel1_recall = _recall(rel1_hit, rel1_set)
        raw_recall = _recall(f_top, all_set)

        if core_total > 0:
            core_recalls.append(core_recall)
        if rel1_total > 0:
            rel1_recalls.append(rel1_recall)
        raw_recalls.append(raw_recall)

        # 模式分类
        if core_total == 0:
            mode = "no_core"
        elif core_recall == 1.0:
            mode = "core_full"
        elif core_recall == 0.0:
            n_core_miss += 1
            mode = "core_miss_rel1_high" if rel1_recall >= 0.5 else "core_miss_rel1_low"
        else:
            n_core_partial += 1
            mode = "core_partial_rel1_high" if rel1_recall >= 0.5 else "core_partial_rel1_low"
        mode_counter[mode] += 1

        # 核心丢失定位（对 core_recall < 1 的 query）
        lost_chunks = []
        if core_total > 0 and core_recall < 1.0:
            missed = core_set - set(core_hit)
            for cid in missed:
                core_lost_total += 1
                if cid in d_top or cid in s_top:
                    core_lost_by_fusion += 1
                    src = "D" if cid in d_top else "S"
                    lost_chunks.append({"chunk_id": cid, "lost_by": "fusion", "found_in": src})
                else:
                    core_lost_by_retrieval += 1
                    lost_chunks.append({"chunk_id": cid, "lost_by": "retrieval", "found_in": None})

        per_query.append({
            "dataset": rec["dataset"],
            "query_id": rec["query_id"],
            "core_total": core_total,
            "rel1_total": rel1_total,
            "core_hit": len(core_hit),
            "rel1_hit": len(rel1_hit),
            "core_recall": round(core_recall, 4),
            "rel1_recall": round(rel1_recall, 4),
            "raw_recall": round(raw_recall, 4),
            "mode": mode,
            "lost_core_chunks": lost_chunks,
        })

    return {
        "per_query": per_query,
        "mode_distribution": dict(mode_counter),
        "n_query": len(records),
        "n_core_miss": n_core_miss,
        "n_core_partial": n_core_partial,
        "core_recall_mean": (sum(core_recalls) / len(core_recalls)) if core_recalls else 0.0,
        "rel1_recall_mean": (sum(rel1_recalls) / len(rel1_recalls)) if rel1_recalls else 0.0,
        "raw_recall_mean": (sum(raw_recalls) / len(raw_recalls)) if raw_recalls else 0.0,
        "core_lost_total": core_lost_total,
        "core_lost_by_fusion": core_lost_by_fusion,
        "core_lost_by_retrieval": core_lost_by_retrieval,
    }


def _mean_ratio(part: int, total: int) -> str:
    return f"{part}/{total} ({part / total * 100:.1f}%)" if total else "0/0"


def print_summary(result: dict) -> None:
    mode_names = {
        "core_full": "核心全找到",
        "core_partial_rel1_high": "丢部分核心·串多(串挤核心)",
        "core_partial_rel1_low": "丢部分核心·串也少(检索弱)",
        "core_miss_rel1_high": "核心全丢·串一堆(严重串挤)",
        "core_miss_rel1_low": "核心全丢·串也漏(彻底失败)",
        "no_core": "无 rel=2 标注",
    }
    n = result["n_query"]
    print("=" * 70)
    print("召回失败归因 — 核心答案(rel=2) vs 跨版本串(rel=1)")
    print("=" * 70)
    print(f"总 query: {n}")
    print(f"总体 recall@10(rel>=1)  = {result['raw_recall_mean']:.4f}")
    print(f"总体 core recall(rel=2) = {result['core_recall_mean']:.4f}  (核心答案真实召回)")
    print(f"总体 rel1 recall(rel=1) = {result['rel1_recall_mean']:.4f}  (跨版本串召回)")
    print()

    print("【失败模式分布】")
    for mode in ["core_full", "core_partial_rel1_high", "core_partial_rel1_low",
                 "core_miss_rel1_high", "core_miss_rel1_low", "no_core"]:
        cnt = result["mode_distribution"].get(mode, 0)
        if cnt:
            print(f"  {mode:<24} {cnt:>5}  ({cnt / n * 100:5.1f}%)  {mode_names[mode]}")

    print()
    print("【核心丢失定位】(core_recall<1 时，逐核心 chunk 看 D/S 路是否已找到)")
    print(f"  丢失核心 chunk 总数: {result['core_lost_total']}")
    print(f"  被融合埋没(D/S 已找到但 F 没进 top-10): "
          f"{_mean_ratio(result['core_lost_by_fusion'], result['core_lost_total'])}")
    print(f"  D/S 本身就找不到(底层检索弱):           "
          f"{_mean_ratio(result['core_lost_by_retrieval'], result['core_lost_total'])}")
    print("=" * 70)


def dump_core_miss(per_query: list[dict], out_path: Path) -> None:
    """单独导出 core_miss（核心全丢）的 query 明细，供后续优先攻关。"""
    miss = [q for q in per_query if q["mode"].startswith("core_miss")]
    miss.sort(key=lambda q: (q["rel1_recall"], -q["rel1_total"]))
    if not miss:
        return
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"核心答案全丢的 query 明细（共 {len(miss)} 条，按串召回降序）\n")
        f.write("=" * 70 + "\n")
        for q in miss:
            f.write(f"[{q['dataset']}] {q['query_id']}  "
                    f"core_hit={q['core_hit']}/{q['core_total']}  "
                    f"rel1_hit={q['rel1_hit']}/{q['rel1_total']}  "
                    f"mode={q['mode']}\n")
            for c in q["lost_core_chunks"]:
                where = f"被融合埋没(在{c['found_in']}路)" if c["lost_by"] == "fusion" else "D/S都找不到"
                f.write(f"    MISS {c['chunk_id']}  ->  {where}\n")
    print(f"\n核心全丢明细已写: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="召回失败归因分析")
    parser.add_argument("--log", type=str, default=None,
                        help="detail_log 路径，默认自动找 results 下最新的 detail_log_*.txt")
    parser.add_argument("--out", type=str, default=None,
                        help="逐 query 归因 JSON 输出路径")
    args = parser.parse_args()

    log_path = Path(args.log) if args.log else find_latest_log()
    if log_path is None or not log_path.exists():
        print(f"[ERROR] 未找到 detail_log，请用 --log 指定", file=sys.stderr)
        sys.exit(1)
    print(f"[LOAD] 解析 {log_path}")

    records = parse_log(log_path)
    if not records:
        print("[ERROR] 解析到 0 条 query，检查日志格式", file=sys.stderr)
        sys.exit(1)

    result = attribute(records)
    print_summary(result)

    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n逐 query 归因 JSON 已写: {out}")

    miss_path = RESULTS_DIR / "recall_attribution_core_miss.txt"
    dump_core_miss(result["per_query"], miss_path)


if __name__ == "__main__":
    main()
