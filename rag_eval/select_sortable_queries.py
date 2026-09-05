"""轻量筛选：复用 recall_rank_attribution.json 的 D/S 排名，用 src_only qrels 抽 sortable query。

不碰 embedder、不跑检索，纯数据筛选（秒级）。

口径：
- src_only rel=2 核心 = qrels_*_src_only.tsv 里 rel=2。
- sortable = 该核心在现有 detail 里 category == "sortable"（D/S top-11~50）。
- 抽中对象 = 有至少一个 src_only sortable 核心的 query。
"""
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "rag_eval" / "results"
GT = PROJECT_ROOT / "rag_eval" / "ground_truth"
DATASETS = ["academic_deepseek", "academic_qwen", "daily_deepseek", "daily_qwen"]

N_PICK = 60
N_COMPARISON = 12
N_PLOT = 42
N_ENTITY = 6


def load_qrels(path):
    qrels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            qid, did = parts[0], parts[1]
            rel = int(parts[2])
            if rel == 0:
                continue
            qrels.setdefault(qid, {})[did] = rel
    return qrels


def load_queries(path):
    qs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            qs[q["query_id"]] = q
    return qs


def main():
    rank = json.load(open(RESULTS / "recall_rank_attribution.json", encoding="utf-8"))
    # detail_map: (dataset, query_id, chunk_id) -> category
    detail_map = {}
    for x in rank["detail"]:
        detail_map[(x["dataset"], x["query_id"], x["chunk_id"])] = x["category"]

    candidates = []  # [(dataset, query_id, category, sortable_chunks)]
    for ds in DATASETS:
        qrels = load_qrels(GT / f"qrels_{ds}_src_only.tsv")
        qs = load_queries(GT / f"queries_{ds}.jsonl")
        for qid, rels in qrels.items():
            core = {c for c, r in rels.items() if r == 2}
            if not core:
                continue
            sortable = [c for c in core if detail_map.get((ds, qid, c)) == "sortable"]
            if sortable:
                qcat = qs.get(qid, {}).get("category", "")
                candidates.append((ds, qid, qcat, sortable))

    print(f"有 src_only sortable 核心的 query 总数: {len(candidates)}")
    by_ds = defaultdict(int)
    for c in candidates:
        by_ds[c[0]] += 1
    print("按数据集分布:", dict(by_ds))
    by_cat = defaultdict(int)
    for c in candidates:
        by_cat[c[2]] += 1
    print("按 category 分布:", dict(by_cat))

    def pick(cat, n_total, max_per_ds):
        pool = sorted([c for c in candidates if c[2] == cat], key=lambda c: -len(c[3]))
        picked = []
        per_ds = defaultdict(int)
        seen = set()
        # 第一轮：每数据集先抽 1 个，保证跨数据集均衡
        for c in pool:
            if len(picked) >= n_total:
                return picked
            ds = c[0]
            if per_ds[ds] >= 1:
                continue
            picked.append(c)
            per_ds[ds] += 1
            seen.add((c[0], c[1]))
        # 第二轮：按 sortable 核心数补足，每数据集最多 max_per_ds
        for c in pool:
            if len(picked) >= n_total:
                break
            if (c[0], c[1]) in seen:
                continue
            ds = c[0]
            if per_ds[ds] >= max_per_ds:
                continue
            picked.append(c)
            per_ds[ds] += 1
            seen.add((c[0], c[1]))
        return picked

    picked = (
        pick("comparison", N_COMPARISON, 4)
        + pick("plot", N_PLOT, 12)
        + pick("entity", N_ENTITY, 2)
    )

    print(f"\n抽中 {len(picked)} 个 query:")
    for ds, qid, qcat, sc in picked:
        print(f"  {ds}/{qid}  category={qcat}  sortable核心={len(sc)}  {sc}")

    out = [
        {"dataset": ds, "query_id": qid, "category": qcat, "sortable_chunks": sc}
        for ds, qid, qcat, sc in picked
    ]
    out_path = RESULTS / "picked_sortable_queries.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OUT] {out_path}")


if __name__ == "__main__":
    main()
