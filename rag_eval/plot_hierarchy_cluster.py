"""plot_unit embedding 聚类，建立主干/子情节层级，输出结果供人工确认。"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
for p in [str(PROJECT_ROOT), str(AGENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from server.rag.tag_store import TagStore, _load_vecs

RESULTS = PROJECT_ROOT / "rag_eval" / "results"
GT = PROJECT_ROOT / "rag_eval" / "ground_truth"
THRESHOLD = 0.68


def main():
    tag_store = TagStore()
    tag_store.load()

    # plot_unit 向量（优先缓存，否则需 embed）
    plot_vecs = _load_vecs("plot_unit_vectors")
    if plot_vecs is None:
        from rag_eval.apply_boost import build_embedder
        embedder = build_embedder()
        tag_store._embedder = embedder
        tag_store.build_plot_unit_vectors()
        plot_vecs = tag_store._plot_unit_vecs

    # 词表 plot_units（过滤待删除）
    vocab = json.load(open(RESULTS / "merged_vocab_full.json", encoding="utf-8"))
    units = [e["canonical"] for e in vocab.get("plot_units", []) if e.get("canonical") != "待删除"]
    units = [u for u in units if u in plot_vecs]

    # 计算 df（每个 canonical 出现在多少个 chunk 的 plot_unit/引用情节 标签里）
    df = defaultdict(int)
    for cid, v in tag_store._labels.items():
        lab = v.get("label") or {}
        if "_error" in lab:
            continue
        if v["genre"] == "research_literature":
            fields = lab.get("引用情节") or []
        else:
            fields = lab.get("plot_unit") or []
        for p in fields:
            canon = tag_store._r2c.get(p, p)
            df[canon] += 1

    # 相似度矩阵 + 连通分量聚类
    vecs = np.asarray([plot_vecs[u] for u in units], dtype=float)
    sim = vecs @ vecs.T
    n = len(units)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i][j] >= THRESHOLD:
                union(i, j)

    groups = defaultdict(list)
    for i, u in enumerate(units):
        groups[find(i)].append(u)

    # 每类选 df 最高的做主情节，其余是子情节
    main_units = {}
    sub_to_main = {}
    for members in groups.values():
        main = max(members, key=lambda m: df.get(m, 0))
        main_units[main] = df.get(main, 0)
        for m in members:
            if m != main:
                sub_to_main[m] = main

    # 输出聚类结果
    lines = []
    lines.append(f"plot_unit 聚类结果（阈值 {THRESHOLD}，共 {len(groups)} 类）")
    lines.append("=" * 70)
    # 按主情节 df 降序排
    for main in sorted(main_units, key=lambda m: -main_units[m]):
        members = groups[next(g for g, ms in groups.items() if main in ms)]
        subs = [m for m in members if m != main]
        lines.append(f"主情节: {main}（df={main_units[main]}）")
        if subs:
            lines.append(f"  子情节: {'、'.join(sorted(subs, key=lambda s: -df.get(s,0)))}")
        else:
            lines.append(f"  子情节: (无，单独一类)")
    lines.append("")
    lines.append(f"单元素类（无子情节的主情节）: "
                 f"{sum(1 for g in groups.values() if len(g)==1)} 个")

    out = RESULTS / "plot_hierarchy_cluster.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[OUT] {out}")

    # 落盘 JSON（供 boost 使用）
    hierarchy = {
        "main_units": {m: df.get(m, 0) for m in main_units},
        "sub_to_main": sub_to_main,
    }
    (RESULTS / "plot_hierarchy.json").write_text(
        json.dumps(hierarchy, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OUT] plot_hierarchy.json（main={len(main_units)}, sub_to_main={len(sub_to_main)}）")


if __name__ == "__main__":
    main()
