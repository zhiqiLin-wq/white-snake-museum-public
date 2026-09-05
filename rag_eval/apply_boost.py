"""阶段 6：应用 boost 规则 + 验证三指标。

流程:
1. chunk 标签归一化: 原始值 -> [{"canonical","raw"}]（代码查表，raw->canonical）
2. fused_base = 0.5 * minmax(dense) + 0.5 * minmax(bm25)
3. boost: 按 query_type 分流的标签 boost（文学/研究双轨 + 版本元信息 + summary 语义）
4. final = (fused_base + boost) * (1.2 if chunk.genre == query.genre else 1.0)
5. 重排 top-10，验证三指标: 核心捞回率 / core recall@10 变化 / 副作用
"""
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
for p in [str(PROJECT_ROOT), str(AGENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rag_eval.loader import load_corpus, load_qrels

RESULTS = PROJECT_ROOT / "rag_eval" / "results"
GT = PROJECT_ROOT / "rag_eval" / "ground_truth"

DENSE_WEIGHT = 0.5
TOP_K = 10
HARD_OFFSET = 10.0        # 硬分组前置偏移（情节题 plot_unit / 地点题 locations / 研究分析题 分析主题）
PLOT_DETAIL_WEIGHT = 0.20  # plot_detail 连续 Jaccard 权重（从 0.08 加大）

GENRE_META = {
    "classical_chinese": {"dynasty": "唐", "text": "李黄", "author": "佚名", "name": "文言小说"},
    "huaben": {"dynasty": "宋", "text": "西湖三塔记", "author": "佚名", "name": "话本"},
    "imitation_huaben": {"dynasty": "明", "text": "白娘子永镇雷峰塔", "author": "冯梦龙", "name": "拟话本"},
    "opera_libretto": {"dynasty": "清", "text": "雷峰塔传奇", "author": "方成培", "name": "戏曲"},
    "chapter_vernacular": {"dynasty": "清", "text": "雷峰塔传奇", "author": "玉山堂主人", "name": "白话神魔小说"},
    "tanci": {"dynasty": "清", "text": "义妖传", "author": "陈遇乾", "name": "苏州弹词"},
    "script": {"dynasty": "现当代", "text": "白蛇传", "author": "田汉", "name": "京剧"},
    "research_literature": {"dynasty": None, "text": "研究文献", "author": "研究团队", "name": "学术"},
}

DIM_FIELD = {"朝代": "dynasty", "文本": "text", "体裁": "name", "作者": "author"}
VERSION_DIMS = {"朝代", "文本", "体裁", "作者"}


# ==============================================================================
# 归一化
# ==============================================================================
def build_raw_to_canonical(vocab):
    m = {}
    for kind in ["persons", "locations", "plot_units", "topics"]:
        for e in vocab.get(kind, []):
            canonical = e.get("canonical", "")
            if canonical == "待删除":
                continue
            for raw in e.get("raw", []):
                m.setdefault(raw, canonical)
    return m


def normalize_list(values, r2c):
    out = []
    for v in values:
        out.append({"canonical": r2c.get(v, v), "raw": v})
    return out


def normalize_chunk_label(label, genre, r2c):
    out = {}
    if genre == "research_literature":
        ent = label.get("引用实体", {}) or {}
        out["涉及版本"] = label.get("涉及版本", []) or []
        out["分析主题"] = normalize_list(label.get("分析主题", []) or [], r2c)
        out["引用实体"] = {
            "persons": normalize_list(ent.get("persons", []) or [], r2c),
            "locations": normalize_list(ent.get("locations", []) or [], r2c),
        }
        out["引用情节"] = normalize_list(label.get("引用情节", []) or [], r2c)
        out["summary"] = label.get("summary", "") or ""
    else:
        out["persons"] = normalize_list(label.get("persons", []) or [], r2c)
        out["locations"] = normalize_list(label.get("locations", []) or [], r2c)
        out["plot_unit"] = normalize_list(label.get("plot_unit", []) or [], r2c)
        out["plot_detail"] = label.get("plot_detail", "") or ""
        out["summary"] = label.get("summary", "") or ""
    return out


# ==============================================================================
# 融合 / boost
# ==============================================================================
def compute_idf(norm_chunk):
    """统计 canonical 实体（persons+locations）的 df，返回归一化 IDF（0.3~1.5）。

    高频实体（白素贞/许仙）区分度低、权重低；低频实体（黑风仙/白鹤童子）权重高。
    """
    df = defaultdict(int)
    for nl in norm_chunk.values():
        if nl is None:
            continue
        persons = _cset(nl.get("persons", []))
        locs = _cset(nl.get("locations", []))
        ent = nl.get("引用实体") or {}
        persons |= _cset(ent.get("persons", []))
        locs |= _cset(ent.get("locations", []))
        for p in persons:
            df[p] += 1
        for l in locs:
            df[l] += 1
    N = len(norm_chunk)
    idf = {e: math.log((N + 1) / (d + 1)) for e, d in df.items()}
    if idf:
        mean_idf = sum(idf.values()) / len(idf)
        return {e: max(0.3, min(1.5, idf[e] / mean_idf)) for e in idf}
    return {}


def _minmax(d):
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    if hi == lo:
        return {k: 0.5 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def _bigrams(s):
    s = re.sub(r"\s+", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def plot_detail_sim(a, b):
    """char bigram Jaccard，返回 0~1 的连续相似度。"""
    if not a or not b:
        return 0.0
    ba, bb = _bigrams(a), _bigrams(b)
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)


def _cset(items):
    return {x["canonical"] for x in items if isinstance(x, dict)}


def _rset(items):
    return {x["raw"] for x in items if isinstance(x, dict)}


def cmp_has_version_dim(cmp_dim):
    return bool(set(cmp_dim.keys()) & VERSION_DIMS) if isinstance(cmp_dim, dict) else False


def cmp_has_appellation(cmp_dim):
    return "称谓" in cmp_dim if isinstance(cmp_dim, dict) else False


def version_matches(genre, cmp_dim):
    if genre not in GENRE_META or not isinstance(cmp_dim, dict):
        return False
    meta = GENRE_META[genre]
    for dim, val in cmp_dim.items():
        if dim not in DIM_FIELD:
            continue
        fv = meta.get(DIM_FIELD[dim])
        if not fv:
            continue
        if isinstance(val, list):
            if fv in val:
                return True
        else:
            if fv == val or fv in val or val in fv:
                return True
    return False


def compute_boost(q, c, c_genre, summary_sim, idf_norm):
    boost = 0.0
    hard_match = False
    qtype = q.get("query_type", "")
    # 归一化 query_type（LLM 可能漏写"题"字，如 comparison vs comparison题）
    _qt_alias = {"comparison": "comparison题", "情节": "情节题", "人物": "人物题", "地点": "地点题", "研究分析": "研究分析题"}
    qtype = _qt_alias.get(qtype, qtype)
    q_persons = _cset(q.get("persons", []))
    q_persons_raw = _rset(q.get("persons", []))
    q_locs = _cset(q.get("locations", []))
    q_plot = _cset(q.get("plot_unit", []))
    q_plot_detail = q.get("plot_detail", "") or ""
    q_topic = _cset(q.get("分析主题", []))
    q_cmp = q.get("对比维度", {}) or {}

    if c_genre == "research_literature":
        c_topic = _cset(c.get("分析主题", []))
        c_vers = set(c.get("涉及版本", []) or [])
        ent = c.get("引用实体", {}) or {}
        c_ent_persons = _cset(ent.get("persons", []))
        c_ent_locs = _cset(ent.get("locations", []))
        c_plot = _cset(c.get("引用情节", []))

        if qtype in ("comparison题", "研究分析题"):
            if q_topic and c_topic & q_topic:
                hard_match = True  # 分析主题硬分组
        if qtype == "comparison题" and cmp_has_version_dim(q_cmp):
            if any(version_matches(g, q_cmp) for g in c_vers):
                boost += 0.05  # 涉及版本线性累加
        if qtype in ("comparison题", "研究分析题"):
            if q_plot & c_plot:
                hard_match = True  # 引用情节硬分组（仅 comparison/研究分析题）
        if qtype == "情节题":
            boost += 0.05 * len(q_plot & c_plot)  # 情节题的引用情节保持弱线性
        if qtype in ("情节题", "人物题", "地点题", "研究分析题"):
            boost += 0.03 * sum(idf_norm.get(p, 1.0) for p in (q_persons & c_ent_persons))
            boost += 0.02 * sum(idf_norm.get(l, 1.0) for l in (q_locs & c_ent_locs))
    else:
        c_plot = _cset(c.get("plot_unit", []))
        c_plot_detail = c.get("plot_detail", "") or ""
        c_persons = _cset(c.get("persons", []))
        c_persons_raw = _rset(c.get("persons", []))
        c_locs = _cset(c.get("locations", []))

        if qtype == "情节题":
            # plot_unit 精确匹配 -> 硬分组标记（不加线性分数，改在排序时前置）
            hard_match = bool(q_plot & c_plot)
            # plot_detail 连续相似度，权重加大
            boost += PLOT_DETAIL_WEIGHT * plot_detail_sim(q_plot_detail, c_plot_detail)
        if qtype in ("情节题", "人物题", "地点题", "comparison题"):
            boost += 0.03 * sum(idf_norm.get(p, 1.0) for p in (q_persons & c_persons))
            boost += 0.02 * sum(idf_norm.get(l, 1.0) for l in (q_locs & c_locs))
        if qtype == "comparison题" and cmp_has_appellation(q_cmp):
            boost += 0.05 * len(q_persons_raw & c_persons_raw)

    # 版本元信息（comparison 题，文学 + 研究都适用；研究文献经 GENRE_META[research] 恒不命中，无害）
    if qtype == "comparison题" and cmp_has_version_dim(q_cmp):
        if c_genre in GENRE_META and version_matches(c_genre, q_cmp):
            boost += 0.05

    # 摘要第二路语义（所有 chunk）
    boost += 0.05 * summary_sim
    return boost, hard_match


# ==============================================================================
# 主流程
# ==============================================================================
def build_embedder():
    from server.rag.config import rag_config
    from server.rag.embedder import Embedder

    print("[INIT] loading embedder (summary 语义路)...", flush=True)
    emb = Embedder(rag_config.embedding_model, local_path=rag_config.embedding_local_path or None)
    emb.start_loading()
    waited = 0
    while not emb.is_ready and waited < 300:
        time.sleep(1)
        waited += 1
    if not emb.is_ready:
        raise RuntimeError("embedder 未就绪")
    return emb


def cosine(a, b):
    import numpy as np
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(a @ b / denom)


def main():
    vocab = json.load(open(RESULTS / "merged_vocab2.json", encoding="utf-8"))
    r2c = build_raw_to_canonical(vocab)

    chunk_labels = json.load(open(RESULTS / "chunk_labels.json", encoding="utf-8"))["chunk_labels"]
    query_labels = {f"{r['dataset']}/{r['query_id']}": r["query_label"] for r in
                    json.load(open(RESULTS / "query_labels.json", encoding="utf-8"))}
    candidates = json.load(open(RESULTS / "retrieved_candidates.json", encoding="utf-8"))

    corpus = load_corpus(GT / "corpus.jsonl")

    # 归一化 chunk 标签
    norm_chunk = {}
    for cid, v in chunk_labels.items():
        lab = v.get("label") or {}
        if "_error" in lab:
            norm_chunk[cid] = None
            continue
        norm_chunk[cid] = normalize_chunk_label(lab, v["genre"], r2c)

    idf_norm = compute_idf(norm_chunk)

    embedder = build_embedder()

    # 预计算所有 chunk summary 向量（全局去重，批量 embed）
    all_summaries = {}
    for r in candidates:
        for c in r["candidates"]:
            nl = norm_chunk.get(c["chunk_id"])
            if nl and nl.get("summary"):
                all_summaries[c["chunk_id"]] = nl["summary"]
    summary_ids = list(all_summaries.keys())
    summary_embs = {}
    if summary_ids:
        vecs = embedder.embed([all_summaries[cid] for cid in summary_ids])
        summary_embs = {cid: v for cid, v in zip(summary_ids, vecs)}
    print(f"[INIT] 预计算 {len(summary_ids)} 个 chunk summary 向量", flush=True)

    # 预计算所有 query 原文向量（批量 embed）
    qvecs = embedder.embed([r["query_text"] for r in candidates])
    query_emb = {f"{r['dataset']}/{r['query_id']}": v for r, v in zip(candidates, qvecs)}
    print(f"[INIT] 预计算 {len(qvecs)} 个 query 向量", flush=True)

    lines = ["=" * 80, "结构化标签 boost 验证结果", "=" * 80]

    sum_salvaged = 0
    sum_sortable = 0
    core_recall_before, core_recall_after = [], []
    side_lost, side_total = 0, 0

    for r in candidates:
        ds, qid = r["dataset"], r["query_id"]
        qkey = f"{ds}/{qid}"
        qlabel = query_labels.get(qkey, {})
        if not isinstance(qlabel, dict) or "_error" in qlabel:
            continue

        # 读 src_only qrels 得到 rel=2 和 rel=1
        rels = load_qrels(GT / f"qrels_{ds}_src_only.tsv").get(qid, {})
        core = {c for c, rel in rels.items() if rel == 2}
        rel1 = {c for c, rel in rels.items() if rel == 1}
        rel_all = core | rel1

        sortable_core = set(r.get("sortable_chunks", []))

        # 计算 summary 相似度（用预计算向量）
        q_emb = query_emb.get(qkey)
        summary_sims = {}
        if q_emb is not None:
            for c in r["candidates"]:
                cid = c["chunk_id"]
                if cid in summary_embs:
                    summary_sims[cid] = cosine(q_emb, summary_embs[cid])

        # fused_base（候选集内 minmax 归一，缺省 0）
        dense_scores = {c["chunk_id"]: c["dense_score"] for c in r["candidates"] if c["dense_score"] is not None}
        sparse_scores = {c["chunk_id"]: c["bm25_score"] for c in r["candidates"] if c["bm25_score"] is not None}
        dn = _minmax(dense_scores)
        sn = _minmax(sparse_scores)

        # boost 前 top-10（纯 fused_base）
        before = {}
        for c in r["candidates"]:
            cid = c["chunk_id"]
            before[cid] = DENSE_WEIGHT * dn.get(cid, 0.0) + (1 - DENSE_WEIGHT) * sn.get(cid, 0.0)
        f_top10_before = [cid for cid, _ in sorted(before.items(), key=lambda x: x[1], reverse=True)[:TOP_K]]

        # boost 后
        final = {}
        q_genre = qlabel.get("genre", "") or ""
        for c in r["candidates"]:
            cid = c["chunk_id"]
            c_genre = c["genre"]
            fused_base = DENSE_WEIGHT * dn.get(cid, 0.0) + (1 - DENSE_WEIGHT) * sn.get(cid, 0.0)
            nl = norm_chunk.get(cid)
            boost, hard_match = compute_boost(qlabel, nl or {}, c_genre, summary_sims.get(cid, 0.0), idf_norm)
            genre_factor = 1.2 if c_genre == q_genre and q_genre else 1.0
            final[cid] = (fused_base + boost) * genre_factor
            if hard_match:
                final[cid] += HARD_OFFSET
        top10_after = [cid for cid, _ in sorted(final.items(), key=lambda x: x[1], reverse=True)[:TOP_K]]

        # 指标 1：核心捞回率
        salvaged = sortable_core & set(top10_after)
        sum_salvaged += len(salvaged)
        sum_sortable += len(sortable_core)

        # 指标 2：core recall@10
        if core:
            core_recall_before.append(len(set(f_top10_before) & core) / len(core))
            core_recall_after.append(len(set(top10_after) & core) / len(core))

        # 指标 3：副作用（挤掉的真相关 chunk）
        rel_in_before = set(f_top10_before) & rel_all
        rel_in_after = set(top10_after) & rel_all
        side_total += len(rel_in_before)
        side_lost += len(rel_in_before - rel_in_after)

        lines.append("")
        lines.append("-" * 80)
        lines.append(f"[{ds}/{qid}] type={qlabel.get('query_type','')} core={len(core)} sortable={len(sortable_core)} 候选={len(r['candidates'])}")
        lines.append(f"  boost 前 top-10: {f_top10_before}")
        lines.append(f"  boost 后 top-10: {top10_after}")
        lines.append(f"  捞回 sortable 核心: {sorted(salvaged) if salvaged else '无'}")
        if core:
            lines.append(f"  core recall@10: {core_recall_before[-1]:.3f} -> {core_recall_after[-1]:.3f}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("汇总")
    lines.append("=" * 80)
    salvaged_rate = sum_salvaged / sum_sortable if sum_sortable else 0.0
    lines.append(f"指标1 核心捞回率: {sum_salvaged}/{sum_sortable} = {salvaged_rate:.3f}")
    if core_recall_before:
        lines.append(f"指标2 core recall@10: {sum(core_recall_before)/len(core_recall_before):.4f} -> "
                     f"{sum(core_recall_after)/len(core_recall_after):.4f} "
                     f"(delta {sum(core_recall_after)/len(core_recall_after)-sum(core_recall_before)/len(core_recall_before):+.4f})")
    else:
        lines.append("指标2 core recall@10: 无有效核心")
    side_rate = side_lost / side_total if side_total else 0.0
    lines.append(f"指标3 副作用: 挤出 {side_lost}/{side_total} = {side_rate:.3f}")

    txt = RESULTS / "boost_eval2.txt"
    txt.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"\n[OUT] {txt}", flush=True)


if __name__ == "__main__":
    main()
