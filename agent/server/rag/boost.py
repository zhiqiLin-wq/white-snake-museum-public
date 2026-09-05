"""boost 重排规则（最终版，严格按 rag_eval/RETRIEVAL_RULES.md）。

fused_base = 0.5 * minmax(dense) + 0.5 * minmax(bm25)
boost = 各层累加（按 query_type 分流 + chunk 双轨 + IDF 加权 + summary 双路）
final = (fused_base + boost) * genre因子；hard_match 加 HARD_OFFSET 前置。
"""
import re

HARD_OFFSET = 10.0
PLOT_DETAIL_WEIGHT = 0.20
PERSON_WEIGHT = 0.10
LOCATION_WEIGHT = 0.09
RAW_WEIGHT = 0.05
CITE_PLOT_WEIGHT = 0.05
VERSION_WEIGHT = 0.05
SUMMARY_DENSE_WEIGHT = 0.10
SUMMARY_SPARSE_WEIGHT = 0.10
GENRE_FACTOR = 1.2
IDF_THRESHOLD = 0.8  # 人物/地点硬分组的高频实体过滤阈值
PLOT_UNIT_SIM_THRESHOLD = 0.75  # plot_unit 语义匹配（embedding 余弦）阈值

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

_QT_ALIAS = {"comparison": "comparison题", "情节": "情节题", "人物": "人物题",
             "地点": "地点题", "研究分析": "研究分析题"}


def _cset(items):
    return {x["canonical"] for x in items if isinstance(x, dict) and x.get("canonical")}


def _rset(items):
    return {x["raw"] for x in items if isinstance(x, dict) and x.get("raw")}


def normalize_qtype(qtype):
    return _QT_ALIAS.get(qtype, qtype)


def _bigrams(s):
    s = re.sub(r"\s+", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def plot_detail_sim(a, b):
    if not a or not b:
        return 0.0
    ba, bb = _bigrams(a), _bigrams(b)
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)


def summary_jaccard(a, b):
    """jieba 分词的 Jaccard 词面重叠。"""
    if not a or not b:
        return 0.0
    try:
        import jieba
        ta = set(jieba.lcut(a))
        tb = set(jieba.lcut(b))
    except ImportError:
        ta = _bigrams(a)
        tb = _bigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


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


def _plot_semantic_match(q_plot, c_plot, plot_unit_sim, threshold):
    """plot_unit 语义匹配（embedding 余弦），解决归并不一致的问题。"""
    if plot_unit_sim is None:
        return False
    for qp in q_plot:
        for cp in c_plot:
            if plot_unit_sim(qp, cp) >= threshold:
                return True
    return False


def compute_boost(q, c_norm, c_genre, summary_dense_sim, summary_sparse_jac, idf, plot_unit_sim=None,
                  plot_unit_threshold=PLOT_UNIT_SIM_THRESHOLD):
    """计算标签 boost。返回 (boost, hard_match)。"""
    boost = 0.0
    hard_match = False
    qtype = normalize_qtype(q.get("query_type", ""))
    q_persons = _cset(q.get("persons", []))
    q_persons_raw = _rset(q.get("persons", []))
    q_locs = _cset(q.get("locations", []))
    q_plot = _cset(q.get("plot_unit", []))
    q_plot_detail = q.get("plot_detail", "") or ""
    q_topic = _cset(q.get("分析主题", []))
    q_cmp = q.get("对比维度", {}) or {}

    if c_genre == "research_literature":
        c_topic = _cset(c_norm.get("分析主题", []))
        c_vers = set(c_norm.get("涉及版本", []) or [])
        ent = c_norm.get("引用实体", {}) or {}
        c_ent_persons = _cset(ent.get("persons", []))
        c_ent_locs = _cset(ent.get("locations", []))
        c_plot = _cset(c_norm.get("引用情节", []))

        if qtype in ("comparison题", "研究分析题"):
            if q_topic and c_topic & q_topic:
                hard_match = True
        if qtype == "comparison题" and cmp_has_version_dim(q_cmp):
            if any(version_matches(g, q_cmp) for g in c_vers):
                boost += VERSION_WEIGHT
        if qtype in ("comparison题", "研究分析题"):
            if q_plot & c_plot or _plot_semantic_match(q_plot, c_plot, plot_unit_sim, plot_unit_threshold):
                hard_match = True
        if qtype == "情节题":
            boost += CITE_PLOT_WEIGHT * len(q_plot & c_plot)
        if qtype in ("情节题", "人物题", "地点题", "研究分析题"):
            boost += PERSON_WEIGHT * sum(idf(p) for p in (q_persons & c_ent_persons))
            boost += LOCATION_WEIGHT * sum(idf(l) for l in (q_locs & c_ent_locs))
    else:
        c_plot = _cset(c_norm.get("plot_unit", []))
        c_plot_detail = c_norm.get("plot_detail", "") or ""
        c_persons = _cset(c_norm.get("persons", []))
        c_persons_raw = _rset(c_norm.get("persons", []))
        c_locs = _cset(c_norm.get("locations", []))

        if qtype == "情节题":
            if q_plot & c_plot or _plot_semantic_match(q_plot, c_plot, plot_unit_sim, plot_unit_threshold):
                hard_match = True
            boost += PLOT_DETAIL_WEIGHT * plot_detail_sim(q_plot_detail, c_plot_detail)
        if qtype in ("人物题", "地点题"):
            # persons/locations 硬分组（只对高 IDF 低频实体，避免高频实体引入噪声）
            if any(idf(p) >= IDF_THRESHOLD for p in (q_persons & c_persons)):
                hard_match = True
            if any(idf(l) >= IDF_THRESHOLD for l in (q_locs & c_locs)):
                hard_match = True
        if qtype in ("情节题", "人物题", "地点题", "comparison题"):
            boost += PERSON_WEIGHT * sum(idf(p) for p in (q_persons & c_persons))
            boost += LOCATION_WEIGHT * sum(idf(l) for l in (q_locs & c_locs))
        if qtype == "comparison题" and cmp_has_appellation(q_cmp):
            boost += RAW_WEIGHT * len(q_persons_raw & c_persons_raw)

    # 版本元信息（comparison 题）
    if qtype == "comparison题" and cmp_has_version_dim(q_cmp):
        if c_genre in GENRE_META and version_matches(c_genre, q_cmp):
            boost += VERSION_WEIGHT

    # summary 双路（dense 语义 + sparse 词面）
    boost += SUMMARY_DENSE_WEIGHT * summary_dense_sim
    boost += SUMMARY_SPARSE_WEIGHT * summary_sparse_jac

    return boost, hard_match
