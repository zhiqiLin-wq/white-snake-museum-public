"""检索意图检测 — 判断查询是否为"研究/学术意图"，用于研究文献软降权。

无 LLM 依赖，纯关键词规则，确定性与可复现（评估管线不经过 Supervisor，
必须自包含）。当查询显式表露研究/学术意图时，研究文献不降权（豁免）；
否则研究文献默认降权（系数 rag_config.research_penalty，默认 0.5）。

Usage:
    from server.rag.query_intent import research_penalty_for
    penalty = research_penalty_for("白娘子的丈夫是谁")
    # penalty -> 0.5  (无研究意图，研究文献降权)
    penalty = research_penalty_for("关于白蛇传的研究综述")
    # penalty -> 1.0  (研究意图，研究文献不降权)
"""

import logging

logger = logging.getLogger(__name__)

# 研究/分析意图触发词。命中任一即视为研究意图 → 研究文献豁免降权。
# 由 rag_eval 全量 query 统计调研得出：在"答案主要是研究文献"的 query 里
# 高频且高精度（该词命中的 query 中，研究答案占比 >= 0.7）的词。
# 已剔除低精度词（新增/三教合一/伦理/体现/差异/何以/如何/变化）与情节词（生子/祭塔）。
RESEARCH_INTENT_KEYWORDS = (
    # 学术核心词
    "研究", "学术", "论文", "学者", "综述", "文学", "文献", "理论",
    "分析", "论证", "统计", "人类学",
    # 演变/变化词
    "演变", "流变", "强化", "弱化", "重塑", "突显", "转型", "转向",
    "取代", "附着", "移位", "转化", "更名", "演绎",
    # 叙事/文本/文化词
    "叙事", "文本", "景观", "符号", "文化", "地理", "地域", "记忆",
    "民间", "关联", "建构", "内涵", "功能", "属性",
    # 宗教/形象词
    "宗教", "宗教惩戒", "儒释道", "象征", "象征意义",
    "形象", "法海形象", "高僧", "观念", "寺庙",
    # 戏曲结构词
    "增设", "折子戏",
)

# 研究文献降权系数由 rag_config.research_penalty 提供（默认 0.5，
# 可由环境变量 RESEARCH_PENALTY 覆盖）。此处不再硬编码，见 server/rag/config.py。


def detect_research_intent(query: str) -> bool:
    """检测查询是否显式表露研究/学术意图。

    Args:
        query: 原始用户查询文本

    Returns:
        True 表示研究意图 → 研究文献不降权（豁免）。
    """
    if not query:
        return False
    return any(kw in query for kw in RESEARCH_INTENT_KEYWORDS)


def research_penalty_for(query: str) -> float:
    """返回研究文献的降权系数。

    Args:
        query: 原始用户查询文本

    Returns:
        研究意图 → 1.0（不降权）；否则 → rag_config.research_penalty。
    """
    if detect_research_intent(query):
        return 1.0
    from .config import rag_config
    return rag_config.research_penalty


# ============================================================================
# 版本/朝代/体裁约束检测 — 输出版本约束（供软 boost，非硬过滤）
# ============================================================================
# 当查询明确指向单一版本（体裁/朝代/具体作品）时，返回版本约束字典
# （{"genre": ...} / {"dynasty": ...}），供检索层做软 boost：匹配版本的
# chunk 融合分乘 >1 系数上浮，别版本不屏蔽不降。多版本对比查询通过
# 信号冲突检测直接放弃约束，避免误伤。
# 注：硬过滤（完全屏蔽）已证实在跨版本 auto-qrels 上净负（zero 翻倍），
# 故改用软 boost（只升不降，温和系数，见 config.version_boost_factor）。

# 具体作品/体裁规则：(关键词元组, dynasty, genre)。命中即同时锁朝代+体裁。
VERSION_GENRE_RULES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("方成培", "水竹居"), "清", "opera_libretto"),
    (("黄图珌",), "清", "opera_libretto"),
    (("冯梦龙", "三言本", "白娘子永镇雷峰塔", "警世通言"), "明", "imitation_huaben"),
    (("章回", "玉山堂主人"), "清", "chapter_vernacular"),
    (("弹词",), "清", "tanci"),
    (("田汉", "京剧"), "现当代", "script"),
    (("太平广记", "李黄"), "唐", "classical_chinese"),
)

# 同名作品（书名歧义）：书名出现但未指定作者时，映射到多本（OR 过滤）。
# 语料中《雷峰塔传奇》有两本：方成培戏曲(opera_libretto) + 玉山堂主人白话(chapter_vernacular)。
AMBIGUOUS_TITLE_GENRES: tuple[tuple[tuple[str, ...], str, tuple[str, ...]], ...] = (
    (("雷峰塔传奇",), "清", ("opera_libretto", "chapter_vernacular")),
)

# 仅朝代词（未命中具体作品时，退而按朝代过滤）
VERSION_DYNASTY_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("现当代", "现当代"),
    ("清代", "清"), ("清朝", "清"),
    ("明代", "明"), ("明朝", "明"),
    ("宋代", "宋"), ("宋朝", "宋"),
    ("唐代", "唐"), ("唐朝", "唐"),
)


def detect_version_filters(query: str) -> dict | None:
    """检测查询明确指向的朝代/体裁，返回版本约束字典（供软 boost）。

    语义：
    - 单版本明确指定（"方成培《雷峰塔》"）→ 返回 {"genre": str}，boost 该版本。
    - 同名书名未指定作者（"《雷峰塔传奇》"）→ 返回 {"genre": [str, ...]}，OR boost 多本。
    - 无版本信号（"白蛇传里"）→ 返回 None，不做版本 boost。
    - 多版本对比（"方成培 vs 冯梦龙"）→ 多朝代或多精确体裁信号冲突 → 返回 None。

    Args:
        query: 原始用户查询文本

    Returns:
        {"genre": str} 或 {"genre": [str, ...]} 或 {"dynasty": str}。
        genre 优先于 dynasty；value 为 list 表示 OR 匹配多个 genre。
        无约束/多版本冲突 → None。
    """
    if not query:
        return None

    dynasties: set[str] = set()
    genres: set[str] = set()  # 精确确定的 genre（作者名/唯一书名）
    ambiguous_genres: list[tuple[str, ...]] = []  # 歧义书名的候选 genre 集合

    for keywords, dynasty, genre in VERSION_GENRE_RULES:
        if any(k in query for k in keywords):
            dynasties.add(dynasty)
            genres.add(genre)

    for keywords, dynasty, genre_list in AMBIGUOUS_TITLE_GENRES:
        if any(k in query for k in keywords):
            dynasties.add(dynasty)
            ambiguous_genres.append(genre_list)

    for keyword, dynasty in VERSION_DYNASTY_KEYWORDS:
        if keyword in query:
            dynasties.add(dynasty)

    # 多朝代 或 多精确体裁 冲突 → 跨版本比较类查询，放弃过滤
    if len(dynasties) > 1 or len(genres) > 1:
        return None

    # 精确 genre 优先（作者名/唯一书名 覆盖 歧义书名）
    if len(genres) == 1:
        return {"genre": next(iter(genres))}

    # 无精确 genre 但命中歧义书名 → 映射到多本（OR 过滤）
    if ambiguous_genres:
        return {"genre": list(ambiguous_genres[0])}

    if len(dynasties) == 1:
        return {"dynasty": next(iter(dynasties))}
    return None
