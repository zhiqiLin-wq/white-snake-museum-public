"""基于别名字典的查询扩展器。

无 LLM 依赖，纯字典匹配。在查询文本中检测已知实体
（角色/地点/事件/概念），用已知别名展开，提升 BM25 关键词命中率。

支持加权扩展: 原始查询词权重 1.0，别名权重 0.5，
避免别名过度匹配导致排序退化。

Usage:
    from server.rag.query_expander import QueryExpander
    expander = QueryExpander()
    result = expander.expand("白娘子的丈夫是谁")
    # result["expanded_query"] -> "白娘子的丈夫是谁 白素贞 白氏 白蛇"
    result = expander.expand_weighted("白娘子的丈夫是谁")
    # result["token_weights"] -> {"白娘子": 1.0, "白素贞": 0.5, ...}
"""

import logging
from typing import Dict, List, Optional, Set

from ..utils.entity_names import (
    CHARACTER_ALIASES,
    LOCATION_ALIASES,
    EVENT_ALIASES,
    CONCEPT_ALIASES,
    build_reverse_index,
    detect_entities_in_query,
)

logger = logging.getLogger(__name__)

# 别名权重 (原始查询词权重 = 1.0)
ALIAS_WEIGHT = 0.5
# 每个实体最多展开的别名数
MAX_ALIASES_PER_ENTITY = 3


class QueryExpander:
    """基于别名字典的查询扩展器。

    在查询文本中检测已知实体，自动展开所有别名变体，
    缩小现代白话查询与文言文原文之间的词汇差距。

    同步方法：无 LLM 依赖，延迟可控。
    """

    def __init__(self):
        self._char_index = build_reverse_index(CHARACTER_ALIASES)
        self._loc_index = build_reverse_index(LOCATION_ALIASES)
        self._event_index = build_reverse_index(EVENT_ALIASES)
        self._concept_index = build_reverse_index(CONCEPT_ALIASES)

        # 按类别索引别名集，用于快速展开
        self._char_aliases = self._build_alias_sets(CHARACTER_ALIASES)
        self._loc_aliases = self._build_alias_sets(LOCATION_ALIASES)
        self._event_aliases = self._build_alias_sets(EVENT_ALIASES)
        self._concept_aliases = self._build_alias_sets(CONCEPT_ALIASES)

    @staticmethod
    def _build_alias_sets(
        alias_dict: Dict[str, List[str]],
    ) -> Dict[str, Set[str]]:
        """将别名字典转为 {canonical: {all_aliases}} 的集合形式。"""
        return {canonical: set(aliases) for canonical, aliases in alias_dict.items()}

    def expand(self, query: str) -> dict:
        """检测查询中的实体并用别名展开。

        按别名长度降序匹配，确保优先命中更长的具体别名。
        展开时将所有已知别名追加到查询末尾，以提升 BM25 关键词匹配。

        Args:
            query: 原始查询文本

        Returns:
            {
                "expanded_query": "展开后的查询字符串",
                "detected_entities": {
                    "characters": {"白素贞", ...},
                    "locations": {"雷峰塔", ...},
                    "events": {"水漫金山", ...},
                    "concepts": {"修炼", ...},
                },
                "added_aliases": ["白氏", "白云仙姑", ...],
            }
        """
        if not query or not query.strip():
            return {
                "expanded_query": query or "",
                "detected_entities": {
                    "characters": set(),
                    "locations": set(),
                    "events": set(),
                    "concepts": set(),
                },
                "added_aliases": [],
            }

        # 检测已知实体
        detected = detect_entities_in_query(
            query,
            char_index=self._char_index,
            loc_index=self._loc_index,
            event_index=self._event_index,
            concept_index=self._concept_index,
        )

        # 收集所有检测到的实体的别名
        all_aliases_to_add: List[str] = []
        seen_aliases: Set[str] = set()

        # 角色别名
        for canonical in detected["characters"]:
            aliases = self._char_aliases.get(canonical, set())
            for a in aliases:
                if a not in seen_aliases and a not in query:
                    all_aliases_to_add.append(a)
                    seen_aliases.add(a)

        # 地点别名
        for canonical in detected["locations"]:
            aliases = self._loc_aliases.get(canonical, set())
            for a in aliases:
                if a not in seen_aliases and a not in query:
                    all_aliases_to_add.append(a)
                    seen_aliases.add(a)

        # 事件别名
        for canonical in detected["events"]:
            aliases = self._event_aliases.get(canonical, set())
            for a in aliases:
                if a not in seen_aliases and a not in query:
                    all_aliases_to_add.append(a)
                    seen_aliases.add(a)

        # 概念别名
        for canonical in detected["concepts"]:
            aliases = self._concept_aliases.get(canonical, set())
            for a in aliases:
                if a not in seen_aliases and a not in query:
                    all_aliases_to_add.append(a)
                    seen_aliases.add(a)

        if all_aliases_to_add:
            expanded = query + " " + " ".join(all_aliases_to_add)
        else:
            expanded = query

        return {
            "expanded_query": expanded,
            "detected_entities": detected,
            "added_aliases": all_aliases_to_add,
        }

    # ------------------------------------------------------------------
    # 加权扩展 (CU-40 增强): 智能别名选择 + 词权重
    # ------------------------------------------------------------------
    def _get_aliases_for(self, canonical: str) -> set[str]:
        """查找实体 canonical 名称对应的所有别名集合。

        依次搜索角色/地点/事件/概念词典，返回找到的别名集。
        未找到返回空集。
        """
        for alias_dict in (self._char_aliases, self._loc_aliases,
                           self._event_aliases, self._concept_aliases):
            if canonical in alias_dict:
                return alias_dict[canonical]
        return set()

    @staticmethod
    def _select_best_aliases(
        aliases: set[str],
        query: str,
        max_count: int = MAX_ALIASES_PER_ENTITY,
    ) -> list[str]:
        """从别名集合中智能选取最有区分度的 N 个。

        选取策略:
        1. 排除已在原查询中出现的别名（避免重复）
        2. 按长度降序排列（中文中越长的别名越具体，IDF 越高）
        3. 排除单字别名（太短，几乎无区分度）
        4. 取前 max_count 个

        Args:
            aliases: 实体全部别名集合
            query: 原始查询文本
            max_count: 最多返回的别名数

        Returns:
            精选后的别名列表
        """
        candidates = [
            a for a in aliases
            if a not in query          # 不在原查询中
            and len(a) >= 2            # 至少二字
        ]
        # 按长度降序: 越长越具体
        candidates.sort(key=len, reverse=True)
        return candidates[:max_count]

    def _collect_aliases_weighted(
        self, canonical: str, query: str,
    ) -> list[str]:
        """查找并精选某实体的别名。

        Args:
            canonical: 实体规范名称
            query: 原始查询

        Returns:
            精选后的别名列表
        """
        aliases = self._get_aliases_for(canonical)
        if not aliases:
            return []
        return self._select_best_aliases(aliases, query)

    def expand_weighted(self, query: str) -> dict:
        """加权查询扩展：智能别名选择 + 词权重。

        与 expand() 的区别:
        - 每实体最多选 3 个最有区分度的别名（非全量追加）
        - 返回 token_weights 字典: 原词 = 1.0，别名 = 0.5
        - 通过 jieba 分词将权重映射到实际检索 token

        Args:
            query: 原始查询文本

        Returns:
            {
                "expanded_query": "展开后的查询字符串",
                "token_weights": {"原词": 1.0, "别名词": 0.5, ...},
                "detected_entities": {...},
                "added_aliases": [...],
            }
        """
        if not query or not query.strip():
            return {
                "expanded_query": query or "",
                "token_weights": {},
                "detected_entities": {
                    "characters": set(), "locations": set(),
                    "events": set(), "concepts": set(),
                },
                "added_aliases": [],
            }

        # 检测已知实体 (与 expand() 相同)
        detected = detect_entities_in_query(
            query,
            char_index=self._char_index,
            loc_index=self._loc_index,
            event_index=self._event_index,
            concept_index=self._concept_index,
        )

        # 智能别名收集: 每实体最多 3 个，过长度的优先
        all_aliases_to_add: list[str] = []
        seen_aliases: set[str] = set()

        for category, canonical_set in [
            ("characters", detected["characters"]),
            ("locations", detected["locations"]),
            ("events", detected["events"]),
            ("concepts", detected["concepts"]),
        ]:
            for canonical in canonical_set:
                best = self._collect_aliases_weighted(canonical, query)
                for a in best:
                    if a not in seen_aliases and a not in query:
                        all_aliases_to_add.append(a)
                        seen_aliases.add(a)

        if all_aliases_to_add:
            expanded = query + " " + " ".join(all_aliases_to_add)
        else:
            expanded = query

        # 构建 token 权重 (通过 jieba 分词映射)
        token_weights = self._build_token_weights(query, all_aliases_to_add)

        return {
            "expanded_query": expanded,
            "token_weights": token_weights,
            "detected_entities": detected,
            "added_aliases": all_aliases_to_add,
        }

    def _build_token_weights(
        self, query: str, aliases: list[str],
    ) -> dict[str, float]:
        """用 jieba 分词构建 token 权重映射。

        原查询 token -> weight=1.0
        别名 token -> weight=0.5 (不覆盖已存在的原查询 token)

        加载与 BM25 相同的自定义词典，确保分词结果一致。
        """
        try:
            import jieba
        except ImportError:
            return {}

        # 加载与 BM25 一致的自定义词典，确保分词一致
        # 动态从 entity_names 导入，与 bm25_retriever.py 共享同一数据源
        try:
            from ..utils.entity_names import (
                CHARACTER_ALIASES, LOCATION_ALIASES, EVENT_ALIASES, CONCEPT_ALIASES,
            )
            _jieba_words: set[str] = set()
            for alias_dict in (CHARACTER_ALIASES, LOCATION_ALIASES, EVENT_ALIASES, CONCEPT_ALIASES):
                for aliases in alias_dict.values():
                    for a in aliases:
                        a = a.strip()
                        if len(a) >= 2:
                            _jieba_words.add(a)
            # 额外文献术语
            for t in ['义妖传', '太平广记', '西湖三塔记', '警世通言', '雷峰塔传奇',
                       '方成培', '陈遇乾', '田汉', '玉山堂主人', '冯梦龙',
                       '西江月', '鹧鸪天', '临江仙', '满江红', '蝶恋花',
                       '回目', '楔子', '入话', '头回', '正话',
                       '话说', '且说', '却说', '正是', '诗曰', '词云', '赞曰']:
                _jieba_words.add(t)
        except ImportError:
            _jieba_words = {
                '白素贞', '白娘子', '白蛇', '许宣', '许汉文', '许仙',
                '小青', '法海', '雷峰塔', '断桥', '西湖', '金山寺',
            }
        for word in _jieba_words:
            jieba.add_word(word)

        weights: dict[str, float] = {}

        # 原查询 token: weight 1.0
        for token in jieba.lcut(query):
            token = token.strip()
            if token:
                weights[token] = 1.0

        # 别名 token: weight 0.5 (不覆盖原词)
        for alias in aliases:
            for token in jieba.lcut(alias):
                token = token.strip()
                if token and token not in weights:
                    weights[token] = ALIAS_WEIGHT

        return weights
