"""B-122: 实体共现矩阵计算。

使用滑动窗口计算实体在段落中的共现关系。
支持缓存（TTL 1 小时）。
"""
import hashlib
import json
import logging
import time
from typing import Optional

from ..rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)

# 检索元数据的 chapter_number 可能是中文数字字符串（Node.js API 风格 "七"）
_CHINESE_NUM_MAP = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
    "八": 8, "九": 9, "十": 10,
}


def _safe_int(value) -> int:
    """章节/段落号健壮转换：支持 int、阿拉伯数字、中文数字，意外值降级 0。"""
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if s.isdigit():
        return int(s)
    if s in _CHINESE_NUM_MAP:
        return _CHINESE_NUM_MAP[s]
    logger.debug(f"cooccurrence: 无法解析章节号 {value!r}，降级为 0")
    return 0

# 缓存 TTL: 1 小时
COOCCURRENCE_CACHE_TTL = 3600


class CooccurrenceCalculator:
    """实体共现矩阵计算器。"""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self._cache: dict[str, tuple[float, dict]] = {}

    async def compute(self, entities: list[str],
                      window_size: int = 10) -> dict:
        """计算实体共现矩阵。

        Args:
            entities: 实体名称列表，如 ["白素贞", "许仙", "法海"]
            window_size: 共现窗口（前后 N 段）

        Returns:
            {"entities": [...], "matrix": [[count, ...], ...]}
        """
        n = len(entities)
        if n == 0:
            return {"entities": [], "matrix": []}
        if n == 1:
            return {"entities": entities, "matrix": [[1]]}

        # 缓存检查
        cache_key = _cache_key(entities, window_size)
        if cache_key in self._cache:
            ts, result = self._cache[cache_key]
            if time.monotonic() - ts < COOCCURRENCE_CACHE_TTL:
                return result

        # 初始化矩阵
        matrix = [[0] * n for _ in range(n)]

        # 对每个实体检索段落
        entity_paragraphs: dict[str, list[tuple[int, int]]] = {}
        for entity in entities:
            docs = await self.retriever.retrieve(entity, top_k=50)
            paragraphs = []
            for d in docs:
                meta = d.get("metadata", {})
                ch = _safe_int(meta.get("chapter_number", 0))
                pi = _safe_int(meta.get("paragraph_index", 0))
                paragraphs.append((ch, pi))
            entity_paragraphs[entity] = paragraphs

        # 计算共现：两个实体在同一窗口内出现
        for i in range(n):
            for j in range(i + 1, n):
                count = self._count_cooccurrence(
                    entity_paragraphs[entities[i]],
                    entity_paragraphs[entities[j]],
                    window_size,
                )
                matrix[i][j] = count
                matrix[j][i] = count

        # 对角线设为实体总出现次数
        for i in range(n):
            matrix[i][i] = len(entity_paragraphs[entities[i]])

        result = {"entities": entities, "matrix": matrix}
        self._cache[cache_key] = (time.monotonic(), result)
        return result

    @staticmethod
    def _count_cooccurrence(paras_a: list[tuple[int, int]],
                            paras_b: list[tuple[int, int]],
                            window_size: int) -> int:
        """计算两个实体在同一窗口内的共现次数。"""
        count = 0
        for ch_a, pi_a in paras_a:
            for ch_b, pi_b in paras_b:
                if ch_a == ch_b and abs(pi_a - pi_b) <= window_size:
                    count += 1
        return count


def _cache_key(entities: list[str], window_size: int) -> str:
    raw = json.dumps({"e": sorted(entities), "w": window_size}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()
