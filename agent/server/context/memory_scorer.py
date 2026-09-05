"""记忆综合评分计算。

公式: Score = w1 * Relevance_norm + w2 * Importance_norm + w3 * Recency_norm

权重:
- w1 (Relevance / 检索相关性): 0.5
- w2 (Importance / 绝对重要性): 0.3
- w3 (Recency / 时间衰减):  0.2

所有维度在计算前做 Min-Max 归一化到 [0, 1]。

衰减周期（从创建起算）:
- 偏好类 (Entity): 90 天
- 事实类 (Archival): 60 天
- 对话摘要类 (Summary): 14 天
- 临时状态类 (Working): 7 天
- 系统约束类 (System): 不衰减
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

_BEIJING_TZ = timezone(timedelta(hours=8))

# 衰减周期（天）
DECAY_DAYS = {
    "Entity": 90,
    "Archival": 60,
    "Summary": 14,
    "Working": 7,
    "System": 365 * 10,  # 几乎不衰减
}

# tombstone 清理：分类型 importance 保底（豁免线）+ TTL（未访问天数）。
# 清理规则: System 永不删；importance >= 保底 → 保留；否则未访问 > TTL → tombstone。
# 取代旧的「统一 score < 0.05」判定（后者因 0.3*importance 保底 0.12 恒高于阈值而永不触发）。
IMPORTANCE_FLOOR = {
    "System": 0.90,
    "Entity": 0.60,
    "Archival": 0.40,
    "Summary": 0.30,
    "Working": 0.30,
}
TTL_DAYS = {
    "System": None,   # 永不清理
    "Entity": 180,
    "Archival": 90,
    "Summary": 30,
    "Working": 7,
}

# 评分权重
W_RELEVANCE = 0.5
W_IMPORTANCE = 0.3
W_RECENCY = 0.2


class MemoryScorer:
    """计算记忆的综合评分。"""

    def __init__(self, memory_db=None):
        """
        Args:
            memory_db: MemoryDB 实例（可选，用于批量降权时读取记忆）
        """
        self._db = memory_db

    def compute_score(self,
                      relevance_raw: float,
                      importance_raw: float,
                      memory_type: str = "Entity",
                      created_at: Optional[str] = None,
                      last_accessed_at: Optional[str] = None) -> float:
        """计算单条记忆的综合评分。

        Args:
            relevance_raw: 检索相似度 (0-1)
            importance_raw: 重要性 (0-1)
            memory_type: 记忆类型
            created_at: 创建时间（ISO 8601）
            last_accessed_at: 最后访问时间（ISO 8601）

        Returns:
            综合评分 (0-1)
        """
        recency_raw = self._compute_recency(memory_type, created_at, last_accessed_at)
        return (
            W_RELEVANCE * relevance_raw
            + W_IMPORTANCE * importance_raw
            + W_RECENCY * recency_raw
        )

    def compute_scores_batch(self,
                              memories: list[dict],
                              relevance_key: str = "score",
                              importance_key: str = "importance",
                              memory_type_key: str = "memory_type",
                              created_at_key: str = "created_at",
                              last_accessed_key: str = "last_accessed_at") -> list[dict]:
        """批量计算记忆评分（自动 Min-Max 归一化）。

        Args:
            memories: 记忆列表
            relevance_key: 检索相关性字段名
            importance_key: 重要性字段名
            memory_type_key: 记忆类型字段名
            created_at_key: 创建时间字段名
            last_accessed_key: 最后访问时间字段名

        Returns:
            带 computed_score 字段的记忆列表（不改变入参）
        """
        if not memories:
            return []

        # 收集原始值
        relevances = []
        importances = []
        recencies = []

        for m in memories:
            r = float(m.get(relevance_key, 0) or 0)
            i = float(m.get(importance_key, 0.5) or 0.5)
            mt = str(m.get(memory_type_key, "Entity"))
            ca = m.get(created_at_key)
            la = m.get(last_accessed_key)
            rec = self._compute_recency(mt, ca, la)

            relevances.append(r)
            importances.append(i)
            recencies.append(rec)

        # Min-Max 归一化
        r_norm = self._min_max_normalize(relevances)
        i_norm = self._min_max_normalize(importances)
        rec_norm = self._min_max_normalize(recencies)

        # 计算综合评分
        result = []
        for idx, m in enumerate(memories):
            score = (
                W_RELEVANCE * r_norm[idx]
                + W_IMPORTANCE * i_norm[idx]
                + W_RECENCY * rec_norm[idx]
            )
            m_copy = dict(m)
            m_copy["computed_score"] = round(score, 4)
            m_copy["score_breakdown"] = {
                "relevance_raw": round(relevances[idx], 4),
                "relevance_norm": round(r_norm[idx], 4),
                "importance_raw": round(importances[idx], 4),
                "importance_norm": round(i_norm[idx], 4),
                "recency_raw": round(recencies[idx], 4),
                "recency_norm": round(rec_norm[idx], 4),
            }
            result.append(m_copy)

        # 按综合评分降序
        result.sort(key=lambda x: x["computed_score"], reverse=True)
        return result

    def _compute_recency(self,
                          memory_type: str,
                          created_at: Optional[str],
                          last_accessed_at: Optional[str]) -> float:
        """计算时间衰减因子。

        指数衰减模型，decay_period 为半衰期。
        访问重置机制：如果最近被访问过，从访问时刻重新算衰减。

        Returns:
            0-1 值，1 = 最新，0 = 完全衰减
        """
        decay_days = DECAY_DAYS.get(memory_type, 60)
        half_life = decay_days / 2.0
        now = datetime.now(_BEIJING_TZ)

        # 使用 last_accessed_at 或 created_at
        ref_time_str = last_accessed_at or created_at
        if ref_time_str is None:
            return 0.5  # 未知时间，取中值

        try:
            ref_time = datetime.fromisoformat(ref_time_str.replace("Z", "+00:00"))
            if ref_time.tzinfo is None:
                ref_time = ref_time.replace(tzinfo=_BEIJING_TZ)
        except (ValueError, TypeError):
            return 0.5

        delta_seconds = (now - ref_time).total_seconds()
        delta_days = max(0, delta_seconds / 86400.0)

        # 指数衰减: exp(-lambda * t), lambda = ln(2) / half_life
        import math
        if half_life <= 0:
            return 1.0
        lam = math.log(2) / half_life
        recency = math.exp(-lam * delta_days)

        return round(recency, 4)

    def _min_max_normalize(self, values: Sequence[float]) -> list[float]:
        """Min-Max 归一化到 [0, 1]。

        若所有值相同，返回全 0.5 避免除以零。
        """
        if not values:
            return []
        min_v = min(values)
        max_v = max(values)
        if max_v - min_v < 1e-9:
            return [0.5] * len(values)
        return [(v - min_v) / (max_v - min_v) for v in values]

    @staticmethod
    def days_since_access(memory: dict) -> Optional[int]:
        """计算记忆的未访问天数（基于 last_accessed_at 或 created_at）。

        Returns:
            未访问天数（>=0）；时间缺失或解析失败返回 None（跳过判定）。
        """
        ref = memory.get("last_accessed_at") or memory.get("created_at")
        if ref is None:
            return None
        try:
            ref_time = datetime.fromisoformat(ref.replace("Z", "+00:00"))
            if ref_time.tzinfo is None:
                ref_time = ref_time.replace(tzinfo=_BEIJING_TZ)
            delta = (datetime.now(_BEIJING_TZ) - ref_time).total_seconds()
            return max(0, int(delta / 86400.0))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def should_tombstone(importance: float, memory_type: str,
                          days_unaccessed: int) -> bool:
        """判断记忆是否应被 tombstone（分类型 importance 保底 + TTL）。

        Args:
            importance: 记忆重要性 (0-1)
            memory_type: 记忆类型
            days_unaccessed: 未访问天数

        Returns:
            True 如果应 tombstone
        """
        if memory_type == "System":
            return False
        floor = IMPORTANCE_FLOOR.get(memory_type, 0.5)
        if importance >= floor:
            return False
        ttl = TTL_DAYS.get(memory_type)
        if ttl is None:
            return False
        return days_unaccessed > ttl

    def get_importance_for_extraction(self, content: str,
                                       role: str = "user",
                                       memory_type: str = "Entity") -> float:
        """根据启发式规则预估 importance（类型驱动，配合 IMPORTANCE_FLOOR）。

        各类型的 importance 分层，使「高于保底（保留）」和「低于保底（可清理）」
        都能自然产生：
        - System 约束 → 0.95（恒高于保底 0.9，永不清理）
        - Entity 身份/偏好 → user 明确 0.9 / 一般 0.7 / assistant 复述 0.5
        - Archival 事实 → 明确事实 0.6 / 碎片 0.3
        - Summary 摘要 → 0.2（低于保底 0.3，靠 TTL 清理）
        - Working 临时 → 0.2（低于保底 0.3，靠 TTL 清理）

        Args:
            content: 记忆内容
            role: 消息角色
            memory_type: 记忆类型

        Returns:
            预估的重要性 (0-1)
        """
        if memory_type == "System":
            return 0.95
        if memory_type == "Entity":
            if role == "user":
                preference_keywords = ["喜欢", "不喜欢", "偏好", "最", "非常",
                                       "讨厌", "最爱", "希望", "想要"]
                for kw in preference_keywords:
                    if kw in content:
                        return 0.9
                return 0.7
            return 0.5  # assistant 复述，低于保底 0.6
        if memory_type == "Archival":
            if role == "user":
                fact_markers = ["年代", "时期", "版本", "位于", "建于",
                                "作者", "来源", "定义", "数据"]
                for kw in fact_markers:
                    if kw in content:
                        return 0.6
            return 0.3  # 碎片，低于保底 0.4
        if memory_type in ("Summary", "Working"):
            return 0.2
        return 0.5
