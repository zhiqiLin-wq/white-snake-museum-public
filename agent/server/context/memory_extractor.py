"""长期记忆提取模块（v2）。

从被弹出的 FIFO 消息中提取值得长期记忆的信息。
流程: 规则过滤 -> 角色过滤(Mem0: 仅user) -> LLM 提取 -> MD5去重 -> 向量去重 -> 双写

v2 改进:
- MD5 精确去重 + 向量去重 双重保护（防线 1+2）
- 仅从 user 消息提取长期记忆（Mem0 策略，防线 1）
- memory_type 规则分类（Entity/Working/Archival/System/Summary）
- importance 启发式赋值

分类规则:
- user消息 -> 偏好/身份/长期特征 -> Entity
- user消息 -> 客观事实知识 -> Archival
- user/assistant -> 当前会话临时状态 -> Working
- 用户明确拒绝/限制 -> System (长期约束，不衰减)
- FIFO 摘要 -> Summary
"""

import logging
import re
from typing import Optional

from .memory_scorer import MemoryScorer

logger = logging.getLogger(__name__)

# 规则过滤：纯问候/客套/噪声的正则
_NOISE_PATTERNS = [
    re.compile(r"^[你好]+[！!。.]*$"),
    re.compile(r"^[谢谢谢]+[！!。.]*$"),
    re.compile(r"^(再见|拜拜|bye|goodbye|see you)[！!。.]*$", re.IGNORECASE),
    re.compile(r"^(OK|ok|好的|嗯|哦|哈哈|呵呵|收到|明白了|知道了)[！!。.]*$"),
    re.compile(r"^[\d\s.,，。、;；:：!！?？\-—+*/=()（）\[\]【】{}]+$"),
]

# 只过滤空串，不做内容价值判断（短身份/约束如"我叫小研"也有价值，噪声交给 _NOISE_PATTERNS + LLM 判 NO）
_MIN_CONTENT_LENGTH = 1

# ---- Preference/Entity keywords ----
_ENTITY_KEYWORDS = [
    "喜欢", "不喜欢", "偏好", "讨厌", "最爱", "感兴趣", "研究",
    "我是", "我叫", "我的", "我在", "我做", "我住在", "我来自",
    "身份", "专业", "方向", "领域", "名字", "年龄", "职业",
    "希望", "想要", "经常", "通常", "总是", "从不",
    "认为", "觉得", "观点", "看法", "评价", "关注", "重视", "在意",
    "看重", "主张", "立场",
]

# ---- Archival/Fact keywords ----
_ARCHIVAL_KEYWORDS = [
    "是", "有", "位于", "建于", "年代", "时期", "版本",
    "作者", "出自", "来源", "数据", "事实", "定义",
]

# ---- Working/Temporary keywords ----
_WORKING_KEYWORDS = [
    "正在", "当前", "现在", "今天", "这次", "刚才", "刚",
    "检查", "排查", "调试", "测试", "分析中", "进行中",
]

MEMORY_EXTRACTION_SYSTEM = (
    "你是一个记忆提取助手。从对话中提取关于用户的长期事实。\n\n"
    "规则：\n"
    "1. 提取关于用户的事实（偏好、身份、知识、约束），以及用户陈述的客观知识或事实（如文献出处、版本、年代、地点等）。\n"
    "2. 从用户自己的话中提取；同时也可提取助手回复中明确确认或复述的用户偏好。\n"
    "3. 不要提取临时的、任务相关的信息，也不要提取助手回复中的文献或分析内容。\n"
    "4. 每个事实单独一行，格式为\"类型：内容\"，类型四选一：Entity（偏好/身份/观点）、Archival（客观知识/事实）、System（约束/要求/拒绝）、Working（临时状态）。\n"
    "   例如：\n"
    "   Entity: 用户偏好中国古典文学。\n"
    "   System: 用户约束：不要发送长代码。\n"
    "   Archival: 雷峰塔位于杭州西湖边。\n"
    "5. 如果用户明确拒绝某事物（例如：\"不要发送长代码\"）或提出明确要求（例如：\"引用文献必须标注出处\"\"分析要按朝代顺序\"），类型用 System，内容以\"用户约束：\"开头。\n"
    "6. 只要表达了用户的偏好、身份、知识或约束，无论措辞是否口语化、句子是否简短、语气是否是否定（如\"别发太长\"\"不喜欢长篇\"\"懒得看\"\"受不了\"），都必须提取，不要因为表达不正式、简短或否定就判 NO。\n"
    "7. 如果确认没有任何值得长期记住的内容，只输出：NO\n\n"
    "对话：{message}\n\n"
    "提取的事实（或 NO）："
)


class MemoryExtractor:
    """从被弹出消息中提取长期记忆（v2）。

    调用方应为每条被弹出的消息调用 extract()。
    仅从 user 消息中提取（Mem0 策略）。
    """

    def __init__(self, llm, recall_store, model: str):
        """
        Args:
            llm: LLMProvider 实例
            recall_store: RecallStore 实例（双写模式）
            model: 用于记忆提取的模型名（如 deepseek-v4-flash）
        """
        self._llm = llm
        self._store = recall_store
        self._model = model
        self._scorer = MemoryScorer()

    async def extract(self, message: dict) -> list[str]:
        """从一条被弹出的消息中提取长期记忆。

        Mem0 策略：仅从 user 消息提取。
        Assistant 消息不产生长期记忆。

        Args:
            message: {"role": "...", "content": "...", ...}

        Returns:
            新增的记忆陈述列表（已成功双写）
        """
        content = str(message.get("content", "")).strip()
        role = message.get("role", "unknown")
        thread_id = message.get("_thread_id", "") or message.get("thread_id", "")

        # 防线 1: Mem0 策略 — 仅提取 user 消息（忽略 assistant/system/tool）
        if role != "user":
            logger.debug("memory_extractor skipped non-user message (role=%s)", role)
            return []

        # 规则过滤
        if not self._should_extract(content):
            logger.debug("memory_extractor filtered: %s...", content[:50])
            return []

        # LLM 提取
        raw_output = await self._llm_extract(content)
        if raw_output is None or raw_output.strip().upper() == "NO":
            logger.debug("memory_extractor LLM returned NO for: %s...", content[:50])
            return []

        # 解析语句
        statements = [
            line.strip() for line in raw_output.strip().split("\n")
            if line.strip() and line.strip().upper() != "NO"
        ]
        if not statements:
            return []

        added = []
        for statement in statements:
            # 解析"类型：内容"前缀（LLM 直接分类），解析失败则 memory_type=None 待规则兜底
            memory_type, text = self._parse_typed_statement(statement)
            if len(text) < 2:
                continue

            # 防线 1: MD5 精确去重（用内容，不含类型前缀）
            if self._store.check_duplicate_md5(text):
                logger.debug("memory_extractor MD5 duplicate: %s", text[:60])
                continue

            # 防线 2: 向量去重
            if self._store.check_duplicate_vector(text):
                logger.debug("memory_extractor vector duplicate: %s", text[:60])
                continue

            # 分类：LLM 输出类型优先，缺失则规则兜底
            if memory_type is None:
                memory_type = self._classify_memory_type(role, text, content)

            # 启发式 importance
            importance = self._scorer.get_importance_for_extraction(
                text, role=role, memory_type=memory_type,
            )

            # 双写
            doc_id = self._store.add_memory(
                text,
                memory_type=memory_type,
                importance=importance,
                confidence=0.7 if importance >= 0.7 else 0.5,
                thread_id=thread_id,
            )
            if doc_id:
                added.append(text)

        if added:
            logger.info(
                "memory_extractor extracted %d memories from %s msg (%d chars): %s",
                len(added), role, len(content), added,
            )

        return added

    async def extract_batch(self, messages: list[dict]) -> list[str]:
        """批量提取记忆（用于压缩/空闲提取场景）。

        合并 user 与 assistant 消息后一次 LLM 调用提取，使助手回复里确认的
        用户偏好也能被捕捉（用户消息被规则过滤时作为兜底）。

        Args:
            messages: FIFO 消息列表

        Returns:
            新增的记忆陈述列表
        """
        thread_id = ""
        parts = []
        for m in messages:
            content = str(m.get("content", "")).strip()
            if not content:
                continue
            tid = m.get("_thread_id", "") or m.get("thread_id", "")
            if tid and not thread_id:
                thread_id = tid
            role = m.get("role")
            if role == "user":
                if self._should_extract(content):
                    parts.append(f"[用户]: {content}")
            elif role == "assistant":
                # 助手回复只取开头（确认/复述部分），避免大量文献内容干扰提取
                parts.append(f"[助手]: {content[:300]}")

        if not parts:
            return []

        combined = "\n---\n".join(parts)
        return await self.extract({"role": "user", "content": combined, "_thread_id": thread_id})

    # ================================================================
    #  规则
    # ================================================================

    def _should_extract(self, content: str) -> bool:
        """规则过滤：判断消息是否值得提取记忆。"""
        if len(content) < _MIN_CONTENT_LENGTH:
            return False
        for pattern in _NOISE_PATTERNS:
            if pattern.match(content):
                return False
        return True

    _VALID_TYPES = ("Entity", "Archival", "System", "Working", "Summary")

    def _parse_typed_statement(self, statement: str):
        """解析"类型：内容"前缀，返回 (type, content)。

        支持中文冒号"："和英文冒号":"。无类型前缀时返回 (None, 原文)。
        """
        for t in self._VALID_TYPES:
            for colon in ("：", ":"):
                prefix = t + colon
                if statement.startswith(prefix):
                    return t, statement[len(prefix):].strip()
        return None, statement

    def _classify_memory_type(self, role: str, statement: str,
                               original_content: str = "") -> str:
        """规则分类 memory_type。

        规则优先级:
        1. 包含约束/拒绝关键词 -> Entity (作为 constraints)
        2. 包含当前/临时关键词 -> Working
        3. 包含偏好/身份关键词 -> Entity
        4. 包含事实性关键词 -> Archival
        5. user 消息默认 -> Entity
        6. assistant 消息 -> Working
        """
        # 约束检测（用户明确拒绝/限制）→ System（长期约束，不衰减）
        # 主分类已改为 LLM 直接输出类型，本规则仅作兜底
        constraint_keywords = ["不要", "禁止", "拒绝", "不能", "别", "不许", "不想"]
        for kw in constraint_keywords:
            if kw in statement:
                return "System"

        # 临时状态
        for kw in _WORKING_KEYWORDS:
            if kw in statement:
                return "Working"

        # 偏好/身份
        for kw in _ENTITY_KEYWORDS:
            if kw in statement:
                return "Entity"

        # 事实知识
        for kw in _ARCHIVAL_KEYWORDS:
            if kw in statement and role == "user":
                return "Archival"

        # 默认
        if role == "user":
            return "Entity"
        return "Working"

    # ================================================================
    #  LLM
    # ================================================================

    async def _llm_extract(self, content: str) -> Optional[str]:
        """调用 LLM 提取记忆陈述。LLM 偶发返回空时重试（最多 3 次）。"""
        user_prompt = MEMORY_EXTRACTION_SYSTEM.format(message=content)
        for attempt in range(3):
            try:
                resp = await self._llm.generate(
                    system="你是一个记忆提取助手。只提取长期的用户事实。",
                    user=user_prompt,
                    model=self._model,
                    max_tokens=1024,
                    temperature=0.3,
                )
                result = (resp.content or "").strip()
                if result:
                    return result
                logger.debug(
                    "memory_extractor LLM empty output (attempt %d): %s...",
                    attempt + 1, content[:50],
                )
            except Exception as e:
                logger.warning("memory_extractor LLM call failed: %s", e)
                return None
        return None
