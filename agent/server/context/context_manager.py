"""FIFO 上下文队列管理 + 递归摘要压缩 + SQLite 持久化 (v15)。

核心职责:
1. 维护每个 thread_id 的 FIFO 消息队列（固定 token 上限）
2. 超限时自动: 弹出 -> 存储到 RecallStore -> 提取长期记忆 -> 更新摘要
3. 70% 压力警告: 提醒 LLM 主动保存关键信息到 Core/Archival Memory
4. 空闲检测: 检测 idle > 30min -> 触发后台异步提取
5. 预热提取: 新用户首次对话结束时触发快速画像建立
6. 持久化到 agent_checkpoints.db 的 context_state 表
7. 构建上下文窗口（system summary + FIFO 队列 + passive injection）供 LLM 消费
"""

import asyncio
import json
import logging
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

_BEIJING_TZ = timezone(timedelta(hours=8))


def _now() -> datetime:
    return datetime.now(_BEIJING_TZ)
from pathlib import Path
from typing import Optional

from .token_counter import count_messages_tokens, count_message_tokens
from ..config import settings

logger = logging.getLogger(__name__)

# 压力警告阈值
PRESSURE_WARN_RATIO = 0.70
# 空闲提取阈值（分钟）
IDLE_EXTRACTION_MINUTES = 30

SUMMARY_UPDATE_SYSTEM = (
    "你是一个对话摘要助手。你的任务是在旧消息被移除时，增量更新对话摘要。"
)

SUMMARY_UPDATE_USER = (
    "这是之前的对话摘要：\n"
    "{existing_summary}\n\n"
    "以下消息已从对话历史中移除以节省空间：\n"
    "{messages}\n\n"
    "创建一个新的、完整的摘要，合并以上所有内容。\n"
    "优先级规则：\n"
    "1. 最高优先级保留关于用户的长期信息：身份、偏好、观点、约束"
    "（如\"用户喜欢X\"\"用户认为X\"\"用户要求X\"\"用户不喜欢X\"），"
    "必须逐条完整列出，一条都不能漏，宁可长也不要丢信息。\n"
    "2. 客观文献知识、情节细节等背景内容可以简要概括，但用户信息必须完整保留。\n"
    "只输出新的摘要文本。"
)

EMPTY_SUMMARY = ""


@dataclass
class ContextState:
    """一个对话线程的上下文状态。"""
    thread_id: str
    messages: deque = field(default_factory=deque)
    summary: str = ""
    total_tokens: int = 0


class ContextManager:
    """FIFO 上下文队列管理器。

    每个 thread_id 对应一个 ContextState。
    持久化到 agent_checkpoints.db 的 context_state 表。
    """

    def __init__(self, db_path: Path, llm, recall_store,
                 max_tokens: int = 90000, summary_model: str = "deepseek-v4-flash",
                 memory_extraction_model: str = "deepseek-v4-flash",
                 memory_db=None, hybrid_retriever=None,
                 background_scheduler=None, persona_loader=None):
        """
        Args:
            db_path: agent_checkpoints.db 的路径
            llm: LLMProvider 实例
            recall_store: RecallStore 实例
            max_tokens: FIFO 队列 token 上限
            summary_model: 摘要更新使用的模型
            memory_extraction_model: 记忆提取使用的模型
            memory_db: MemoryDB 实例（用于空闲检测和提取状态追踪）
            hybrid_retriever: HybridRetriever 实例（用于被动注入）
            background_scheduler: BackgroundTaskScheduler 实例（用于空闲提取）
            persona_loader: PersonaLoader 实例（用于 Core Memory 格式化）
        """
        self._db_path = db_path
        self._llm = llm
        self._store = recall_store
        self._max_tokens = max_tokens
        self._summary_model = summary_model
        self._memory_model = memory_extraction_model
        self._memory_db = memory_db
        self._hybrid_retriever = hybrid_retriever
        self._scheduler = background_scheduler
        self._persona_loader = persona_loader
        self._states: dict[str, ContextState] = {}
        self._lock = asyncio.Lock()
        self._init_db()
        self._memory_extractor = None

    def _init_db(self):
        """创建 context_state 表（如果不存在）。"""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS context_state (
                    thread_id TEXT PRIMARY KEY,
                    messages_json TEXT NOT NULL DEFAULT '[]',
                    summary TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def _get_memory_extractor(self):
        """延迟加载 MemoryExtractor（避免循环导入）。"""
        if self._memory_extractor is None:
            from .memory_extractor import MemoryExtractor
            self._memory_extractor = MemoryExtractor(
                self._llm, self._store, self._memory_model,
            )
        return self._memory_extractor

    # ================================================================
    #  加载 / 保存
    # ================================================================

    async def load(self, thread_id: str) -> ContextState:
        """从 SQLite 加载指定线程的上下文状态。若不存在则创建新的。"""
        async with self._lock:
            if thread_id in self._states:
                return self._states[thread_id]

            state = ContextState(thread_id=thread_id)
            try:
                conn = sqlite3.connect(str(self._db_path))
                try:
                    row = conn.execute(
                        "SELECT messages_json, summary FROM context_state WHERE thread_id = ?",
                        (thread_id,)
                    ).fetchone()
                    if row:
                        messages_list = json.loads(row[0])
                        state.messages = deque(messages_list)
                        state.summary = row[1] or ""
                        state.total_tokens = count_messages_tokens(messages_list)
                        state.total_tokens += self._summary_token_cost(state.summary)
                        logger.info(
                            "context loaded thread=%s: %d msgs, %d tokens, summary=%d chars",
                            thread_id, len(messages_list), state.total_tokens, len(state.summary),
                        )
                finally:
                    conn.close()
            except Exception as e:
                logger.warning("context load failed for %s: %s, using empty state", thread_id, e)

            self._states[thread_id] = state
            return state

    async def save(self, thread_id: str):
        """将指定线程的上下文状态持久化到 SQLite。"""
        async with self._lock:
            state = self._states.get(thread_id)
            if state is None:
                return
            messages_json = json.dumps(
                self._strip_internal_fields(state.messages),
                ensure_ascii=False, default=str,
            )
            now = _now().isoformat()
            try:
                conn = sqlite3.connect(str(self._db_path))
                try:
                    conn.execute(
                        "INSERT OR REPLACE INTO context_state "
                        "(thread_id, messages_json, summary, updated_at) VALUES (?, ?, ?, ?)",
                        (thread_id, messages_json, state.summary, now),
                    )
                    conn.commit()
                finally:
                    conn.close()
                logger.debug(
                    "context saved thread=%s: %d msgs, summary=%d chars",
                    thread_id, len(state.messages), len(state.summary),
                )
            except Exception as e:
                logger.error("context save failed for %s: %s", thread_id, e)

    # ================================================================
    #  FIFO 入队 + 压缩
    # ================================================================

    async def enqueue(self, thread_id: str, message: dict):
        """将一条消息入队。若超限则触发弹出 -> 存储 -> 提取 -> 摘要流程。

        同时执行:
        - 70% 压力警告: 检查 token 使用率，超过 70% 时注入警告
        - 空闲检测: 检查是否 idle > 30min，触发后台提取
        - 预热标记: 新用户首次对话标记

        Args:
            thread_id: 线程 ID
            message: 消息 dict，可包含 _ephemeral=True 标记
        """
        state = await self.load(thread_id)
        message["_thread_id"] = thread_id

        popped_for_summary = []
        popped_for_extraction = []
        async with self._lock:
            # 空闲检测（在消息入队前检查）
            if self._memory_db is not None:
                extraction_state = self._memory_db.get_extraction_state(thread_id)
                last_active = extraction_state.get("last_active_at")
                last_extracted = extraction_state.get("last_extracted_index", -1)

                # 空闲检测: idle > 30min
                if last_active is not None:
                    try:
                        last_dt = datetime.fromisoformat(last_active.replace("Z", "+00:00"))
                        if last_dt.tzinfo is None:
                            last_dt = last_dt.replace(tzinfo=_BEIJING_TZ)
                        delta_minutes = (datetime.now(_BEIJING_TZ) - last_dt).total_seconds() / 60
                        if delta_minutes > IDLE_EXTRACTION_MINUTES and self._scheduler is not None:
                            # 后台异步提取未处理的消息
                            unprocessed = self._get_unprocessed_messages(
                                state, last_extracted,
                            )
                            if unprocessed:
                                logger.info(
                                    "context idle detected: thread=%s idle for %.0f min, "
                                    "%d unprocessed messages",
                                    thread_id, delta_minutes, len(unprocessed),
                                )
                                task = asyncio.create_task(
                                    self._scheduler.process_idle_extraction(
                                        thread_id, unprocessed, last_extracted,
                                    )
                                )
                                task.add_done_callback(self._log_task_failure)
                    except (ValueError, TypeError) as e:
                        logger.debug("idle detection timestamp parse failed: %s", e)

                # 更新活跃时间
                self._memory_db.touch_active_at(thread_id)

            state.messages.append(message)
            state.total_tokens += count_message_tokens(message)

            # 70% 压力警告
            pressure_ratio = state.total_tokens / self._max_tokens if self._max_tokens > 0 else 0
            if pressure_ratio > PRESSURE_WARN_RATIO:
                logger.info(
                    "context PRESSURE WARNING: thread=%s at %.0f%% capacity (%d/%d tokens)",
                    thread_id, pressure_ratio * 100, state.total_tokens, self._max_tokens,
                )

            popped_count = 0
            while state.total_tokens > self._max_tokens and len(state.messages) > 1:
                popped = state.messages.popleft()
                state.total_tokens -= count_message_tokens(popped)
                popped_count += 1

                if settings.recall_storage_enabled:
                    self._store.add_recall_message(popped)

                if not popped.get("_ephemeral"):
                    popped_for_summary.append(popped)

                if settings.memory_extraction_enabled:
                    popped_for_extraction.append(popped)

            if popped_count > 0:
                logger.info(
                    "context FIFO popped %d msgs for thread=%s, remaining: %d msgs, %d tokens",
                    popped_count, thread_id, len(state.messages), state.total_tokens,
                )
                # FIFO 弹出导致索引平移，同步前移提取游标，避免漏提取/重复提取 (P1-4)
                if self._memory_db is not None:
                    self._adjust_extraction_index_after_pop(thread_id, popped_count)

        # 锁外：摘要更新 + 长期记忆提取并行执行 (P1-8 / P-fix)。
        # 移出锁内，避免 LLM 网络调用阻塞全局锁；同步等待完成，使本轮压缩的
        # 产物（新摘要、新长期记忆）在本轮 build_memory_context_text 时即可见，
        # 不再慢一拍。
        tasks = []
        if popped_for_summary:
            tasks.append(self._update_summary(state, popped_for_summary))
        if popped_for_extraction:
            extractor = self._get_memory_extractor()
            tasks.append(extractor.extract_batch(popped_for_extraction))
        if tasks:
            await asyncio.gather(*tasks)

    async def _update_summary(self, state: ContextState, popped_messages: list):
        """批量增量更新摘要: 旧摘要 + 多条弹出消息 -> LLM -> 新摘要。

        调用方不得在持有 self._lock 时调用。LLM 网络调用在锁外执行，
        仅在读/写 state.summary 时短暂加锁，避免全局锁阻塞其他线程 (P-fix)。
        """
        items = []
        for popped in popped_messages:
            role = popped.get("role", "unknown")
            content = str(popped.get("content", ""))[:300]
            if not content.strip():
                continue
            items.append(f"[{role}]: {content}")
        if not items:
            return

        messages_block = "\n".join(items)[:2000]

        async with self._lock:
            # 旧摘要截断，控制摘要输入长度，避免长对话下 reasoning 越滚越长
            existing = (state.summary or EMPTY_SUMMARY)[:2000]
        user_prompt = SUMMARY_UPDATE_USER.format(
            existing_summary=existing if existing else "(no prior summary)",
            messages=messages_block,
        )

        try:
            resp = await self._llm.generate(
                system=SUMMARY_UPDATE_SYSTEM,
                user=user_prompt,
                model=self._summary_model,
                max_tokens=2048,
                temperature=0.3,
            )
            new_summary = resp.content.strip()
            if not new_summary:
                # LLM 偶发返回空：保留原摘要，避免清空已累积的摘要
                logger.warning("context summary update returned empty, keeping old summary")
                return
        except Exception as e:
            logger.warning("context summary update failed: %s", e)
            return

        async with self._lock:
            old_len = len(state.summary)
            old_summary_tokens = self._summary_token_cost(state.summary)
            state.summary = new_summary
            state.total_tokens += self._summary_token_cost(new_summary) - old_summary_tokens
            logger.info(
                "context summary updated: %d -> %d chars. New summary preview: %s...",
                old_len, len(new_summary), new_summary[:120],
            )

    # ================================================================
    #  构建上下文窗口
    # ================================================================

    def build_context_window(self, thread_id: str) -> list[dict]:
        """构建传给 LLM 的上下文窗口消息列表。

        Returns:
            消息列表（不含 persona system prompt），格式:
            [system: "Conversation summary so far: ..."]  (仅当摘要非空)
            [user: msg_1]
            [assistant: resp_1]
            ...
        """
        state = self._states.get(thread_id)
        if state is None:
            return []

        context = []
        if state.summary:
            context.append({
                "role": "system",
                "content": f"Conversation summary so far: {state.summary}",
            })
        context.extend(list(state.messages))
        return context

    def build_history_messages(self, thread_id: str,
                               exclude_current_query: bool = True) -> list[dict]:
        """返回纯 user/assistant 的历史消息，供 supervisor 消费。

        排除 role=system 的消息（摘要 / Core Memory / 被动注入改走
        build_memory_context_text 注入 system prompt，避免被 supervisor
        的 history 过滤丢弃）。

        exclude_current_query=True 时排除最后一条（enqueue(当前 query) 之后，
        FIFO 末尾即为当前 query，由 supervisor 单独追加，避免重复注入）。
        """
        state = self._states.get(thread_id)
        if state is None:
            return []
        msgs = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in state.messages
            if m.get("role") in ("user", "assistant")
        ]
        if exclude_current_query and msgs:
            msgs = msgs[:-1]
        return msgs

    def build_memory_context_text(self, thread_id: str, user_query: str = "") -> str:
        """把摘要 + Core Memory + 被动注入拼成一段纯文本，供注入 system prompt。

        返回的文本由调用方（supervisor）作为 system prompt 的一部分传给 LLM，
        不再作为 role=system 的消息混入 history（避免被消费端丢弃）。
        """
        parts = []

        # Core Memory 注入（persona + human 块）
        if self._memory_db is not None:
            persona = self._memory_db.get_core_memory_block("persona")
            if persona and persona.get("content") and self._persona_loader:
                persona_text = self._persona_loader.format_for_context(
                    persona["content"], "persona",
                )
                if persona_text:
                    parts.append(f"[Core Memory - Persona]\n{persona_text}")
            human = self._memory_db.get_core_memory_block("human")
            if human and human.get("content") and self._persona_loader:
                human_text = self._persona_loader.format_for_context(
                    human["content"], "human",
                )
                if human_text:
                    parts.append(f"[Core Memory - User Profile]\n{human_text}")

        # 摘要
        state = self._states.get(thread_id)
        if state and state.summary:
            parts.append(f"Conversation summary so far: {state.summary}")

        # 压力警告（上下文接近上限时提醒 LLM 主动保存记忆，避免信息被淘汰）
        if state and self._max_tokens > 0:
            ratio = state.total_tokens / self._max_tokens
            if ratio > PRESSURE_WARN_RATIO:
                available = max(0, self._max_tokens - state.total_tokens)
                parts.append(
                    f"[Memory pressure warning] context is {ratio * 100:.0f}% full "
                    f"({available} tokens remaining). Consider saving important "
                    f"information using save_to_memory before it gets evicted from "
                    f"context. You can save to 'core' (user profile) or "
                    f"'archival' (long-term facts)."
                )

        # 被动注入
        injection = self._build_passive_injection_text(user_query, thread_id)
        if injection:
            parts.append(injection)

        return "\n\n".join(p for p in parts if p)

    def _build_passive_injection_text(self, user_query: str, thread_id: str = "") -> str:
        """检索相关记忆并格式化为注入文本（无结果返回空串）。"""
        if self._hybrid_retriever is None or not user_query.strip():
            return ""
        try:
            budget = (
                self._compute_injection_budget(thread_id)
                if thread_id else getattr(settings, 'memory_injection_token_budget', 12000)
            )
            result = self._hybrid_retriever.search(
                user_query,
                source="all",
                score_threshold=0.5,
                token_budget=budget,
            )
            return self._hybrid_retriever.format_for_injection(result)
        except Exception as e:
            logger.warning("passive injection failed: %s", e)
            return ""

    def _compute_injection_budget(self, thread_id: str) -> int:
        """动态计算被动注入的 token 预算：总窗口 - 已占用。

        已占用 = FIFO+摘要（state.total_tokens）+ Core Memory + system prompt 估算，
        避免注入内容把整体上下文撑爆模型窗口。上限为 memory_injection_token_budget。
        """
        total = getattr(settings, 'context_window_tokens', 120000)
        state = self._states.get(thread_id)
        used = state.total_tokens if state else 0
        if self._memory_db is not None:
            for block in ("persona", "human"):
                b = self._memory_db.get_core_memory_block(block)
                if b:
                    used += b.get("token_count", 0) or 0
        used += 2000  # AgentLoop 硬编码 system prompt 的粗估
        cap = getattr(settings, 'memory_injection_token_budget', 12000)
        return max(0, min(total - used, cap))

    def build_context_window_with_injection(self, thread_id: str,
                                              user_query: str) -> list[dict]:
        """构建带被动注入的上下文窗口。

        在用户查询消息前注入检索到的相关记忆。
        如果所有结果低于阈值，不注入任何内容。

        Args:
            thread_id: 线程 ID
            user_query: 当前的用户查询（用于检索）

        Returns:
            消息列表，末尾包含注入的记忆上下文
        """
        context = self.build_context_window(thread_id)
        self._apply_passive_injection(context, user_query)
        return context

    def build_context_window_with_core_memory(self, thread_id: str,
                                                user_query: str = "") -> list[dict]:
        """构建完整的上下文窗口（含 Core Memory + 被动注入）。

        上下文结构:
        [system: Core Memory persona]
        [system: Core Memory human]
        [system: Conversation summary]
        [system: Passive injection]   <- 仅当有相关记忆
        [user: msg_1]
        [assistant: resp_1]
        ...
        [user: current_query]

        Args:
            thread_id: 线程 ID
            user_query: 当前用户查询

        Returns:
            完整的消息列表
        """
        context = []

        # Core Memory 注入（persona + human 块）
        if self._memory_db is not None:
            # Persona block
            persona = self._memory_db.get_core_memory_block("persona")
            if persona and persona.get("content") and self._persona_loader:
                persona_text = self._persona_loader.format_for_context(
                    persona["content"], "persona",
                )
                if persona_text:
                    context.append({
                        "role": "system",
                        "content": f"[Core Memory - Persona]\n{persona_text}",
                    })
            # Human block
            human = self._memory_db.get_core_memory_block("human")
            if human and human.get("content") and self._persona_loader:
                human_text = self._persona_loader.format_for_context(
                    human["content"], "human",
                )
                if human_text:
                    context.append({
                        "role": "system",
                        "content": f"[Core Memory - User Profile]\n{human_text}",
                    })

        # Summary + FIFO
        context.extend(self.build_context_window(thread_id))

        self._apply_passive_injection(context, user_query)
        return context

    def _apply_passive_injection(self, context: list[dict], user_query: str) -> None:
        """检索相关记忆并注入到上下文（用户查询之前）。"""
        if self._hybrid_retriever is None or not user_query.strip():
            return
        try:
            result = self._hybrid_retriever.search(
                user_query,
                source="all",
                score_threshold=0.5,
                token_budget=getattr(settings, 'memory_injection_token_budget', 12000),
            )
            injection_text = self._hybrid_retriever.format_for_injection(result)
            if injection_text:
                self._insert_injection(context, injection_text)
                logger.debug("passive injection: %d chars injected", len(injection_text))
        except Exception as e:
            logger.warning("passive injection failed: %s", e)

    def _insert_injection(self, context: list[dict], injection_text: str) -> None:
        """把被动注入插入到用户查询之前（而非追加到末尾）。

        上下文结构里最后一条是刚入队的 user query，
        注入应放在它前面，让 LLM 先看到相关记忆再看到问题。
        """
        injection_msg = {
            "role": "system",
            "content": injection_text,
            "_ephemeral": True,
        }
        if context:
            context.insert(len(context) - 1, injection_msg)
        else:
            context.append(injection_msg)

    @staticmethod
    def _summary_token_cost(summary: str) -> int:
        """计算摘要作为 system 消息注入的 token 开销（与 load 保持一致）。"""
        if not summary:
            return 0
        return count_message_tokens({
            "role": "system",
            "content": f"Conversation summary so far: {summary}",
        })

    @staticmethod
    def _strip_internal_fields(messages) -> list[dict]:
        """持久化前剥离内部字段（_thread_id/_ephemeral 等），避免污染存储。"""
        internal = {"_thread_id", "_ephemeral"}
        return [
            {k: v for k, v in m.items() if k not in internal}
            for m in messages
        ]

    # ================================================================
    #  压力检查 & 空闲检测
    # ================================================================

    def get_pressure_ratio(self, thread_id: str) -> float:
        """获取上下文 token 使用率 (0-1)。"""
        state = self._states.get(thread_id)
        if state is None or self._max_tokens <= 0:
            return 0.0
        return state.total_tokens / self._max_tokens

    def is_under_pressure(self, thread_id: str) -> bool:
        """检查是否超过 70% 压力阈值。"""
        return self.get_pressure_ratio(thread_id) > PRESSURE_WARN_RATIO

    def get_pressure_warning(self, thread_id: str) -> str:
        """生成压力警告消息（注入到上下文中提示 LLM）。"""
        ratio = self.get_pressure_ratio(thread_id)
        if ratio <= PRESSURE_WARN_RATIO:
            return ""

        available = self._max_tokens - self.get_state(thread_id).total_tokens if self.get_state(thread_id) else 0
        return (
            f"[system] Memory pressure warning: context is {ratio*100:.0f}% full "
            f"({available} tokens remaining). "
            f"Consider saving important information using save_to_memory before it gets "
            f"evicted from context. You can save to 'core' (user profile) or "
            f"'archival' (long-term facts)."
        )

    @staticmethod
    def _log_task_failure(task: "asyncio.Task") -> None:
        """后台任务完成回调：记录异常，避免 fire-and-forget 静默失败。"""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning("background task failed silently: %s", exc)

    def _adjust_extraction_index_after_pop(self, thread_id: str, popped_count: int) -> None:
        """FIFO 弹出后同步前移提取游标，保持游标与当前 FIFO 位置对齐。"""
        try:
            cur = self._memory_db.get_extraction_state(thread_id).get("last_extracted_index", -1)
            new = max(-1, cur - popped_count)
            if new != cur:
                self._memory_db.update_extraction_state(thread_id, new)
                logger.debug("extraction index adjusted after %d pops: %d -> %d", popped_count, cur, new)
        except Exception as e:
            logger.warning("extraction index adjust failed for %s: %s", thread_id, e)

    def _get_unprocessed_messages(self, state: ContextState,
                                   last_extracted_index: int) -> list[dict]:
        """获取未被提取过的 FIFO 消息。"""
        messages = list(state.messages)
        if last_extracted_index < 0:
            return messages
        start = min(last_extracted_index + 1, len(messages))
        return messages[start:]

    # ================================================================
    #  辅助方法
    # ================================================================

    def get_state(self, thread_id: str) -> Optional[ContextState]:
        """获取指定线程的 ContextState（不触发加载）。"""
        return self._states.get(thread_id)

    def ensure_loaded(self, thread_id: str):
        """同步检查线程是否已加载，若未加载则直接创建空状态入内存。

        用于服务启动时不需要 I/O 的场景。
        注意: 这不会触发 SQLite 读取，如需完整历史请用 load()。
        """
        if thread_id not in self._states:
            self._states[thread_id] = ContextState(thread_id=thread_id)

    def get_stats(self, thread_id: str) -> dict:
        """获取指定线程的统计信息。"""
        state = self._states.get(thread_id)
        if state is None:
            return {
                "thread_id": thread_id,
                "loaded": False,
                "message_count": 0,
                "total_tokens": 0,
                "summary_length": 0,
                "summary_preview": "",
            }
        return {
            "thread_id": thread_id,
            "loaded": True,
            "message_count": len(state.messages),
            "total_tokens": state.total_tokens,
            "summary_length": len(state.summary),
            "summary_preview": state.summary[:200] if state.summary else "",
        }

    async def delete_thread(self, thread_id: str):
        """删除某 thread 的上下文状态（SQLite context_state 表 + 内存 _states）。"""
        async with self._lock:
            self._states.pop(thread_id, None)
            try:
                conn = sqlite3.connect(str(self._db_path))
                try:
                    conn.execute("DELETE FROM context_state WHERE thread_id = ?", (thread_id,))
                    conn.commit()
                    logger.info("context delete_thread: thread=%s removed", thread_id)
                finally:
                    conn.close()
            except Exception as e:
                logger.warning("context delete_thread failed for %s: %s", thread_id, e)

    def log_context_snapshot(self, thread_id: str):
        """打印上下文当前状态的日志快照。"""
        state = self._states.get(thread_id)
        if state is None:
            logger.info("=== Context Snapshot: (not loaded) ===")
            return
        logger.info("=" * 60)
        logger.info("Context Snapshot for thread=%s", thread_id)
        logger.info("  Messages: %d | Tokens: %d/%d", len(state.messages), state.total_tokens, self._max_tokens)
        logger.info("  Summary (%d chars): %s...", len(state.summary), state.summary[:200])
        logger.info("  Message Queue (oldest -> newest):")
        for i, msg in enumerate(state.messages):
            role = msg.get("role", "?")
            content = str(msg.get("content", ""))[:80]
            ephemeral = " [EPHEMERAL]" if msg.get("_ephemeral") else ""
            logger.info("    [%d] %s%s: %s...", i, role, ephemeral, content)
        logger.info("=" * 60)
        logger.info("  RecallStore stats: %s", self._store.stats)
        logger.info("=" * 60)

    async def close(self):
        """保存所有状态并清理。"""
        for thread_id in list(self._states.keys()):
            await self.save(thread_id)
        self._states.clear()
