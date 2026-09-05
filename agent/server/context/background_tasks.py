"""后台任务调度器。

在 FastAPI lifespan 中运行，负责:
1. 软删除清理 (tombstone): 每天凌晨执行一次
   - score < 0.05 且 30 天未访问 -> 标记为 tombstone
   - 写入 changelog，不物理删除
2. 记忆合并整合 (consolidation): 每天凌晨执行一次
   - 对 cosine > 0.7 的记忆执行合并
   - 旧条目标记为 consolidated
3. 空闲提取处理: 由 ContextManager 触发，本模块提供异步执行
4. Embedding 补偿重试: 定时重试 pending_embedding 的消息
"""

import asyncio
import json
import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

_BEIJING_TZ = timezone(timedelta(hours=8))

# 定时任务间隔
MERGE_CONSOLIDATE_HOUR = 3     # 每天凌晨 3:00 执行合并
TOMBSTONE_CLEANUP_HOUR = 3     # 每天凌晨 3:00 执行软删除
EMBEDDING_RETRY_MINUTES = 15   # 每 15 分钟重试一次 embedding 补偿
IDLE_EXTRACTION_MINUTES = 30   # 空闲 30 分钟触发提取


class BackgroundTaskScheduler:
    """后台任务调度器。

    所有任务在同一个 asyncio 事件循环中运行。
    通过与 ContextManager 和 MemoryDB 的协作完成生命周期管理。
    """

    def __init__(self, memory_db, recall_store, context_manager=None,
                 memory_extractor=None, embedder=None, persona_loader=None):
        """
        Args:
            memory_db: MemoryDB 实例
            recall_store: RecallStore 实例
            context_manager: ContextManager 实例（用于空闲提取）
            memory_extractor: MemoryExtractor 实例
            embedder: Embedder 实例（用于合并时的向量计算）
            persona_loader: PersonaLoader 实例（用于空闲提取时更新 human 块）
        """
        self._db = memory_db
        self._store = recall_store
        self._cm = context_manager
        self._extractor = memory_extractor
        self._embedder = embedder
        self._persona = persona_loader
        self._running = False

    def attach_context_manager(self, context_manager):
        """回填 idle 提取所需的 ContextManager 依赖。

        ContextManager 需要 scheduler 的引用（enqueue 时触发 idle 提取），
        而 scheduler 又需要 ContextManager（idle 提取时读 FIFO 与获取 extractor），
        存在循环依赖。故由 build_memory_system 在两者都创建后调用本方法回填。
        """
        self._cm = context_manager
        if context_manager is not None:
            self._extractor = context_manager._get_memory_extractor()

    async def start(self):
        """启动所有后台任务。"""
        self._running = True
        logger.info("background_tasks scheduler starting")
        asyncio.create_task(self._daily_maintenance_loop())
        asyncio.create_task(self._embedding_retry_loop())
        logger.info("background_tasks scheduler started: daily_maintenance + embedding_retry")

    async def stop(self):
        """停止所有后台任务。"""
        self._running = False
        logger.info("background_tasks scheduler stopped")

    # ================================================================
    #  每天凌晨执行: 软删除 + 合并整合
    # ================================================================

    async def _daily_maintenance_loop(self):
        """每天凌晨执行维护任务。"""
        while self._running:
            await self._sleep_until_next_run(MERGE_CONSOLIDATE_HOUR)
            if not self._running:
                break
            try:
                logger.info("background_tasks: starting daily maintenance")
                await self._run_tombstone_cleanup()
                await self._run_memory_merge()
                logger.info("background_tasks: daily maintenance complete")
            except Exception as e:
                logger.error("background_tasks daily maintenance failed: %s", e, exc_info=True)

    async def _run_tombstone_cleanup(self):
        """软删除清理：按分类型 importance 保底 + TTL 标记低价值过期记忆。"""
        from .memory_scorer import MemoryScorer
        scorer = MemoryScorer(self._db)

        candidates = self._db.get_memories_for_decay(days_unaccessed=0)
        if not candidates:
            logger.debug("tombstone cleanup: no candidates")
            return

        tombstone_count = 0
        for mem in candidates:
            days = scorer.days_since_access(mem)
            if days is None:
                continue
            if scorer.should_tombstone(
                importance=mem.get("importance", 0.5),
                memory_type=mem.get("memory_type", "Entity"),
                days_unaccessed=days,
            ):
                self._db.update_memory_status(mem["memory_id"], "tombstone", reason="decay_threshold")
                tombstone_count += 1

        if tombstone_count > 0:
            logger.info("tombstone cleanup: %d memories marked as tombstone", tombstone_count)

    async def _run_memory_merge(self):
        """记忆合并整合：相似度 > 0.7 的记忆合并为一条。"""
        if self._embedder is None:
            logger.debug("memory merge: no embedder available, skipping")
            return

        candidates = self._db.get_memories_for_merge(min_similarity=0.7, limit=50)
        if len(candidates) < 2:
            return

        # 获取 embeddings
        texts = [m["content"] for m in candidates]
        try:
            embeddings = self._embedder.embed(texts)
        except Exception as e:
            logger.warning("memory merge embed failed: %s", e)
            return

        # 两两比较 cosine similarity
        merged_pairs = []
        n = len(candidates)
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                if sim > 0.7:
                    merged_pairs.append((i, j, sim))

        if not merged_pairs:
            return

        # 对每对执行合并
        merge_count = 0
        seen_ids = set()
        for i, j, sim in merged_pairs:
            mid_a = candidates[i]["memory_id"]
            mid_b = candidates[j]["memory_id"]
            if mid_a in seen_ids or mid_b in seen_ids:
                continue
            seen_ids.add(mid_a)
            seen_ids.add(mid_b)

            # 合并内容
            content_a = candidates[i]["content"]
            content_b = candidates[j]["content"]
            merged_content = f"{content_a}; {content_b}"

            # 取较高 importance
            importance = max(
                candidates[i].get("importance", 0.5),
                candidates[j].get("importance", 0.5),
            )

            # 创建新合并条目
            new_id = self._store.add_memory(
                merged_content,
                memory_type=candidates[i].get("memory_type", "Entity"),
                importance=importance,
                confidence=0.7,
            )

            if new_id:
                # 标记旧条目为 consolidated
                self._db.update_memory_status(mid_a, "consolidated",
                                                reason=f"merged into {new_id}")
                self._db.update_memory_status(mid_b, "consolidated",
                                                reason=f"merged into {new_id}")
                self._db.record_changelog(
                    new_id, "MERGE",
                    old_value=json.dumps({
                        "source_a": mid_a, "content_a": content_a,
                        "source_b": mid_b, "content_b": content_b,
                    }, ensure_ascii=False),
                    new_value=merged_content,
                    reason=f"cosine_similarity={sim:.3f}",
                )
                merge_count += 1
                logger.info("memory merge: %s + %s -> %s (sim=%.3f)", mid_a, mid_b, new_id, sim)

        if merge_count > 0:
            logger.info("memory merge: %d pairs merged into new memories", merge_count)

    # ================================================================
    #  空闲提取处理
    # ================================================================

    async def process_idle_extraction(self, thread_id: str,
                                       messages: list[dict],
                                       last_extracted_index: int):
        """异步处理空闲提取。

        由 ContextManager 在检测到 idle > 30min 时调用。

        Args:
            thread_id: 线程 ID
            messages: FIFO 中 last_extracted_index 之后的消息列表
            last_extracted_index: 上次提取位置
        """
        if not messages:
            logger.debug("idle extraction: no new messages for thread=%s", thread_id)
            return

        logger.info(
            "idle extraction: thread=%s, %d messages to process (index %d->%d)",
            thread_id, len(messages), last_extracted_index,
            last_extracted_index + len(messages),
        )

        if self._extractor is not None:
            try:
                # 批量提取
                new_memories = await self._extractor.extract_batch(messages)
                logger.info(
                    "idle extraction: %d new memories for thread=%s",
                    len(new_memories), thread_id,
                )
            except Exception as e:
                logger.error("idle extraction LLM call failed: %s", e)

        # 更新提取进度（clamp 到当前 FIFO 末尾，避免异步执行期间 FIFO 变化导致游标越界）
        new_index = last_extracted_index + len(messages)
        if self._cm is not None:
            cur_state = self._cm.get_state(thread_id)
            if cur_state is not None:
                new_index = min(new_index, len(cur_state.messages) - 1)
        self._db.update_extraction_state(thread_id, new_index)

        # 尝试更新 Core Memory (human 块)
        if self._persona is not None and self._cm is not None:
            try:
                await self._update_human_from_idle(thread_id, messages)
            except Exception as e:
                logger.warning("idle extraction human update failed: %s", e)

    async def _update_human_from_idle(self, thread_id: str,
                                       messages: list[dict]):
        """从空闲消息中提取用户画像更新。"""
        user_contents = [
            str(m.get("content", ""))[:500]
            for m in messages
            if m.get("role") == "user"
            and len(str(m.get("content", ""))) > 20
        ]
        if not user_contents:
            return

        # 简单规则提取（不依赖 LLM 调用，避免额外成本）
        extracted = {}
        for content in user_contents:
            # 偏好关键词
            for prefix in ["我喜欢", "我偏好", "我研究方向", "我是", "我叫", "我在", "我的"]:
                if prefix in content:
                    idx = content.index(prefix)
                    snippet = content[idx:idx + 80]
                    if "preferences" not in extracted:
                        extracted["preferences"] = []
                    extracted["preferences"].append(snippet)
                    break

            # 约束关键词
            for kw in ["不要", "别", "禁止", "不能", "不想"]:
                if kw in content:
                    if "constraints" not in extracted:
                        extracted["constraints"] = []
                    idx = max(0, content.index(kw) - 10)
                    snippet = content[idx:idx + 100]
                    extracted["constraints"].append(snippet)
                    break

        if extracted:
            import json
            current = self._db.get_core_memory_block("human")
            current_content = current["content"] if current else "{}"
            try:
                current_obj = json.loads(current_content)
            except (json.JSONDecodeError, TypeError):
                current_obj = {}

            for key, values in extracted.items():
                if key not in current_obj:
                    current_obj[key] = {}
                if isinstance(current_obj[key], dict):
                    for v in values:
                        if "_extracted" not in current_obj[key]:
                            current_obj[key]["_extracted"] = []
                        if v not in current_obj[key]["_extracted"]:
                            current_obj[key]["_extracted"].append(v)

            merged = json.dumps(current_obj, ensure_ascii=False, indent=2)
            self._db.set_core_memory_block("human", merged)
            logger.info("idle extraction: human block updated with %d fields", len(extracted))

    # ================================================================
    #  Embedding 补偿重试
    # ================================================================

    async def _embedding_retry_loop(self):
        """每隔 EMBEDDING_RETRY_MINUTES 重试 pending_embedding 的消息。"""
        while self._running:
            await asyncio.sleep(EMBEDDING_RETRY_MINUTES * 60)
            if not self._running:
                break
            try:
                await self._retry_pending_embeddings()
            except Exception as e:
                logger.error("embedding retry failed: %s", e)

    async def _retry_pending_embeddings(self):
        """重试所有 pending_embedding=True 的消息。"""
        pending = self._db.get_pending_embeddings(limit=50)
        if not pending:
            return

        logger.info("embedding retry: %d pending messages", len(pending))
        success_count = 0
        for msg in pending:
            try:
                embedding = self._embedder.embed([msg["content"]])[0]
                source = msg.get("source", "recall")

                if source == "memory_extraction":
                    collection = self._store._memory_collection
                    # 从 SQLite 补全长期记忆元数据，与 add_memory 正常写入的 metadata 对齐
                    mem = self._db.get_long_term_memory(msg["message_id"])
                    metadata = {
                        "message_id": msg["message_id"],
                        "memory_type": mem.get("memory_type", "Entity") if mem else "Entity",
                        "importance": mem.get("importance", 0.5) if mem else 0.5,
                        "confidence": mem.get("confidence", 0.5) if mem else 0.5,
                        "timestamp": msg.get("timestamp", ""),
                    }
                else:
                    collection = self._store._recall_collection
                    metadata = {
                        "message_id": msg["message_id"],
                        "role": msg.get("role", ""),
                        "thread_id": msg.get("thread_id", ""),
                        "timestamp": msg.get("timestamp", ""),
                    }

                if collection is not None:
                    collection.add(
                        embeddings=[embedding],
                        documents=[msg["content"]],
                        metadatas=[metadata],
                        ids=[msg["message_id"]],
                    )
                    self._db.mark_embedding_done(msg["message_id"])
                    success_count += 1
            except Exception as e:
                logger.warning("embedding retry failed for %s: %s", msg["message_id"], e)

        if success_count > 0:
            logger.info("embedding retry: %d/%d succeeded", success_count, len(pending))

    # ================================================================
    #  辅助
    # ================================================================

    async def _sleep_until_next_run(self, target_hour: int):
        """休眠到下一个 target_hour (北京时间)。"""
        now = datetime.now(_BEIJING_TZ)
        next_run = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
        if next_run <= now:
            next_run = next_run + timedelta(days=1)
        wait_seconds = (next_run - now).total_seconds()
        logger.debug("daily maintenance: sleeping %.1f minutes until %s",
                      wait_seconds / 60, next_run.isoformat())
        await asyncio.sleep(wait_seconds)

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """计算余弦相似度。"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return dot / (norm_a * norm_b)
