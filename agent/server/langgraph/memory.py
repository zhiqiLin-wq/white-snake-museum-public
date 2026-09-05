"""对话记忆管理 — LangGraph Checkpointer 封装 (v8 AsyncSqliteSaver + WAL + TTL)。

B-001: WAL 模式
B-004: TTL 清理
B-005: 健康检查
B-006: Thread ID 隔离验证日志
B-007: 使用量监控
"""
import asyncio
import logging
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone as _timezone

BEIJING_TZ = _timezone(timedelta(hours=8))
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 会话过期时间（6 个月）
CONVERSATION_TTL_SECONDS = 180 * 24 * 60 * 60
# 活跃 thread 警告阈值
ACTIVE_THREAD_WARN_THRESHOLD = 100


def _cleanup_orphan_conversations(checkpoint_db_path: Path, thread_ids: list[str]) -> None:
    """删除 user.db 中 conversations 表的孤儿条目。

    当 agent_checkpoints.db 中的过期 checkpoint 被 TTL 清理后，
    同步删除 user.db 中对应的 conversations 行，防止"幽灵对话"。

    Args:
        checkpoint_db_path: agent_checkpoints.db 的完整路径
        thread_ids: 已从 checkpoints 中删除的 thread_id 列表
    """
    if not thread_ids:
        return
    # agent_checkpoints.db 与 user.db 在同一用户目录下
    user_db_path = checkpoint_db_path.parent / "user.db"
    if not user_db_path.exists():
        return
    try:
        conn = sqlite3.connect(str(user_db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            # 必须开启外键约束，否则 messages 表的 ON DELETE CASCADE 不生效，
            # 删 conversations 时会留下孤儿 messages 正文 (P-fix)。
            conn.execute("PRAGMA foreign_keys=ON")
            placeholders = ",".join("?" for _ in thread_ids)
            result = conn.execute(
                f"DELETE FROM conversations WHERE id IN ({placeholders})",
                thread_ids,
            )
            conn.commit()
            orphan_count = result.rowcount
            if orphan_count > 0:
                logger.info(
                    f"清理孤儿 conversations: 删除 {orphan_count} 条 (user.db: {user_db_path})"
                )
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"清理孤儿 conversations 失败 (user.db: {user_db_path}): {e}")


class ConversationMemory:
    """管理 LangGraph 对话检查点。

    使用 AsyncSqliteSaver 持久化（兼容 sync + async 操作）。
    回退: MemorySaver（开发环境）。
    """

    def __init__(self, store_type: str = "sqlite", db_path: str | None = None):
        if store_type == "sqlite" and db_path is None:
            raise ValueError(
                "ConversationMemory store_type='sqlite' requires an explicit db_path. "
                "Use get_user_memory(user_id) to get a per-user checkpointer."
            )
        self.store_type = store_type
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            self.db_path = Path(".")  # unused; MemorySaver mode doesn't need a path
        self._saver = None
        self._conn = None
        self._active_thread_ids: set[str] = set()
        self._checkpointer_type = "none"

        if store_type == "sqlite":
            self._init_sqlite()
        else:
            from langgraph.checkpoint.memory import MemorySaver
            self._saver = MemorySaver()
            self._checkpointer_type = "MemorySaver"
            logger.info("对话记忆: 内存模式（仅开发环境）")

    def _init_sqlite(self):
        # 不在这里创建 AsyncSqliteSaver（需要 async 上下文）。
        # 只保存路径和类型标记，实际的 checkpointer 在 _async_init() 中创建。
        try:
            import aiosqlite  # noqa: F401 — 验证 aiosqlite 可用
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: F401

            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._saver_cm = None  # 将在 _async_init() 中设置
            self._checkpointer_type = "AsyncSqliteSaver"
            self._saver = None
            self._conn_async = None  # aiosqlite 连接引用

            # 通过临时同步连接启用 WAL
            sync_conn = sqlite3.connect(str(self.db_path))
            try:
                sync_conn.execute("PRAGMA journal_mode=WAL")
                logger.info(f"WAL 模式已启用: {self.db_path}")
            finally:
                sync_conn.close()
            logger.info(f"对话记忆: AsyncSqliteSaver ({self.db_path})")
        except ImportError:
            logger.warning("AsyncSqliteSaver 不可用，回退到内存模式")
            from langgraph.checkpoint.memory import MemorySaver
            self._saver = MemorySaver()
            self._checkpointer_type = "MemorySaver"

    async def _async_init(self):
        """异步初始化：直接创建 aiosqlite 连接并构造 AsyncSqliteSaver。

        不使用 from_conn_string()，因为 sqlite:/// 前缀在 Windows 路径下
        （含反斜杠和盘符）会导致 URI 解析失败 → "unable to open database file"。

        必须在 async 上下文中调用（如 get_user_memory）。
        对于 MemorySaver，此方法为空操作。
        """
        if self._checkpointer_type == "AsyncSqliteSaver" and self._saver is None:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            self._conn_async = await aiosqlite.connect(str(self.db_path))
            self._saver = AsyncSqliteSaver(self._conn_async)
            logger.debug(f"AsyncSqliteSaver 已创建: {self.db_path}")

    # ================================================================
    #  B-001: WAL 模式
    # ================================================================

    def _enable_wal_mode(self):
        """在 SQLite 连接上启用 WAL 模式，允许并发读写。"""
        try:
            if self._checkpointer_type == "AsyncSqliteSaver":
                sync_conn = sqlite3.connect(str(self.db_path))
                try:
                    sync_conn.execute("PRAGMA journal_mode=WAL")
                    logger.info(f"WAL 模式已启用: {self.db_path}")
                finally:
                    sync_conn.close()
            elif self._conn is not None:
                self._conn.execute("PRAGMA journal_mode=WAL")
                logger.info(f"WAL 模式已启用: {self.db_path}")
        except Exception as e:
            logger.warning(f"WAL 模式启用失败: {e}，使用默认 journal_mode")

    def verify_wal_mode(self) -> str:
        """验证 WAL 模式是否成功启用。

        Returns:
            'wal' / 'delete' / 'memory' / 'unknown'
        """
        try:
            if self._conn is not None:
                result = self._conn.execute("PRAGMA journal_mode").fetchone()
                return result[0] if result else "unknown"
            if self._checkpointer_type == "AsyncSqliteSaver":
                sync_conn = sqlite3.connect(str(self.db_path))
                try:
                    result = sync_conn.execute("PRAGMA journal_mode").fetchone()
                    return result[0] if result else "unknown"
                finally:
                    sync_conn.close()
            return "unknown"
        except Exception as e:
            logger.warning(f"WAL 验证失败: {e}")
            return "unknown"

    # ================================================================
    #  B-004: Checkpointer 状态 TTL 清理
    # ================================================================

    async def cleanup_expired_sessions(self) -> list[str]:
        """删除 over TTL 的过期会话。

        以 user.db 中 conversations 表的 updated_at 作为过期判断依据
        （该字段在每次 Agent 对话同步时更新，可靠反映最后活跃时间）。
        与 conversations 关联的 agent_checkpoints.db 中的 checkpoints /
        checkpoint_blobs / checkpoint_writes 一并清理。

        注意: LangGraph 的 checkpoint 列存储 msgpack BLOB 而非 JSON 文本，
        SQLite 的 json_extract() 无法解析，因此过期判断必须在 Python 端完成。
        conversations.updated_at 是 SQLite datetime 格式字符串，可直接比较。

        Returns:
            过期的 thread_id 列表（供调用方联动清理记忆/召回）
        """
        if self._checkpointer_type == "MemorySaver":
            logger.debug("MemorySaver 无需 TTL 清理")
            return []

        cutoff_str = (datetime.now(BEIJING_TZ) - timedelta(seconds=CONVERSATION_TTL_SECONDS)).strftime('%Y-%m-%d %H:%M:%S')
        deleted_count = 0
        expired_thread_ids: list[str] = []

        try:
            # 从 user.db 的 conversations 表查出过期 thread_id
            # user.db 与 agent_checkpoints.db 在同一用户目录下
            user_db_path = self.db_path.parent / "user.db"
            if user_db_path.exists():
                conn = sqlite3.connect(str(user_db_path))
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                    cursor = conn.execute(
                        "SELECT id FROM conversations WHERE updated_at < ?",
                        (cutoff_str,)
                    )
                    expired_thread_ids = [row[0] for row in cursor.fetchall()]
                finally:
                    conn.close()

            if not expired_thread_ids:
                return []

            logger.info(
                f"TTL 清理: 发现 {len(expired_thread_ids)} 个过期会话 "
                f"(cutoff={cutoff_str}Z, TTL={CONVERSATION_TTL_SECONDS // 86400}d)"
            )

            # 清理 agent_checkpoints.db 中的 checkpoints / blobs / writes
            if not self.db_path.exists():
                logger.debug(f"agent_checkpoints.db 不存在，跳过 checkpoint 清理: {self.db_path}")
            elif self._checkpointer_type == "AsyncSqliteSaver":
                import aiosqlite
                async with aiosqlite.connect(str(self.db_path)) as db:
                    for thread_id in expired_thread_ids:
                        await db.execute("DELETE FROM checkpoint_blobs WHERE thread_id = ?", (thread_id,))
                        await db.execute("DELETE FROM checkpoint_writes WHERE thread_id = ?", (thread_id,))
                        await db.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
                        deleted_count += 1
                        self._active_thread_ids.discard(thread_id)
                    await db.commit()
            elif self._conn is not None:
                for thread_id in expired_thread_ids:
                    self._conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = ?", (thread_id,))
                    self._conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = ?", (thread_id,))
                    self._conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
                    deleted_count += 1
                    self._active_thread_ids.discard(thread_id)
                self._conn.commit()

            # 同步清理 user.db 中 conversations 表的过期条目
            _cleanup_orphan_conversations(self.db_path, expired_thread_ids)

            if deleted_count > 0:
                logger.info(
                    f"TTL 清理完成: 删除 {deleted_count} 个过期会话 "
                    f"(cutoff={cutoff_str}Z)"
                )
        except Exception as e:
            logger.error(f"TTL 清理失败: {e}", exc_info=True)

        return expired_thread_ids

    async def start_cleanup_scheduler(self, interval_seconds: int = 21600):
        """启动定时清理任务（默认每 6 小时）。

        应在 FastAPI lifespan 中作为 asyncio.create_task 启动。
        """
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                await self.cleanup_expired_sessions()
            except Exception as e:
                logger.error(f"定时清理任务异常: {e}")

    # ================================================================
    #  B-005: Checkpointer 健康检查
    # ================================================================

    def get_health_status(self) -> dict:
        """返回 checkpointer 的健康状态信息。"""
        return {
            "checkpointer_type": self._checkpointer_type,
            "checkpointer_ready": self.is_ready,
            "active_sessions": len(self._active_thread_ids),
            "wal_mode": self.verify_wal_mode(),
        }

    # ================================================================
    #  B-006: Thread ID 生成与隔离验证
    # ================================================================

    def record_thread_access(self, thread_id: str):
        """记录 thread_id 访问，并检测隔离问题。

        如果活跃 thread 数超过阈值，发出警告。
        """
        self._active_thread_ids.add(thread_id)
        active_count = len(self._active_thread_ids)

        if active_count > ACTIVE_THREAD_WARN_THRESHOLD:
            logger.warning(
                f"活跃 thread 数超过阈值: {active_count} > {ACTIVE_THREAD_WARN_THRESHOLD}，"
                f"请检查是否有 thread_id 泄漏"
            )

        logger.debug(f"thread_id={thread_id} 访问记录，当前活跃: {active_count}")

    def are_thread_ids_distinct(self, thread_id_a: str, thread_id_b: str) -> bool:
        """检查两个 thread_id 是否是不同的字符串值。

        此方法只做基础标识符比对，不涉及 checkpointer 层的数据隔离验证。
        实际隔离由 per-user 的 agent_checkpoints.db 文件系统级分离保证。

        Returns:
            True 表示两个 thread_id 是不同的字符串值
        """
        if thread_id_a == thread_id_b:
            logger.warning(f"thread_id 冲突: {thread_id_a} == {thread_id_b}")
            return False
        return True

    # 保留旧方法名作为别名，兼容已有调用
    def verify_thread_isolation(self, thread_id_a: str, thread_id_b: str) -> bool:
        """[已废弃] 请使用 are_thread_ids_distinct。"""
        return self.are_thread_ids_distinct(thread_id_a, thread_id_b)

    # ================================================================
    #  B-007: Memory 使用量监控
    # ================================================================

    @property
    def active_thread_count(self) -> int:
        return len(self._active_thread_ids)

    def get_usage_stats(self) -> dict:
        """获取当前内存使用统计。"""
        stats = {
            "active_thread_count": len(self._active_thread_ids),
            "checkpointer_type": self._checkpointer_type,
            "warning_threshold": ACTIVE_THREAD_WARN_THRESHOLD,
        }
        if self._checkpointer_type in ("AsyncSqliteSaver", "SqliteSaver"):
            conn = self._conn
            temp_conn = None
            try:
                if conn is None and self._checkpointer_type == "AsyncSqliteSaver":
                    temp_conn = sqlite3.connect(str(self.db_path))
                    conn = temp_conn
                if conn is not None:
                    cursor = conn.execute("SELECT COUNT(*) FROM checkpoints")
                    stats["total_checkpoints"] = cursor.fetchone()[0]
                    cursor = conn.execute("SELECT COUNT(DISTINCT thread_id) FROM checkpoints")
                    stats["total_threads_in_db"] = cursor.fetchone()[0]
            except Exception:
                logger.error("get_usage_stats: 无法读取 checkpointer 统计信息", exc_info=True)
            finally:
                if temp_conn:
                    temp_conn.close()
        return stats

    # ================================================================
    #  公共接口
    # ================================================================

    def get_checkpointer(self):
        return self._saver

    @property
    def is_ready(self) -> bool:
        return self._saver is not None

    async def aclose(self):
        """关闭底层数据库连接。

        对于 AsyncSqliteSaver：关闭 aiosqlite 连接，
        对于 SqliteSaver：关闭同步 sqlite3 连接，
        对于 MemorySaver：无操作。
        """
        if self._conn is not None:
            self._conn.close()
            self._conn = None

        if self._conn_async is not None:
            await self._conn_async.close()
            self._conn_async = None

        self._saver = None


# ================================================================
#  D3-1: 用户专属 Checkpointer 工厂函数（带缓存，防止重复创建和连接泄漏）
# ================================================================

# UUID v4 格式校验，防止路径遍历攻击
_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
    re.IGNORECASE,
)

_user_memory_cache: dict[str, 'ConversationMemory'] = {}
_user_memory_last_used: dict[str, float] = {}
_user_memory_lock = asyncio.Lock()

# 闲置清理: 30 分钟未使用的 ConversationMemory 实例会被清除
_MEMORY_CACHE_IDLE_MAX_SECONDS = 30 * 60


async def _cleanup_idle_memories() -> int:
    """清除闲置超时的 ConversationMemory 缓存条目。"""
    now = time.monotonic()
    async with _user_memory_lock:
        stale_ids = [
            uid for uid, last in _user_memory_last_used.items()
            if now - last > _MEMORY_CACHE_IDLE_MAX_SECONDS
        ]
        for uid in stale_ids:
            await _close_user_memory_unsafe(uid)
        if stale_ids:
            logger.info(f"ConversationMemory 闲置清理: 关闭 {len(stale_ids)} 个实例")
        return len(stale_ids)


async def get_user_memory(user_id: str) -> Optional['ConversationMemory']:
    """为指定用户获取或创建专属的 ConversationMemory 实例（缓存复用）。

    每个用户独立的 agent_checkpoints.db，实现对话历史的文件系统级隔离。

    Args:
        user_id: 用户 ID（UUID 字符串）

    Returns:
        ConversationMemory 实例，指向用户专属的 agent_checkpoints.db。
        如果 user_id 为空字符串、None 或格式不合法，返回 None。
    """
    if not user_id:
        return None

    # 防御性校验: user_id 必须是合法 UUID v4 格式，防止路径遍历
    if not _UUID_RE.match(user_id):
        logger.warning(f"get_user_memory: 非法 user_id 格式，拒绝访问: {user_id[:20]}...")
        return None

    # 每次访问时触发一次闲置清理（在锁外调用，_cleanup_idle_memories 内部自行获取锁）
    # 轻量级预检：仅在 cache 非空时才调用，避免不必要的函数开销
    if _user_memory_cache:
        await _cleanup_idle_memories()

    async with _user_memory_lock:
        # 二次检查 + 创建都持有锁，避免并发请求同一 user_id 时重复创建
        # 实例并泄漏 aiosqlite 连接（先写覆盖后写，被覆盖的连接无人关闭）
        if user_id in _user_memory_cache:
            _user_memory_last_used[user_id] = time.monotonic()
            return _user_memory_cache[user_id]

        from ..config import PROJECT_ROOT as _prj_root
        _project_root = _prj_root.parent
        # 验证项目根目录正确（应包含 data 目录）
        if not (_project_root / "data").is_dir():
            logger.error(
                f"项目根目录解析错误: {_project_root} 下未找到 data/ 目录。"
                f"请检查 config.py 的 PROJECT_ROOT 是否正确。"
            )
            return None
        db_path = _project_root / "data" / "users" / user_id / "agent_checkpoints.db"

        # 双保险：确保父目录存在（ConversationMemory._init_sqlite 内部也会执行此操作）
        db_path.parent.mkdir(parents=True, exist_ok=True)

        memory = ConversationMemory(store_type="sqlite", db_path=str(db_path))
        await memory._async_init()

        _user_memory_cache[user_id] = memory
        _user_memory_last_used[user_id] = time.monotonic()

    return memory


async def _close_user_memory_unsafe(user_id: str) -> None:
    """内部使用：关闭并移除指定用户的 ConversationMemory（需在锁内调用）。"""
    memory = _user_memory_cache.pop(user_id, None)
    _user_memory_last_used.pop(user_id, None)
    if memory is not None:
        await memory.aclose()


async def close_user_memory(user_id: str) -> None:
    """关闭并移除指定用户的 ConversationMemory。"""
    async with _user_memory_lock:
        await _close_user_memory_unsafe(user_id)


async def close_all_user_memories() -> None:
    """关闭所有缓存的用户 ConversationMemory（用于服务关闭清理）。"""
    async with _user_memory_lock:
        for user_id in list(_user_memory_cache.keys()):
            await _close_user_memory_unsafe(user_id)


async def cleanup_all_expired_sessions() -> int:
    """对所有缓存的用户执行 TTL 过期清理，并联动清理过期对话的记忆/召回。

    Returns:
        总共删除的会话数
    """
    from ..context.memory_registry import delete_user_conversation_memory
    total = 0
    async with _user_memory_lock:
        memories = list(_user_memory_cache.items())
    for user_id, memory in memories:
        try:
            expired_ids = await memory.cleanup_expired_sessions()
            total += len(expired_ids)
            if expired_ids:
                logger.info(f"TTL 清理: user={user_id} 删除 {len(expired_ids)} 个过期会话")
                for tid in expired_ids:
                    await delete_user_conversation_memory(user_id, tid)
        except Exception as e:
            logger.warning(f"TTL 清理: user={user_id} 失败: {e}")
    return total


async def start_global_cleanup_scheduler(interval_seconds: int = 21600):
    """启动全局 TTL 清理调度器（默认每 6 小时）。

    应在 FastAPI lifespan 中作为 asyncio.create_task 启动。
    遍历所有缓存的用户 ConversationMemory 实例并执行过期清理。
    """
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            total = await cleanup_all_expired_sessions()
            if total > 0:
                logger.info(f"全局 TTL 清理: 共删除 {total} 个过期会话")
        except Exception as e:
            logger.error(f"全局 TTL 清理异常: {e}")
