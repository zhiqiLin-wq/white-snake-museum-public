"""per-user MemorySystem 工厂 + 缓存。

多用户隔离：每个用户独立的 memory.db / context_state.db / ChromaDB 集合。
main.py 注入一个「创建函数」（给定 user_id 返回新的 MemorySystem），
本模块负责缓存复用与并发保护。

与 langgraph/memory.py 的 get_user_memory 是同一套 per-user 隔离思路：
checkpoint 走 agent_checkpoints.db，记忆系统走本模块的 MemorySystem。
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

# async def create_fn(user_id: str) -> MemorySystem
_create_memory_system = None

_memory_system_cache: dict[str, object] = {}
_lock = asyncio.Lock()


def set_memory_system_factory(create_fn):
    """注入 per-user MemorySystem 创建函数（由 main.py 在 lifespan 中调用）。

    create_fn 签名: async def create_fn(user_id: str) -> MemorySystem
    """
    global _create_memory_system
    _create_memory_system = create_fn


async def get_user_memory_system(user_id: str):
    """按 user_id 获取或创建 MemorySystem（缓存复用 + 并发保护）。

    Raises:
        RuntimeError: 工厂尚未注入
    """
    if _create_memory_system is None:
        raise RuntimeError("memory system factory not initialized")
    if user_id in _memory_system_cache:
        return _memory_system_cache[user_id]
    async with _lock:
        if user_id in _memory_system_cache:
            return _memory_system_cache[user_id]
        ms = await _create_memory_system(user_id)
        _memory_system_cache[user_id] = ms
        return ms


async def close_all_memory_systems() -> None:
    """关闭所有缓存的 MemorySystem（用于服务关闭清理）。"""
    async with _lock:
        systems = list(_memory_system_cache.values())
        _memory_system_cache.clear()
    for ms in systems:
        try:
            scheduler = getattr(ms, "scheduler", None)
            if scheduler is not None:
                await scheduler.stop()
            cm = getattr(ms, "context_manager", None)
            if cm is not None:
                await cm.close()
        except Exception as e:
            logger.warning("memory system close failed: %s", e)


async def delete_user_conversation_memory(user_id: str, thread_id: str) -> dict:
    """删除某用户某对话产生的全部记忆/召回数据（跨存储联动）。

    供两个入口复用，保证「删对话即删记忆」语义一致：
    1. Python 删除端点（用户手动删对话，由 Node 回调）。
    2. TTL 过期清理（cleanup_all_expired_sessions）。

    Returns:
        {"status": "ok"|"degraded", "deleted_messages": int, ...}
    """
    if _create_memory_system is None:
        return {"status": "degraded", "deleted_messages": 0}
    if not user_id or not thread_id:
        return {"status": "ok", "deleted_messages": 0}
    try:
        ms = await get_user_memory_system(user_id)
    except RuntimeError:
        return {"status": "degraded", "deleted_messages": 0}
    if ms is None:
        return {"status": "ok", "deleted_messages": 0}
    try:
        recall_store = getattr(ms, "recall_store", None)
        memory_db = getattr(ms, "memory_db", None)
        context_manager = getattr(ms, "context_manager", None)
        if recall_store is not None:
            recall_store.delete_by_thread(thread_id)
        stats = {"deleted_messages": 0, "deleted_memories": 0, "deleted_recall_meta": 0}
        if memory_db is not None:
            stats = memory_db.delete_thread_data(thread_id)
        if context_manager is not None:
            await context_manager.delete_thread(thread_id)
        return {"status": "ok", **stats}
    except Exception as e:
        logger.warning(
            "delete_user_conversation_memory failed for user=%s thread=%s: %s",
            user_id, thread_id, e,
        )
        return {"status": "degraded", "deleted_messages": 0, "error": str(e)}
