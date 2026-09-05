"""内存缓存，含 TTL + stale-while-revalidate。"""
import time
import asyncio
from typing import Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    value: Any
    expiry: float
    stale_expiry: float


class TTLCache:
    """简单的 TTL 内存缓存。

    支持 stale-while-revalidate：过期后先返回旧值，后台刷新。
    """

    def __init__(self, ttl_seconds: float = 600, stale_seconds: float = 300):
        self.ttl = ttl_seconds
        self.stale = stale_seconds
        self._store: dict[str, CacheEntry] = {}
        self._refreshing: set[str] = set()

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        now = time.monotonic()
        if now > entry.stale_expiry:
            # 完全过期
            del self._store[key]
            return None
        return entry.value

    def set(self, key: str, value: Any):
        now = time.monotonic()
        self._store[key] = CacheEntry(
            value=value,
            expiry=now + self.ttl,
            stale_expiry=now + self.ttl + self.stale,
        )

    def is_stale(self, key: str) -> bool:
        """检查是否已过期但未完全失效（可触发后台刷新）。"""
        entry = self._store.get(key)
        if entry is None:
            return False
        return time.monotonic() > entry.expiry

    async def get_or_fetch(
        self,
        key: str,
        fetcher: Callable[[], Awaitable[Any]],
    ) -> Any:
        """获取缓存，miss 时调用 fetcher 填充。"""
        value = self.get(key)
        if value is not None:
            # stale-while-revalidate
            if self.is_stale(key) and key not in self._refreshing:
                self._refreshing.add(key)
                asyncio.create_task(self._refresh(key, fetcher))
            return value

        # Cache miss
        return await self._refresh(key, fetcher)

    async def _refresh(self, key: str, fetcher: Callable[[], Awaitable[Any]]) -> Any:
        try:
            value = await fetcher()
            self.set(key, value)
            return value
        except Exception as e:
            logger.warning(f"缓存刷新失败 [{key}]: {e}")
            # 有旧值就返回旧值
            old = self._store.get(key)
            if old:
                return old.value
            raise
        finally:
            self._refreshing.discard(key)

    def invalidate(self, key: str):
        self._store.pop(key, None)

    def clear(self):
        self._store.clear()
