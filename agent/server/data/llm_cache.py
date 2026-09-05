"""B-125: LLM 响应缓存。

对相同的 LLM 请求（相同 prompt + model + temperature）进行缓存。
缓存 key = sha256(prompt)。TTL: 10 分钟。
"""
import hashlib
import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# 缓存 TTL: 10 分钟
LLM_CACHE_TTL_SECONDS = 600


class LLMResponseCache:
    """LLM 响应缓存。"""

    def __init__(self, ttl_seconds: int = LLM_CACHE_TTL_SECONDS):
        self._cache: dict[str, tuple[float, str]] = {}
        self.ttl = ttl_seconds

    def cache_key(self, system: str, user: str, model: str,
                  max_tokens: int, temperature: float) -> str:
        """生成缓存 key（含 max_tokens，避免不同 max_tokens 命中同一缓存）。"""
        raw = json.dumps({
            "s": system,
            "u": user,
            "m": model,
            "mt": max_tokens,
            "t": temperature,
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[str]:
        """获取缓存的响应内容。"""
        entry = self._cache.get(key)
        if entry is None:
            return None
        ts, content = entry
        if time.monotonic() - ts > self.ttl:
            del self._cache[key]
            return None
        return content

    def set(self, key: str, content: str):
        """缓存响应内容。"""
        self._cache[key] = (time.monotonic(), content)

    def invalidate(self, key: str):
        self._cache.pop(key, None)

    def clear(self):
        self._cache.clear()

    def size(self) -> int:
        return len(self._cache)
