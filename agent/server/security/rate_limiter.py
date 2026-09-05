"""令牌桶限流器。"""
import time
import logging
from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from ..config import settings

logger = logging.getLogger(__name__)


class TokenBucket:
    """令牌桶算法 — 允许短时突发，限制平均速率。"""

    def __init__(self, rate: float, burst: int):
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)
        self.last_refill = time.monotonic()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(float(self.burst), self.tokens + elapsed * self.rate)
        self.last_refill = now

    def consume(self) -> bool:
        self._refill()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class RateLimiter:
    """按 IP 的令牌桶限流。"""

    def __init__(self, rate: float | None = None, burst: int | None = None):
        self.rate = rate if rate is not None else settings.rate_limit_per_second
        self.burst = burst if burst is not None else settings.rate_limit_burst
        self.buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(self.rate, self.burst)
        )

    def is_allowed(self, client_ip: str) -> bool:
        return self.buckets[client_ip].consume()


# 全局限流器实例
rate_limiter = RateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI 限流中间件。"""

    # ★ 免限流路径：健康检查和元数据端点不应受限
    EXEMPT_PATHS = {"/health", "/skills", "/rebuild-index", "/reload-templates"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)
        client_ip = request.client.host if request.client else "unknown"
        if not rate_limiter.is_allowed(client_ip):
            raise HTTPException(
                status_code=429,
                detail={"error": "RATE_LIMITED", "message": "请求过于频繁，请稍后再试"},
            )
        return await call_next(request)
