"""内存中的指标收集器，通过 /health 暴露。"""
import time
from collections import defaultdict


class Metrics:
    def __init__(self):
        self.start_time = time.monotonic()
        self.request_count = 0
        self.error_count = 0
        self.degraded_count = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0
        self.retrieval_hit_count: dict[str, int] = defaultdict(int)
        self.intent_distribution: dict[str, int] = defaultdict(int)

    def record_request(self, tokens: int = 0, latency_ms: float = 0, degraded: bool = False):
        self.request_count += 1
        self.total_tokens += tokens
        self.total_latency_ms += latency_ms
        if degraded:
            self.degraded_count += 1

    def record_error(self):
        self.error_count += 1

    def record_intent(self, intent: str):
        self.intent_distribution[intent] += 1

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.request_count, 1)

    @property
    def error_rate(self) -> float:
        return self.error_count / max(self.request_count, 1)

    @property
    def snapshot(self) -> dict:
        return {
            "requests": self.request_count,
            "errors": self.error_count,
            "error_rate": f"{self.error_rate:.2%}",
            "degraded": self.degraded_count,
            "total_tokens": self.total_tokens,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "intent_distribution": dict(self.intent_distribution),
        }
