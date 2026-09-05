"""RAG 检索评估 — 性能基准测试 (Phase 6)。

T06.01~T06.06: 延迟 / 吞吐量 / 内存 / 冷热缓存 / 缓存一致性。
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

# 动态添加 agent 目录
AGENT_DIR = Path(__file__).resolve().parent.parent / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))


# ==============================================================================
# T06.02: bench_latency
# ==============================================================================
def bench_latency(
    retriever,
    queries: list[str],
    n_warmup: int = 5,
    n_repeat: int = 50,
) -> dict:
    """延迟测试: 对每条查询计时。

    Args:
        retriever: HybridRetriever 实例
        queries: 查询文本列表
        n_warmup: 预热查询数
        n_repeat: 测量查询数

    Returns:
        {p50_ms, p95_ms, p99_ms, mean_ms, min_ms, max_ms, n}
    """
    # Warmup
    for i in range(min(n_warmup, len(queries))):
        asyncio.run(retriever.retrieve(queries[i], top_k=10))

    # Measure
    latencies: list[float] = []
    measure_qs = queries[:n_repeat] if len(queries) >= n_repeat else queries * (n_repeat // len(queries) + 1)
    for i in range(n_repeat):
        q = measure_qs[i % len(measure_qs)]
        start = time.monotonic()
        asyncio.run(retriever.retrieve(q, top_k=10))
        elapsed = (time.monotonic() - start) * 1000
        latencies.append(elapsed)

    latencies.sort()
    n = len(latencies)
    return {
        "p50_ms": _percentile(latencies, 50),
        "p95_ms": _percentile(latencies, 95),
        "p99_ms": _percentile(latencies, 99),
        "mean_ms": sum(latencies) / n,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "n": n,
    }


def _percentile(sorted_data: list[float], p: float) -> float:
    n = len(sorted_data)
    idx = int(p / 100.0 * n)
    return sorted_data[min(idx, n - 1)]


# ==============================================================================
# T06.03: bench_throughput
# ==============================================================================
async def bench_throughput(
    retriever,
    queries: list[str],
    concurrency_levels: list[int] | None = None,
) -> dict:
    """吞吐量测试。

    Args:
        retriever: HybridRetriever 实例
        queries: 查询文本列表
        concurrency_levels: [1, 10, 50, 100]

    Returns:
        {concurrency: {qps, p50_ms, p95_ms, p99_ms, errors}}
    """
    if concurrency_levels is None:
        concurrency_levels = [1, 10, 50, 100]

    results: dict = {}

    for concurrency in concurrency_levels:
        # 扩展查询列表以匹配并发数
        cycle = queries * (concurrency // len(queries) + 1)
        batch = cycle[:concurrency]

        start = time.monotonic()
        errors = 0
        tasks = [retriever.retrieve(q, top_k=10) for q in batch]
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=120)
        except asyncio.TimeoutError:
            errors = concurrency

        elapsed = time.monotonic() - start
        qps = concurrency / elapsed if elapsed > 0 else 0

        results[f"concurrency={concurrency}"] = {
            "qps": round(qps, 1),
            "total_time_s": round(elapsed, 2),
            "errors": errors,
        }

    return results


# ==============================================================================
# T06.04: bench_memory
# ==============================================================================
def bench_memory() -> dict:
    """内存基准测试: Embedder / Reranker / ChromaDB / BM25 / RSS。

    Returns:
        {component: memory_mb, ...}
    """
    try:
        import psutil
    except ImportError:
        return {
            "error": "psutil 未安装。安装: pip install psutil",
            "rss_mb": 0,
        }

    process = psutil.Process()
    mem_info = process.memory_info()

    return {
        "rss_mb": round(mem_info.rss / 1024 / 1024, 1),
        "vms_mb": round(mem_info.vms / 1024 / 1024, 1),
        "note": "内存细分为组件级别需在模型加载前后分别测量 (T06.04 完整实现)",
    }


# ==============================================================================
# T06.05: 冷启动 vs 热缓存
# ==============================================================================
async def bench_cold_vs_warm(
    retriever,
    query: str,
    n_warm_runs: int = 5,
) -> dict:
    """冷启动 vs 热缓存对比。

    Args:
        retriever: HybridRetriever 实例
        query: 测试查询
        n_warm_runs: 热缓存测量次数

    Returns:
        {cold_ms, warm_ms, cache_hit_rate}
    """
    # 清除缓存
    if hasattr(retriever, '_query_cache'):
        retriever._query_cache.clear()

    # 冷启动
    start = time.monotonic()
    await retriever.retrieve(query, top_k=10)
    cold_ms = (time.monotonic() - start) * 1000

    # 热缓存 (连续 5 次，取后 3 次均值)
    warm_times: list[float] = []
    for i in range(n_warm_runs):
        start = time.monotonic()
        await retriever.retrieve(query, top_k=10)
        warm_times.append((time.monotonic() - start) * 1000)

    warm_ms = sum(warm_times[-3:]) / 3 if len(warm_times) >= 3 else sum(warm_times) / len(warm_times)

    return {
        "cold_ms": round(cold_ms, 1),
        "warm_ms": round(warm_ms, 1),
        "speedup": round(cold_ms / max(warm_ms, 0.001), 1),
    }


# ==============================================================================
# T06.06: 缓存一致性检查
# ==============================================================================
async def check_cache_consistency(retriever, query: str) -> dict:
    """验证缓存一致: 第一次和第二次查询结果完全相同。

    Args:
        retriever: HybridRetriever 实例
        query: 测试查询

    Returns:
        {consistent: bool, diff_count: int, diff_details: [...]}
    """
    # 清除缓存
    if hasattr(retriever, '_query_cache'):
        retriever._query_cache.clear()

    # 第一次查询
    result_a = await retriever.retrieve(query, top_k=10)

    # 第二次查询 (应命中缓存)
    result_b = await retriever.retrieve(query, top_k=10)

    # 比较
    ids_a = [r.get("chunk_id", "") for r in result_a]
    ids_b = [r.get("chunk_id", "") for r in result_b]

    consistent = ids_a == ids_b
    diff_count = sum(1 for a, b in zip(ids_a, ids_b) if a != b)

    return {
        "consistent": consistent,
        "diff_count": diff_count,
        "len_a": len(ids_a),
        "len_b": len(ids_b),
    }


# ==============================================================================
# T06.01: CLI
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="RAG 检索性能基准")
    parser.add_argument("--bench", choices=["latency", "throughput", "memory", "coldvswarm", "cache"], required=True)
    parser.add_argument("--config", help="YAML 配置文件")
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-repeat", type=int, default=50)
    parser.add_argument("--concurrency", type=str, default="10,50,100", help="逗号分隔")
    args = parser.parse_args()

    print(f"[BENCH] {args.bench} — 需要真实 RAG 组件初始化")
    print("[INFO] 完整性能基准需在生产环境中运行（含真实模型/索引）")
    print("[INFO] 当前为框架骨架 — 指标格式与设计文档 4.1 节一致")


if __name__ == "__main__":
    main()
