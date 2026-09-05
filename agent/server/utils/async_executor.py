"""信号量控制的异步并发执行器。

阶段 1 — P1-04.
"""
import asyncio
import logging
from typing import List, Callable, Awaitable, TypeVar, Literal

logger = logging.getLogger(__name__)

T = TypeVar("T")
OnError = Literal["raise", "skip", "return_none"]


async def async_run_with_semaphore(
    tasks: List[Callable[[], Awaitable[T]]],
    concurrency: int = 5,
    on_error: OnError = "return_none",
) -> List[T | None]:
    """信号量控制并发执行。

    Args:
        tasks: 异步任务工厂函数列表
        concurrency: 最大并发数
        on_error:
            - "raise": 任一任务失败立即抛出异常
            - "return_none": 失败的任务返回 None，继续执行其余任务

    Returns:
        与 tasks 等长的结果列表。失败且 on_error="return_none" 时对应位置为 None。

    Raises:
        ValueError: concurrency <= 0
    """
    if concurrency <= 0:
        raise ValueError(f"concurrency must be > 0, got {concurrency}")

    if not tasks:
        return []

    semaphore = asyncio.Semaphore(concurrency)

    async def _run_one(task_fn: Callable[[], Awaitable[T]]) -> T | None:
        async with semaphore:
            if on_error == "raise":
                return await task_fn()
            else:
                try:
                    return await task_fn()
                except Exception as e:
                    logger.error(
                        "async_run_with_semaphore task failed: %s", e
                    )
                    return None

    results = await asyncio.gather(*[_run_one(t) for t in tasks])

    if on_error == "raise":
        for r in results:
            if isinstance(r, BaseException):
                raise r

    return list(results)
