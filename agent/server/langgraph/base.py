"""子图抽象基类 — 所有 workflow 的统一接口 (U04-01)。

使用方式:
    class MyWorkflow(BaseWorkflow):
        async def run(self, query, thread_id, **kwargs) -> WorkflowResult: ...
        async def run_stream(self, query, thread_id, **kwargs) -> AsyncGenerator[dict, None]: ...
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator
from ..api.schemas import WorkflowResult


class BaseWorkflow(ABC):
    """所有 LangGraph 子图的统一抽象接口。

    每个子图必须实现:
    - run():        非流式执行 → WorkflowResult
    - run_stream(): 流式执行 → AsyncGenerator[dict, None]
    """

    @abstractmethod
    async def run(self, query: str, thread_id: str = "default", **kwargs) -> WorkflowResult:
        """非流式执行工作流。"""
        ...

    @abstractmethod
    async def run_stream(self, query: str, thread_id: str = "stream", **kwargs) -> AsyncGenerator[dict, None]:
        """流式执行工作流 — 产出 SSE 事件 (不含 done 事件)。"""
        ...
