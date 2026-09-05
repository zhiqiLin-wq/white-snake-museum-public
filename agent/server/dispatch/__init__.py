"""统一工具调度层。

收敛分散在 agent_loop / langgraph_supervisor / evolution_workbench / skills_pipeline
的工具调用入口，统一处理 user_id 注入、tool_call_context 设置、MCP 调用、结果截断
与检索路径埋点，便于定位编排链路中的错误。
"""
from .tool_dispatcher import ToolDispatcher

__all__ = ["ToolDispatcher"]
