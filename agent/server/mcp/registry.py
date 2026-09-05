"""MCP 工具注册中心。

B-026: 结构化工具调用日志
B-027: 统一错误处理与降级

工具调用埋点：call_tool 是所有工具调用的唯一汇聚点，
统一记录完整输入参数 + 输出结果 + 错误现场到 logs/tool/tool_calls.log，
并通过 trace_id 与 LLM 调用串联，便于定位编排链路中的错误。
"""
import inspect
import logging
import re
import time
import traceback
import httpx
from typing import Optional

logger = logging.getLogger(__name__)


def _safe_record_tool_call(tool_name: str, arguments: dict, result,
                           latency_ms: float, success: bool,
                           error=None, error_type=None, traceback_str=None):
    """安全地记录工具调用（埋点本身失败不影响主流程）。"""
    try:
        from ..observability.tracer import Tracer
        Tracer.record_tool_call(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            latency_ms=latency_ms,
            success=success,
            error=error,
            error_type=error_type,
            traceback_str=traceback_str,
        )
    except Exception as e:
        # 埋点自身出错绝不影响工具调用主流程
        logger.debug(f"[TOOL_TRACE] 记录工具调用失败(忽略): {e}")


class MCPToolRegistry:
    """管理所有 MCP 工具。"""

    def __init__(self):
        self._tools: dict[str, dict] = {}
        self._handlers: dict[str, callable] = {}
        # name -> (accepted_param_names | None, accepts_var_keyword)
        # accepted 为 None 表示无法内省（此时全传），用于过滤内部注入参数
        # （如 AgentLoop 注入的 user_id），避免 handler(**arguments) 因多余关键字抛 TypeError
        self._handler_params: dict[str, tuple] = {}

    def register(self, name: str, description: str, inputSchema: dict, handler: callable):
        self._tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": inputSchema,
        }
        self._handlers[name] = handler
        try:
            sig = inspect.signature(handler)
            accepted = set(sig.parameters)
            accepts_kw = any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            self._handler_params[name] = (accepted, accepts_kw)
        except (TypeError, ValueError):
            self._handler_params[name] = (None, True)
        logger.info(f"MCP tool registered: {name}")

    def list_tools(self) -> list[dict]:
        return list(self._tools.values())

    def get_tool(self, name: str) -> Optional[dict]:
        return self._tools.get(name)

    async def call_tool(self, name: str, arguments: dict,
                        progress_callback=None) -> dict:
        """调用工具并返回结果。

        B-026: 结构化日志记录工具名、参数、耗时、成功/失败。
        B-027: 统一错误处理——返回 error 而非抛出异常。

        progress_callback: 可选的中途进度回调 async (message: str) -> None。
        仅当 handler 签名声明了 progress_callback 参数时注入，长耗时工具
        （标注/对比/管线）在内部关键节点调用，实现前端实时进度推送。

        工具调用埋点：完整记录输入参数、输出结果、错误现场到
        logs/tool/tool_calls.log，并挂载到当前 Trace（trace_id 关联 LLM 调用）。
        caller/purpose 由各编排模式通过 tool_call_context 设置。
        """
        handler = self._handlers.get(name)
        if not handler:
            duration_ms = 0
            err_msg = f"工具 '{name}' 未注册"
            logger.warning(f"[MCP] tool={name} {err_msg}")
            _safe_record_tool_call(
                name, dict(arguments) if arguments else {}, None,
                duration_ms, success=False, error=err_msg, error_type="ToolNotRegistered",
            )
            return {"error": err_msg}

        # 过滤内部注入参数（如 AgentLoop 注入的 user_id），只传 handler 声明的参数，
        # 避免 handler(**arguments) 因多余关键字抛 TypeError 被静默吞掉
        accepted, accepts_kw = self._handler_params.get(name, (None, True))
        if accepted is not None and not accepts_kw:
            arguments = {k: v for k, v in arguments.items() if k in accepted}

        # 记录实际传给 handler 的参数快照（用于埋点和排查）。
        # 必须在注入 progress_callback 之前构建，避免函数对象进入埋点日志
        arguments_snapshot = dict(arguments) if arguments else {}

        # 进度回调注入：仅 handler 显式声明 progress_callback 时才传，
        # 回调异常不应打断工具主流程（包一层容错）
        if progress_callback is not None and (
            accepted is None or "progress_callback" in accepted
        ):
            async def _safe_progress(message: str):
                try:
                    await progress_callback(message)
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"[MCP] tool={name} progress_callback error: {e}")
            arguments["progress_callback"] = _safe_progress

        t0 = time.monotonic()
        try:
            result = await handler(**arguments)
            duration_ms = int((time.monotonic() - t0) * 1000)
            # B-026: 结构化工具调用日志（agent.log 仍保留精简版）
            logger.debug(
                f"[MCP] tool={name} duration_ms={duration_ms} success=True"
            )
            # 工具调用埋点：完整输入输出写入 logs/tool/tool_calls.log
            _safe_record_tool_call(
                name, arguments_snapshot, result,
                duration_ms, success=True,
            )
            return {"result": result}
        except (RuntimeError, ValueError, TypeError, httpx.HTTPError, ConnectionError) as e:
            duration_ms = int((time.monotonic() - t0) * 1000)
            logger.error(f"[MCP] tool={name} duration_ms={duration_ms} success=False error={e}")
            # 记录已知异常的完整现场（含 error_type，便于分类统计）
            _safe_record_tool_call(
                name, arguments_snapshot, None,
                duration_ms, success=False,
                error=str(e), error_type=type(e).__name__,
                traceback_str=traceback.format_exc(),
            )
            # 参数缺失的 TypeError 转译成 LLM 可直接执行的友好提示
            # （如 handler() missing 1 required positional argument: 'action'）
            _missing = re.search(
                r"missing (\d+) required positional argument[s]?: '([^']+)'", str(e)
            )
            if _missing:
                return {
                    "error": (
                        f"工具调用缺少必填参数「{_missing.group(2)}」（共缺 {_missing.group(1)} 个）。"
                        f"请对照该工具的 inputSchema 补全参数后重试。"
                    ),
                    "degraded": False,
                }
            return {"error": str(e), "degraded": True}
        except Exception as e:
            duration_ms = int((time.monotonic() - t0) * 1000)
            _full_tb = traceback.format_exc()
            logger.error(f"[MCP] tool={name} duration_ms={duration_ms} unexpected_error={e}\n{_full_tb}")
            # 记录未知异常的完整 traceback（最关键的可定位现场）
            _safe_record_tool_call(
                name, arguments_snapshot, None,
                duration_ms, success=False,
                error=str(e), error_type=type(e).__name__,
                traceback_str=_full_tb,
            )
            return {"error": f"{e}\n\nFull traceback:\n{_full_tb}", "degraded": True}

