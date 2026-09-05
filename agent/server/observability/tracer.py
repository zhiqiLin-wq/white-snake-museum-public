"""请求链路追踪 + LLM 调用全量 trace 记录 + 工具调用 trace 记录。

LLM trace 设计要点:
- contextvar 传递当前调用上下文（节点名/purpose/上下游）
- Provider 层一处埋点，自动记录每次 generate() 的全量输入输出
- 实时打 JSON 日志（[LLM_TRACE] 前缀）+ 挂载到内存 Trace 对象
- run_stream 结束时打印汇总摘要

工具调用 trace 设计要点:
- MCPRegistry.call_tool 一处埋点，自动记录每次工具调用的输入输出
- contextvar 传递调用方（agent_loop/langgraph_supervisor/skills_pipeline/evolution_workbench）与 purpose
- 实时打 JSON 日志（[TOOL_CALL] 前缀, logs/tool/tool_calls.log）+ 挂载到内存 Trace
- trace_id 贯穿 LLM 调用与工具调用，错误时可串联定位
"""
import json
import uuid
import time
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

current_trace: ContextVar[Optional["Trace"]] = ContextVar("trace", default=None)

# LLM 调用上下文：由各节点/MCP工具在调用 llm.generate() 前设置
# 存 {node, purpose, upstream, downstream}
current_llm_context: ContextVar[Optional[dict]] = ContextVar("llm_context", default=None)

# 工具调用上下文：由各编排模式（agent_loop/langgraph/skills/evolution）在调用工具前设置
# 存 {caller, purpose, upstream, downstream}
# MCPRegistry.call_tool 会读取它来标注本次工具调用的归属
current_tool_context: ContextVar[Optional[dict]] = ContextVar("tool_context", default=None)

# 工具调用结果在日志中保留的最大字符数（防止检索类工具的大结果撑爆日志）
TOOL_RESULT_LOG_MAX_CHARS = 5000


@dataclass
class Span:
    name: str
    start_time: float = field(default_factory=time.monotonic)
    end_time: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000 if self.end_time > 0 else 0


@dataclass
class LLMCallRecord:
    """单次 LLM 调用的完整记录（全量 prompt，不截断）。"""
    call_id: str                        # 序号 (llm_001, llm_002...)
    timestamp: str                       # ISO 格式
    node: str                            # 调用方：图节点名 或 MCP工具名
    purpose: str                        # 语义标签：decompose_motifs / judge_claim 等
    upstream: str                       # 上游模块
    downstream: str                     # 下游模块
    model: str
    system_prompt: str                  # 全量 system prompt
    user_prompt: str                    # 全量 user prompt
    response: str                       # 全量 AI 返回
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_total: int = 0
    latency_ms: float = 0.0
    temperature: float = 0.7
    max_tokens: int = 2048
    cached: bool = False                # 是否命中缓存
    degraded: bool = False              # 是否走了降级路径
    error: Optional[str] = None          # 失败时记录错误类型+消息


@dataclass
class ToolCallRecord:
    """单次工具调用的完整记录（输入参数 + 输出结果 + 错误现场）。

    用于定位工具编排链路中的错误：通过 trace_id 与 LLM 调用串联，
    可还原"哪次 LLM 决策 → 调用了哪个工具 → 传入什么参数 → 返回什么结果 → 是否失败"。
    """
    call_id: str                          # 序号 (tc_001, tc_002...)
    trace_id: str                         # 关联当前 Trace（串联 LLM 调用）
    timestamp: str                        # ISO 格式
    caller: str                           # 调用方：agent_loop / langgraph_supervisor / skills_pipeline / evolution_workbench
    purpose: str                          # 语义标签：工具名 或 retrieve_evidence / dynasty_compare 等
    tool_name: str                        # 被调用的工具名
    arguments: dict                       # 输入参数（全量）
    result: Any = None                    # 输出结果（超过 TOOL_RESULT_LOG_MAX_CHARS 会截断）
    result_truncated: bool = False       # result 是否被截断
    result_original_size: int = 0         # 原始结果大小（字符数），便于判断截断程度
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None           # 失败时的错误消息
    error_type: Optional[str] = None      # 失败时的异常类型名
    traceback: Optional[str] = None      # 失败时的完整 traceback（便于定位）
    upstream: str = ""                    # 上游模块
    downstream: str = ""                  # 下游模块


@dataclass
class Trace:
    trace_id: str
    conversation_id: str
    query: str
    spans: list[Span] = field(default_factory=list)
    total_tokens: int = 0
    degraded: bool = False
    # LLM trace 扩展
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    _llm_call_counter: int = 0
    # 工具调用 trace 扩展
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    _tool_call_counter: int = 0

    def add_llm_call(self, record: LLMCallRecord):
        """记录一次 LLM 调用，累加 token。"""
        self.llm_calls.append(record)
        self.total_tokens += record.tokens_total
        if record.degraded:
            self.degraded = True

    def llm_summary(self) -> dict:
        """生成 LLM 调用汇总摘要。"""
        by_purpose: dict[str, int] = {}
        by_node: dict[str, int] = {}
        total_latency = 0.0
        cached_count = 0
        degraded_count = 0
        error_count = 0
        for c in self.llm_calls:
            by_purpose[c.purpose] = by_purpose.get(c.purpose, 0) + 1
            by_node[c.node] = by_node.get(c.node, 0) + 1
            total_latency += c.latency_ms
            if c.cached:
                cached_count += 1
            if c.degraded:
                degraded_count += 1
            if c.error:
                error_count += 1
        return {
            "trace_id": self.trace_id,
            "total_calls": len(self.llm_calls),
            "total_tokens": self.total_tokens,
            "total_latency_ms": round(total_latency, 1),
            "cached_calls": cached_count,
            "degraded_calls": degraded_count,
            "error_calls": error_count,
            "by_purpose": by_purpose,
            "by_node": by_node,
        }

    def llm_full_export(self) -> list[dict]:
        """导出全量 LLM 调用记录（含完整 prompt）。"""
        result = []
        for c in self.llm_calls:
            d = {
                "call_id": c.call_id,
                "timestamp": c.timestamp,
                "node": c.node,
                "purpose": c.purpose,
                "upstream": c.upstream,
                "downstream": c.downstream,
                "model": c.model,
                "system_prompt": c.system_prompt,
                "user_prompt": c.user_prompt,
                "response": c.response,
                "tokens_in": c.tokens_in,
                "tokens_out": c.tokens_out,
                "tokens_total": c.tokens_total,
                "latency_ms": round(c.latency_ms, 1),
                "temperature": c.temperature,
                "max_tokens": c.max_tokens,
                "cached": c.cached,
                "degraded": c.degraded,
                "error": c.error,
            }
            result.append(d)
        return result

    # ===== 工具调用 trace =====

    def add_tool_call(self, record: ToolCallRecord):
        """记录一次工具调用。"""
        self.tool_calls.append(record)
        if not record.success:
            self.degraded = True

    def tool_summary(self) -> dict:
        """生成工具调用汇总摘要。"""
        by_tool: dict[str, int] = {}
        by_caller: dict[str, int] = {}
        total_latency = 0.0
        error_count = 0
        for t in self.tool_calls:
            by_tool[t.tool_name] = by_tool.get(t.tool_name, 0) + 1
            by_caller[t.caller] = by_caller.get(t.caller, 0) + 1
            total_latency += t.latency_ms
            if not t.success:
                error_count += 1
        return {
            "trace_id": self.trace_id,
            "total_calls": len(self.tool_calls),
            "total_latency_ms": round(total_latency, 1),
            "error_calls": error_count,
            "by_tool": by_tool,
            "by_caller": by_caller,
        }

    def tool_full_export(self) -> list[dict]:
        """导出全量工具调用记录（含输入参数和输出结果）。"""
        result = []
        for t in self.tool_calls:
            result.append({
                "call_id": t.call_id,
                "trace_id": t.trace_id,
                "timestamp": t.timestamp,
                "caller": t.caller,
                "purpose": t.purpose,
                "tool_name": t.tool_name,
                "arguments": t.arguments,
                "result": t.result,
                "result_truncated": t.result_truncated,
                "result_original_size": t.result_original_size,
                "latency_ms": round(t.latency_ms, 1),
                "success": t.success,
                "error": t.error,
                "error_type": t.error_type,
                "traceback": t.traceback,
                "upstream": t.upstream,
                "downstream": t.downstream,
            })
        return result


class Tracer:
    @staticmethod
    def start_trace(conversation_id: str, query: str) -> Trace:
        trace = Trace(
            trace_id=uuid.uuid4().hex[:12],
            conversation_id=conversation_id,
            query=query,
        )
        current_trace.set(trace)
        return trace

    @staticmethod
    def get_current() -> Optional[Trace]:
        return current_trace.get()

    @staticmethod
    def clear():
        current_trace.set(None)

    # ===== LLM trace 上下文管理 =====

    @staticmethod
    def set_llm_context(node: str, purpose: str,
                        upstream: str = "", downstream: str = ""):
        """设置当前 LLM 调用上下文。

        由图节点/MCP工具在调用 llm.generate() 前调用。
        Provider 层埋点会读取这个上下文来标注调用归属。
        """
        current_llm_context.set({
            "node": node,
            "purpose": purpose,
            "upstream": upstream,
            "downstream": downstream,
        })

    @staticmethod
    def get_llm_context() -> Optional[dict]:
        return current_llm_context.get()

    @staticmethod
    def clear_llm_context():
        current_llm_context.set(None)

    # ===== LLM 调用记录入口（由 Provider 层调用）=====

    @staticmethod
    def record_llm_call(
        model: str,
        system_prompt: str,
        user_prompt: str,
        response: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        tokens_total: int = 0,
        latency_ms: float = 0.0,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        cached: bool = False,
        degraded: bool = False,
        error: Optional[str] = None,
    ) -> Optional[LLMCallRecord]:
        """记录一次 LLM 调用。

        从 contextvar 读取调用上下文（node/purpose/upstream/downstream）。
        如果没有活跃的 Trace，只打日志不挂载（不影响非 trace 流程）。
        """
        ctx = current_llm_context.get() or {}
        node = ctx.get("node", "unknown")
        purpose = ctx.get("purpose", "unknown")
        upstream = ctx.get("upstream", "")
        downstream = ctx.get("downstream", "")

        trace = current_trace.get()
        call_seq = 0
        if trace:
            trace._llm_call_counter += 1
            call_seq = trace._llm_call_counter

        call_id = f"llm_{call_seq:03d}" if trace else f"llm_?"
        ts = datetime.now(timezone.utc).isoformat()

        record = LLMCallRecord(
            call_id=call_id,
            timestamp=ts,
            node=node,
            purpose=purpose,
            upstream=upstream,
            downstream=downstream,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=response,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_total=tokens_total,
            latency_ms=latency_ms,
            temperature=temperature,
            max_tokens=max_tokens,
            cached=cached,
            degraded=degraded,
            error=error,
        )

        # 通道 1：LLM 独立日志文件（logs/llm/llm_calls.log，专用 logger + propagate=False，不污染 agent.log）
        log_obj = {
            "call_id": call_id,
            "timestamp": ts,
            "node": node,
            "purpose": purpose,
            "upstream": upstream,
            "downstream": downstream,
            "model": model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response": response,
            "tokens_total": tokens_total,
            "latency_ms": round(latency_ms, 1),
            "cached": cached,
            "degraded": degraded,
            "error": error,
        }
        log_json = json.dumps(log_obj, ensure_ascii=False)
        try:
            from .logger import get_llm_logger
            llm_logger = get_llm_logger()
            llm_logger.info(log_json)  # ← 进 llm_calls.log（带 [LLM_CALL] 前缀，grep 友好）
        except Exception:
            # LLM logger 没初始化时 fallback 到普通 logger，不至于丢日志
            logger.info("[LLM_TRACE] %s", log_json)

        # 通道 2：挂载到内存 Trace 对象
        if trace:
            trace.add_llm_call(record)

        return record

    # ===== 工具调用 trace 上下文管理 =====

    @staticmethod
    def set_tool_context(caller: str, purpose: str,
                         upstream: str = "", downstream: str = ""):
        """设置当前工具调用上下文。

        由各编排模式（agent_loop/langgraph/skills/evolution）在调用工具前设置。
        MCPRegistry.call_tool 会读取它来标注本次工具调用的归属。
        """
        current_tool_context.set({
            "caller": caller,
            "purpose": purpose,
            "upstream": upstream,
            "downstream": downstream,
        })

    @staticmethod
    def get_tool_context() -> Optional[dict]:
        return current_tool_context.get()

    @staticmethod
    def clear_tool_context():
        current_tool_context.set(None)

    # ===== 工具调用记录入口（由 MCPRegistry.call_tool 调用）=====

    @staticmethod
    def record_tool_call(
        tool_name: str,
        arguments: dict,
        result: Any = None,
        latency_ms: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        traceback_str: Optional[str] = None,
    ) -> Optional[ToolCallRecord]:
        """记录一次工具调用（完整输入参数 + 输出结果 + 错误现场）。

        从 contextvar 读取调用上下文（caller/purpose/upstream/downstream）和 trace_id。
        如果没有活跃的 Trace，只打日志不挂载（不影响非 trace 流程）。

        result 超过 TOOL_RESULT_LOG_MAX_CHARS 会截断，保留原大小信息。
        """
        ctx = current_tool_context.get() or {}
        caller = ctx.get("caller", "unknown")
        purpose = ctx.get("purpose", tool_name)
        upstream = ctx.get("upstream", "")
        downstream = ctx.get("downstream", "")

        trace = current_trace.get()
        trace_id = trace.trace_id if trace else ""
        call_seq = 0
        if trace:
            trace._tool_call_counter += 1
            call_seq = trace._tool_call_counter

        call_id = f"tc_{call_seq:03d}" if trace else f"tc_?"
        ts = datetime.now(timezone.utc).isoformat()

        # 计算原始结果大小，必要时截断（防止检索类工具的大结果撑爆日志）
        result_for_log, result_truncated, original_size = _truncate_for_log(result)

        record = ToolCallRecord(
            call_id=call_id,
            trace_id=trace_id,
            timestamp=ts,
            caller=caller,
            purpose=purpose,
            tool_name=tool_name,
            arguments=arguments if isinstance(arguments, dict) else {"_raw": str(arguments)},
            result=result_for_log,
            result_truncated=result_truncated,
            result_original_size=original_size,
            latency_ms=latency_ms,
            success=success,
            error=error,
            error_type=error_type,
            traceback=traceback_str,
            upstream=upstream,
            downstream=downstream,
        )

        # 通道 1：工具调用独立日志文件（logs/tool/tool_calls.log）
        log_obj = {
            "call_id": call_id,
            "trace_id": trace_id,
            "timestamp": ts,
            "caller": caller,
            "purpose": purpose,
            "tool_name": tool_name,
            "arguments": record.arguments,
            "result": record.result,
            "result_truncated": result_truncated,
            "result_original_size": original_size,
            "latency_ms": round(latency_ms, 1),
            "success": success,
            "error": error,
            "error_type": error_type,
            "traceback": traceback_str,
            "upstream": upstream,
            "downstream": downstream,
        }
        log_json = json.dumps(log_obj, ensure_ascii=False, default=str)
        try:
            from .logger import get_tool_logger
            tool_logger = get_tool_logger()
            tool_logger.info(log_json)  # ← 进 tool_calls.log（带 [TOOL_CALL] 前缀）
        except Exception:
            logger.info("[TOOL_CALL] %s", log_json)

        # 通道 2：挂载到内存 Trace 对象
        if trace:
            trace.add_tool_call(record)

        return record


def _truncate_for_log(value: Any, max_chars: int = TOOL_RESULT_LOG_MAX_CHARS) -> tuple:
    """对工具结果做日志侧截断（保留原大小信息）。

    返回 (截断后的值, 是否截断, 原始字符数)。
    截断后追加 truncation 标记，便于在日志中识别。
    """
    try:
        if isinstance(value, str):
            size = len(value)
            if size > max_chars:
                return value[:max_chars] + f"\n...[truncated: {size - max_chars} chars]", True, size
            return value, False, size
        if value is None:
            return None, False, 0
        s = json.dumps(value, ensure_ascii=False, default=str)
        size = len(s)
        if size > max_chars:
            # 对 dict/list 截断：返回一个标记对象，保留原大小
            return {
                "_truncated": True,
                "_original_size": size,
                "_preview": s[:max_chars] + f"\n...[truncated: {size - max_chars} chars]",
            }, True, size
        return value, False, size
    except Exception:
        # 序列化失败兜底
        return {"_serialize_error": str(value)[:200]}, False, 0


class _SpanContext:
    def __init__(self, name: str):
        self.name = name
        self.span: Optional[Span] = None

    def __enter__(self):
        self.span = Span(name=self.name)
        trace = current_trace.get()
        if trace:
            trace.spans.append(self.span)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            self.span.end_time = time.monotonic()
            logger.debug(
                "span=%s duration_ms=%.1f",
                self.span.name,
                self.span.duration_ms,
            )
        return False


# ===== 便捷上下文管理器：在节点/工具中一行设置 LLM 调用上下文 =====

@contextmanager
def llm_call_context(node: str, purpose: str,
                      upstream: str = "", downstream: str = ""):
    """设置 LLM 调用上下文的上下文管理器。

    用法:
        with llm_call_context("test_one_claim", "judge_claim",
                               upstream="prepare_claims",
                               downstream="_route_after_one_claim"):
            await self.llm.generate(...)
    """
    token = current_llm_context.set({
        "node": node,
        "purpose": purpose,
        "upstream": upstream,
        "downstream": downstream,
    })
    try:
        yield
    finally:
        current_llm_context.reset(token)


# ===== 便捷上下文管理器：在编排模式调用工具前一行设置工具调用上下文 =====

@contextmanager
def tool_call_context(caller: str, purpose: str,
                      upstream: str = "", downstream: str = ""):
    """设置工具调用上下文的上下文管理器。

    用法:
        with tool_call_context("agent_loop", "dynasty_compare",
                                upstream="react_loop",
                                downstream="mcp_call_tool"):
            await self._call_tool(tc.name, tc.arguments)

    MCPRegistry.call_tool 会读取这个上下文，把本次工具调用标注到
    对应的 caller/purpose，并写入 logs/tool/tool_calls.log。
    """
    token = current_tool_context.set({
        "caller": caller,
        "purpose": purpose,
        "upstream": upstream,
        "downstream": downstream,
    })
    try:
        yield
    finally:
        current_tool_context.reset(token)
