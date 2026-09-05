"""技能 Pipeline — 串联多个 tool/workflow 步骤。"""
import logging
import json
import time
import traceback
import httpx
from typing import Any, AsyncGenerator

from ..llm.provider import LLMFatalError, RateLimitError, APITimeoutError
from ..observability.tracer import Tracer, llm_call_context, tool_call_context
from ..dispatch.tool_dispatcher import ToolDispatcher

logger = logging.getLogger(__name__)


def _record_tool(tool_name: str, arguments: dict, result, latency_ms: float,
                 success: bool, error=None, error_type=None):
    """记录工具调用到 logs/tool/tool_calls.log（埋点失败不影响主流程）。

    Pipeline 中 search_literature 等步骤直接调 retriever，不经 MCPRegistry，
    因此需要手动调用此函数记录，确保检索调用也有完整输入输出日志。
    """
    try:
        Tracer.record_tool_call(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            latency_ms=latency_ms,
            success=success,
            error=error,
            error_type=error_type,
            traceback_str=traceback.format_exc() if not success else None,
        )
    except Exception as e:
        logger.debug(f"[TOOL_TRACE] pipeline 记录工具调用失败(忽略): {e}")


class PipelineExecutor:
    """执行 Skill Pipeline：按顺序执行 step，前一步输出作为后一步输入。"""

    def __init__(self, llm, retriever, prompt_registry, mcp_registry=None, node_client=None, checkpointer=None):
        self.llm = llm
        self.retriever = retriever
        self.prompts = prompt_registry
        self.mcp_registry = mcp_registry
        # 统一工具调度层：tool 步骤的未硬编码工具经此真正执行（+ 埋点）
        self.dispatcher = ToolDispatcher(mcp_registry=mcp_registry, retriever=retriever)
        self.node_client = node_client
        self._checkpointer = checkpointer

    async def execute(
        self,
        pipeline_def: list[dict],
        variables: dict,
        stream_callback=None,
    ) -> dict:
        """执行管道。"""
        context = {"variables": variables}
        step_results = {}

        for i, step_def in enumerate(pipeline_def):
            step_name = step_def.get("step", f"step_{i}")
            step_type = step_def.get("type", "tool")

            if stream_callback:
                await stream_callback({
                    "type": "pipeline_step",
                    "step": step_name,
                    "status": "running",
                })

            try:
                result = await self._run_step(step_def, context)
                step_results[step_name] = result
                context[step_name] = result

                if stream_callback:
                    await stream_callback({
                        "type": "pipeline_step",
                        "step": step_name,
                        "status": "completed",
                        "summary": self._summarize(result),
                    })
            except (LLMFatalError, RateLimitError, APITimeoutError, RuntimeError, ValueError, httpx.HTTPError) as e:
                logger.error(f"Pipeline 步骤失败 [{step_name}]: {e}")
                if stream_callback:
                    await stream_callback({
                        "type": "pipeline_step",
                        "step": step_name,
                        "status": "error",
                        "error": str(e),
                    })
                step_results[step_name] = {"error": str(e)}

        return {
            "step_results": step_results,
            "final_context": context,
        }

    async def _run_step(self, step_def: dict, context: dict) -> Any:
        typ = step_def.get("type", "tool")

        if typ == "tool":
            tool_name = step_def.get("tool", "")
            params = self._render_params(step_def.get("params", {}), context)

            if tool_name == "search_literature":
                # 检索直接调 retriever，不经 MCPRegistry，需手动埋点
                q = params.get("query", "")
                top_k = params.get("top_k", 10)
                t0 = time.monotonic()
                try:
                    with tool_call_context(
                        caller="skills_pipeline", purpose="search_literature",
                        upstream="pipeline_step", downstream="retriever.retrieve",
                    ):
                        docs = await self.retriever.retrieve(q, top_k=top_k)
                    _record_tool("search_literature", {"query": q, "top_k": top_k},
                                 docs, (time.monotonic() - t0) * 1000, success=True)
                    return docs
                except Exception as e:
                    _record_tool("search_literature", {"query": q, "top_k": top_k},
                                 None, (time.monotonic() - t0) * 1000, success=False,
                                 error=str(e), error_type=type(e).__name__)
                    raise
            elif tool_name == "get_location_info":
                return params  # 简化处理
            else:
                # 修复：真正执行工具（经 ToolDispatcher 统一调度 + 自动埋点），
                # 而非返回 {"tool":..., "params":...} 占位字典导致后续步骤拿到空结果
                tool_result = await self.dispatcher.dispatch(
                    tool_name, params, caller="skills_pipeline",
                    purpose=tool_name,
                    upstream="pipeline_step", downstream="mcp_call_tool",
                )
                return tool_result.result if tool_result.success else {"error": tool_result.error}

        elif typ == "llm_task":
            # 修复：补设 llm_call_context，避免 LLM 调用被标记为 node=unknown/purpose=unknown
            # （违反硬约束：LLM 调用必须有非 unknown 的 node 和 purpose 标签）
            prompt_name = step_def.get("prompt", "")
            system, user = self.prompts.render(prompt_name, context)
            with llm_call_context(
                node="skills_pipeline", purpose=prompt_name or "llm_task",
                upstream="pipeline_step", downstream="llm.generate",
            ):
                resp = await self.llm.generate(system, user, max_tokens=2048, temperature=0.7)
            return {"content": resp.content, "tokens": resp.tokens}

        elif typ == "langgraph":
            workflow_name = step_def.get("workflow", "literature_qa")
            query = context.get("variables", {}).get("query", "")

            if workflow_name == "location_story":
                from ..langgraph.location_story import LocationStoryWorkflow
                location_name = context.get("variables", {}).get("location_name", "雷峰塔")
                wf = LocationStoryWorkflow(
                    self.llm, self.retriever, self.prompts,
                    node_client=self.node_client,
                    checkpointer=self._checkpointer,
                )
                result = await wf.run(location_name, query=query)
                result["answer"] = result.get("narrative", "")
                return result
            elif workflow_name == "comparison":
                from ..langgraph.comparison import ComparisonWorkflow
                wf = ComparisonWorkflow(
                    self.llm, self.retriever, self.prompts,
                    checkpointer=self._checkpointer,
                )
                result = await wf.run(aspect=query, query=query)
                result["answer"] = result.get("comparison_result", "")
                return result
            else:
                from ..langgraph.literature_qa import LiteratureQAWorkflow
                wf = LiteratureQAWorkflow(
                    self.llm, self.retriever, self.prompts,
                    checkpointer=self._checkpointer,
                )
                return await wf.run(query)

        return {}

    def _render_params(self, params: dict, context: dict) -> dict:
        """渲染参数中的模板变量 {{ ... }}。"""
        rendered = {}
        for k, v in params.items():
            if isinstance(v, str) and "{{" in v:
                # 简单模板替换
                for ctx_key, ctx_val in context.items():
                    placeholder = f"{{{{ {ctx_key} }}}}"
                    v = v.replace(placeholder, str(ctx_val))
            rendered[k] = v
        return rendered

    def _summarize(self, result: Any) -> str:
        """生成步骤结果摘要。"""
        if isinstance(result, list):
            return f"返回 {len(result)} 项"
        elif isinstance(result, dict):
            if "error" in result:
                return f"错误: {result['error']}"
            return f"完成 ({len(result)} 个字段)"
        return str(result)[:50]
