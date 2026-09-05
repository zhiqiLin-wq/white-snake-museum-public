"""技能执行调度器。"""
import logging
from typing import Optional, AsyncGenerator

from ..llm.provider import LLMFatalError, RateLimitError, APITimeoutError

logger = logging.getLogger(__name__)


class SkillExecutor:
    """执行单个技能（单 workflow 型 / 流式型）。"""

    def __init__(self, llm, retriever, prompt_registry, graph_builder=None, node_client=None, checkpointer=None):
        self.llm = llm
        self.retriever = retriever
        self.prompts = prompt_registry
        self.graph_builder = graph_builder
        self.node_client = node_client
        self._checkpointer = checkpointer

    async def execute(
        self,
        skill_def: dict,
        user_query: str,
        stream: bool = False,
    ) -> dict:
        """执行技能并返回结果。"""
        workflow_name = skill_def.get("workflow", "literature_qa")
        system_prompt = skill_def.get("system_prompt", "")
        tools_config = skill_def.get("tools", [])

        # 1. 预执行 tools（提前检索）
        pre_results = {}
        for tool_cfg in tools_config:
            for tool_name, params in tool_cfg.items():
                try:
                    # 调用 MCP 工具
                    if tool_name == "search_literature":
                        docs = await self.retriever.retrieve(
                            params.get("query", user_query),
                            top_k=params.get("top_k", 10),
                        )
                        pre_results["search_literature"] = docs
                    elif tool_name == "get_location_info":
                        # 通过 node_client 获取
                        pre_results["get_location_info"] = {"name": params.get("location_name", "")}
                except (RuntimeError, ValueError, KeyError) as e:
                    logger.warning(f"预执行工具失败 [{tool_name}]: {e}")

        # 2. 运行 workflow
        workflow_name = skill_def.get("workflow", "literature_qa")

        if workflow_name == "location_story":
            from ..langgraph.location_story import LocationStoryWorkflow
            location_name = pre_results.get("get_location_info", {}).get("name", "")
            if not location_name:
                # 从查询中提取地名
                from ..utils.location_names import LOCATION_ALIASES
                for canonical, aliases in LOCATION_ALIASES.items():
                    for alias in aliases:
                        if alias in user_query:
                            location_name = canonical
                            break
                    if location_name:
                        break
            wf = LocationStoryWorkflow(
                self.llm, self.retriever, self.prompts,
                node_client=self.node_client,
                checkpointer=self._checkpointer,
            )
            result = await wf.run(location_name or "雷峰塔", query=user_query)
            result["answer"] = result.get("narrative", "")
        elif workflow_name == "comparison":
            from ..langgraph.comparison import ComparisonWorkflow
            wf = ComparisonWorkflow(
                self.llm, self.retriever, self.prompts,
                checkpointer=self._checkpointer,
            )
            aspect = skill_def.get("tools", [{}])[0].get("compare_chapters", {}).get("aspect", user_query) if skill_def.get("tools") else user_query
            result = await wf.run(aspect=aspect, query=user_query)
            result["answer"] = result.get("comparison_result", "")
        else:
            from ..langgraph.literature_qa import LiteratureQAWorkflow
            wf = LiteratureQAWorkflow(
                self.llm, self.retriever, self.prompts,
                checkpointer=self._checkpointer,
            )
            result = await wf.run(user_query, skill=skill_def.get("name"))

        # 如果有自定义 system_prompt，叠加到结果
        if system_prompt and result.get("answer"):
            # 在回答前添加技能特定的上下文提示
            result["skill_used"] = skill_def.get("name")
            result["skill_display"] = skill_def.get("display_name", "")

        result["pre_tool_results"] = {
            k: len(v) if isinstance(v, list) else v
            for k, v in pre_results.items()
        }

        return result

    async def execute_stream(
        self,
        skill_def: dict,
        user_query: str,
    ) -> AsyncGenerator[dict, None]:
        """流式执行技能。"""
        workflow_name = skill_def.get("workflow", "literature_qa")
        system_prompt = skill_def.get("system_prompt", "")

        # 发送技能激活事件
        yield {
            "type": "skill_activated",
            "data": {
                "name": skill_def.get("name"),
                "display_name": skill_def.get("display_name", ""),
                "icon": skill_def.get("icon", "📖"),
            },
        }

        # 预执行 tools
        tools_config = skill_def.get("tools", [])
        for tool_cfg in tools_config:
            for tool_name, params in tool_cfg.items():
                yield {"type": "status", "data": f"正在执行: {tool_name}..."}
                if tool_name == "search_literature":
                    try:
                        docs = await self.retriever.retrieve(
                            params.get("query", user_query),
                            top_k=params.get("top_k", 10),
                        )
                        yield {"type": "status", "data": f"预检索完成: {len(docs)} 段文献"}
                    except (RuntimeError, httpx.HTTPError) as e:
                        yield {"type": "status", "data": f"预检索失败: {e}"}

        # 如果有自定义 system_prompt，用 skill 的 prompt 直接生成
        if system_prompt:
            docs = getattr(self, '_last_docs', [])
            sys_prompt = system_prompt
            user_prompt = f"用户问题：{user_query}\n\n请基于文献回答。"
            try:
                async for text in await self.llm.generate_stream(sys_prompt, user_prompt, max_tokens=2048, temperature=0.7):
                    yield {"type": "chunk", "data": text}
            except (LLMFatalError, RateLimitError, APITimeoutError) as e:
                yield {"type": "chunk", "data": f"\n（生成中断: {e}）"}
        elif workflow_name == "location_story":
            from ..langgraph.location_story import LocationStoryWorkflow
            from ..utils.location_names import LOCATION_ALIASES

            location_name = ""
            for canonical, aliases in LOCATION_ALIASES.items():
                for alias in aliases:
                    if alias in user_query:
                        location_name = canonical
                        break
                if location_name:
                    break

            wf = LocationStoryWorkflow(
                self.llm, self.retriever, self.prompts,
                node_client=self.node_client,
                checkpointer=self._checkpointer,
            )
            async for event in wf.run_stream(location_name or "未知地点", query=user_query):
                if event.get("type") == "done":
                    continue  # 由 SkillExecutor 统一发出 done
                yield event
        elif workflow_name == "comparison":
            from ..langgraph.comparison import ComparisonWorkflow
            wf = ComparisonWorkflow(
                self.llm, self.retriever, self.prompts,
                checkpointer=self._checkpointer,
            )
            async for event in wf.run_stream(aspect=user_query, query=user_query):
                if event.get("type") == "done":
                    continue
                yield event
        else:
            # 走默认 workflow
            from ..langgraph.literature_qa import LiteratureQAWorkflow
            wf = LiteratureQAWorkflow(
                self.llm, self.retriever, self.prompts,
                checkpointer=self._checkpointer,
            )
            async for event in wf.run_stream(user_query, skill=skill_def.get("name")):
                if event.get("type") == "done":
                    continue
                yield event

        yield {"type": "done", "data": {
            "workflow": workflow_name,
            "skill": skill_def.get("name"),
        }}
