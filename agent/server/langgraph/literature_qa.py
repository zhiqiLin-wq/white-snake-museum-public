"""文献问答 StateGraph (v6 简化版) — classify → retrieve → generate → format。

⚠️ 活跃使用中: 仍被 supervisor.py / skills/pipeline.py / skills/executor.py 引用。
Agent Loop 的 ReAct 路径是主入口，此工作流作为 Skills 系统和 Supervisor 非流式路径的回退选项。
修改时注意同步更新所有调用方。

assess → reformulate 循环已移入 AgentLoop 的 ReAct 循环 (U02-08)。
"""
import json
import logging
from typing import Optional, AsyncGenerator
from langgraph.graph import StateGraph, END

from .state import LiteratureQAState
from ..llm.json_utils import parse_llm_json
from ..llm.provider import LLMFatalError, RateLimitError, APITimeoutError

logger = logging.getLogger(__name__)


class LiteratureQAWorkflow:
    """文献问答工作流 (v6 简化版 — 线性管道)。"""

    def __init__(self, llm, retriever, prompt_registry, checkpointer=None):
        self.llm = llm
        self.retriever = retriever
        self.prompts = prompt_registry
        self._checkpointer = checkpointer
        self._graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(LiteratureQAState)

        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("format_response", self._format_response)

        workflow.set_entry_point("classify_intent")
        workflow.add_edge("classify_intent", "retrieve")
        workflow.add_edge("retrieve", "generate_answer")
        workflow.add_edge("generate_answer", "format_response")
        workflow.add_edge("format_response", END)

        kwargs = {}
        if self._checkpointer is not None:
            kwargs["checkpointer"] = self._checkpointer
        return workflow.compile(**kwargs)

    # _route_after_assess 已删除 (U02-08) — 评估/改写逻辑移入 AgentLoop

    async def _classify_intent(self, state: LiteratureQAState) -> dict:
        """意图分类 + 关键词提取（v4 上下文感知）。"""
        query = state["query"]
        messages = state.get("messages", [])

        # 构建对话历史
        history_parts = []
        for msg in messages[-8:]:
            role = msg.get("role", "") if isinstance(msg, dict) else ""
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            if role == "user":
                history_parts.append(f"[用户]: {content}")
            elif role == "assistant":
                history_parts.append(f"[助手]: {content[:200]}")
        conversation_history = "\n".join(history_parts[:-1]) if len(history_parts) > 1 else ""

        from ..config import settings as app_settings
        fast_model = app_settings.effective_fast_model
        if self.prompts is None:
            system = "你是白蛇传文本意图分类器。请将用户查询分类。只返回JSON：{\"intent\":\"factual\", \"keywords\":[...]}"
            user = f"查询：{query}\n历史：{conversation_history}"
        else:
            system, user = self.prompts.render("classifier", {
                "query": query,
                "conversation_history": conversation_history,
            })
        resp = await self.llm.generate(system, user, model=fast_model, max_tokens=256, temperature=0.3)

        # 尝试解析 JSON
        data = parse_llm_json(resp.content)
        return {
            "intent": data.get("intent", "factual"),
            "keywords": data.get("keywords", []),
        }

    async def _retrieve(self, state: LiteratureQAState) -> dict:
        """混合检索。"""
        query = state["query"]
        keywords = state.get("keywords", [])
        search_query = " ".join(keywords) if keywords else query

        iteration = state.get("search_iteration", 0)
        docs = await self.retriever.retrieve(search_query)
        return {
            "retrieved_docs": docs,
            "search_iteration": iteration + 1,
        }

    # _assess_sufficiency 和 _reformulate 已删除 (U02-08)
    # 评估/改写循环已移入 AgentLoop 的 ReAct 循环

    async def _generate_answer(self, state: LiteratureQAState) -> dict:
        """生成带引用的回答。"""
        docs = state.get("retrieved_docs", [])

        # 收集涉及的章节
        chapters = list(set(
            d.get("metadata", {}).get("chapter_title", "")
            for d in docs
        ))

        history_text = self._build_history_text(state.get("messages", []))
        if self.prompts is None:
            system = "你是白蛇传文献学者。请基于检索结果回答问题，引用具体章节。"
            user = f"问题：{state['query']}\n\n文献：{chr(10).join(d.get('content', '')[:500] for d in docs)}"
        else:
            system, user = self.prompts.render("qa_generator", {
                "query": state["query"],
                "documents": docs,
                "literature_title": "白蛇传文献选集",
                "chapters": chapters if chapters else ["全文"],
            })
        # 注入历史上下文到 system prompt
        if history_text:
            system += history_text
        # 注入记忆上下文（摘要 + Core Memory + 被动注入）(P-fix)
        system_context = state.get("system_context", "")
        if system_context:
            system += f"\n\n{system_context}"
        resp = await self.llm.generate_with_fallback(system, user, max_tokens=8192, temperature=0.7)
        return {
            "answer": resp.content,
            "degraded": resp.degraded,
            "total_tokens_used": resp.tokens,
        }

    async def _format_response(self, state: LiteratureQAState) -> dict:
        """整理引用来源。"""
        docs = state.get("retrieved_docs", [])
        sources = []
        for d in docs[:5]:
            metadata = d.get("metadata", {})
            sources.append({
                "chapter_number": metadata.get("chapter_number", ""),
                "chapter_title": metadata.get("chapter_title", ""),
                "excerpt": d.get("content", "")[:200],
                "relevance_score": round(d.get("relevance_score", d.get("score", 0)), 2),
            })
        return {"sources": sources}

    def _build_history_text(self, history: Optional[list] = None) -> str:
        """将历史消息列表格式化为上下文文本。"""
        if not history:
            return ""
        lines = ["\n\n## 对话历史上下文\n"]
        for msg in history:
            # 兼容 dict 和 LangGraph message 对象
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                # LangGraph message object (HumanMessage/AIMessage)
                role = msg.type
                content = msg.content
            elif isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
            else:
                continue
            label = "用户" if role in ("user", "human") else "AI助手" if role in ("assistant", "ai") else role
            lines.append(f"**{label}**: {content}")
        lines.append("\n请结合以上对话历史理解用户当前的问题。如果用户当前问题不完整（如使用代词'那里'、'他'等），请根据历史上下文推断其真实意图。")
        return "\n".join(lines)

    # ===== 公共接口 =====

    async def run(self, query: str, skill: Optional[str] = None, thread_id: Optional[str] = None, history: Optional[list] = None, system_context: str = "") -> dict:
        """执行工作流（非流式）。"""
        initial_state: LiteratureQAState = {
            "query": query,
            "skill": skill,
            "thread_id": thread_id or "default",
            "messages": history or [],
            "system_context": system_context,
            "search_iteration": 0,
            "sufficient": False,
            "retrieved_docs": [],
            "keywords": [],
            "answer": "",
            "sources": [],
            "tool_calls": [],
            "total_tokens_used": 0,
            "degraded": False,
            "error": None,
        }
        config = {"configurable": {"thread_id": thread_id}} if self._checkpointer and thread_id else None
        return await self._graph.ainvoke(initial_state, config) if config else await self._graph.ainvoke(initial_state)

    async def run_stream(self, query: str, skill: Optional[str] = None, history: Optional[list] = None, thread_id: str = "stream") -> AsyncGenerator[dict, None]:
        """执行工作流（流式）— 含详细的实时进度。"""
        initial_state: LiteratureQAState = {
            "query": query,
            "skill": skill,
            "thread_id": thread_id,
            "messages": history or [],
            "search_iteration": 0,
            "sufficient": False,
            "retrieved_docs": [],
            "keywords": [],
            "answer": "",
            "sources": [],
            "tool_calls": [],
            "total_tokens_used": 0,
            "degraded": False,
            "error": None,
        }

        # === 步骤 1: 意图分类 (快速模型, ~200ms) ===
        yield {"type": "status", "data": "Step 1/5: 分析问题意图 (快速模型)..."}
        state = initial_state.copy()
        updates = await self._classify_intent(state)
        state.update(updates)
        intent = state.get("intent", "unknown")
        keywords = state.get("keywords", [])
        intent_names = {"factual": "事实查询", "story": "故事讲述", "analysis": "文学分析", "comparison": "对比分析"}
        yield {"type": "status", "data": f"Step 1/5: 意图={intent_names.get(intent, intent)}, 关键词=[{' '.join(keywords[:5])}]"}

        # === 步骤 2: 检索 ===
        yield {"type": "status", "data": f"Step 2/5: 从文献库中语义检索..."}
        updates = await self._retrieve(state)
        state.update(updates)
        doc_count = len(state.get("retrieved_docs", []))
        if doc_count > 0:
            chapters_found = list(set(
                d.get("metadata", {}).get("chapter_title", "")
                for d in state.get("retrieved_docs", [])
            ))
            yield {"type": "status", "data": f"Step 2/5: 检索完成 -> 找到 {doc_count} 段文献 (涉及: {'、'.join(chapters_found[:3])})"}
        else:
            yield {"type": "status", "data": f"Step 2/5: 检索完成 -> 找到 {doc_count} 段文献"}

        # === 步骤 3: 流式生成（主力模型）=== (v6: 移除了 assess/reformulate 循环)
        yield {"type": "status", "data": f"Step 3/3: 调用主力模型流式生成 (基于 {doc_count} 段文献)..."}

        chapters = list(set(
            d.get("metadata", {}).get("chapter_title", "")
            for d in state.get("retrieved_docs", [])
        ))
        if self.prompts is None:
            system = "你是白蛇传文献学者。请基于检索结果回答问题，引用具体章节。"
            user = f"问题：{query}\n\n文献：{chr(10).join(d.get('content', '')[:500] for d in state.get('retrieved_docs', []))}"
        else:
            system, user = self.prompts.render("qa_generator", {
                "query": query,
                "documents": state.get("retrieved_docs", []),
                "literature_title": "白蛇传文献选集",
                "chapters": chapters if chapters else ["全文"],
            })
        # 注入历史上下文
        history_text = self._build_history_text(state.get("messages", []))
        if history_text:
            system += history_text

        chunk_count = 0
        async for text in await self.llm.generate_stream(system, user, max_tokens=8192, temperature=0.7):
            chunk_count += len(text)
            yield {"type": "chunk", "data": text}
        yield {"type": "status", "data": f"生成完成 (共 {chunk_count} 字) -> 整理引用..."}

        # 整理引用
        updates = await self._format_response(state)
        for src in updates.get("sources", []):
            yield {"type": "source", "data": src}

        state.update(updates)
        yield {"type": "status", "data": f"完成! 引用 {len(updates.get('sources', []))} 条文献"}

        yield {"type": "done", "data": {
            "workflow": "literature_qa",
            "degraded": state.get("degraded", False),
        }}


def jieba_lite(text: str) -> list[str]:
    """简单的中文分词（不依赖 jieba），用于 fallback。"""
    # 简单按标点和空格分割，取 2 字以上片段作为关键词
    import re
    words = re.split(r'[，。！？；：、\s]+', text)
    result = []
    for w in words:
        if len(w) >= 2:
            result.append(w)
        if len(result) >= 5:
            break
    return result if result else [text]
