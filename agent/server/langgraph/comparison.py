"""对比分析 StateGraph — parse_query → auto_select_texts → retrieve_per_topic → extract_contrast → generate_comparison。

⚠️ 活跃使用中: 仍被 supervisor.py / skills/pipeline.py / skills/executor.py / compare_chapters.py 引用。
Agent Loop 的 ReAct 路径是主入口，但 Skills 系统和 compare_chapters 工具仍直接调用此子图。
修改时注意同步更新所有调用方。

升级要点（agent-deep-design Phase 1）：
- 新增 parse_comparison_query：提取对比对象、维度、时间范围
- 新增 auto_select_texts：根据时间范围自动选择对应朝代文本
- retrieve_per_topic 改为并行检索
- extract_contrast 强制输出结构化对比表
- generate_comparison 区分三层差异（表层/中层/深层）
- 支持 stream 流式输出 + checkpointer 持久化
"""
import json
import logging
import asyncio
from typing import Optional, AsyncGenerator
from langgraph.graph import StateGraph, END

from .state import ComparisonState
from ..llm.json_utils import parse_llm_json
from ..llm.provider import LLMFatalError, RateLimitError, APITimeoutError

logger = logging.getLogger(__name__)

# 论文中的6篇核心文本 → 朝代映射
ERA_TEXT_MAP = {
    "唐": ["唐代《李黄》"],
    "宋": ["宋代《西湖三塔记》"],
    "明": ["明代《白娘子永镇雷峰塔》"],
    "清": ["清代《雷峰塔传奇》"],
}


class ComparisonWorkflow:
    """跨章节/跨角色对比分析工作流（升级版）。"""

    def __init__(self, llm, retriever, prompt_registry, checkpointer=None):
        self.llm = llm
        self.retriever = retriever
        self.prompts = prompt_registry
        self._checkpointer = checkpointer
        self._graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(ComparisonState)

        workflow.add_node("parse_comparison_query", self._parse_comparison_query)
        workflow.add_node("auto_select_texts", self._auto_select_texts)
        workflow.add_node("retrieve_per_topic", self._retrieve_per_topic)
        workflow.add_node("extract_contrast", self._extract_contrast)
        workflow.add_node("generate_comparison", self._generate_comparison)

        workflow.set_entry_point("parse_comparison_query")
        workflow.add_edge("parse_comparison_query", "auto_select_texts")
        workflow.add_edge("auto_select_texts", "retrieve_per_topic")
        workflow.add_edge("retrieve_per_topic", "extract_contrast")
        workflow.add_edge("extract_contrast", "generate_comparison")
        workflow.add_edge("generate_comparison", END)

        kwargs = {}
        if self._checkpointer is not None:
            kwargs["checkpointer"] = self._checkpointer
        return workflow.compile(**kwargs)

    # ===== 节点实现 =====

    async def _parse_comparison_query(self, state: ComparisonState) -> dict:
        """从用户问题提取：对比对象、维度、时间范围。"""
        query = state.get("query", "")
        aspect = state.get("aspect", "")

        search_query = query or aspect
        if not search_query:
            return {"topics": [], "compare_dimensions": [], "era_range": []}

        prompt = f"""分析以下对比查询，提取关键信息：

用户问题：{search_query}

请以 JSON 格式返回：
{{
  "compare_objects": ["对比对象1", "对比对象2", ...],
  "dimensions": ["对比维度1", "对比维度2", ...],
  "era_range": ["涉及的朝代1", ...],
  "comparison_type": "character|location|theme|other"
}}

对比维度可以是：形象特征、叙事功能、文化内涵、时代背景、宗教色彩等。
朝代识别：唐/宋/明/清，未明确指定则为空数组。"""

        resp = await self.llm.generate(
            system="你是比较文学研究专家。请解析对比查询。只返回 JSON。",
            user=prompt,
            max_tokens=512,
            temperature=0.3,
        )
        data = parse_llm_json(resp.content)
        return {
            "topics": data.get("compare_objects", []),
            "compare_dimensions": data.get("dimensions", []),
            "era_range": data.get("era_range", []),
            "comparison_type": data.get("comparison_type", "other"),
        }

    async def _auto_select_texts(self, state: ComparisonState) -> dict:
        """根据时间范围自动选择对应朝代的文本。"""
        era_range = state.get("era_range", [])
        topics = state.get("topics", [])

        selected_chapters = []
        if era_range:
            for era in era_range:
                if era in ERA_TEXT_MAP:
                    selected_chapters.extend(ERA_TEXT_MAP[era])

        # 如果用户没有指定朝代但指定了对比对象，尝试从对比对象反推
        if not selected_chapters:
            selected_chapters = ["全文"]

        return {
            "chapters": selected_chapters,
            "topics": topics if topics else state.get("topics", ["白蛇传"])
        }

    async def _retrieve_per_topic(self, state: ComparisonState) -> dict:
        """为每个主题并行检索文献。"""
        topics = state.get("topics", [])
        aspect = state.get("aspect", "")
        query = state.get("query", "")

        if not topics:
            topics = [aspect or query or "白蛇传"]

        async def retrieve_one(topic: str) -> tuple[str, list]:
            search_query = f"{topic} {aspect} {query}"
            docs = await self.retriever.retrieve(search_query)
            return topic, docs

        # 并行检索
        results = await asyncio.gather(*[retrieve_one(t) for t in topics])
        per_topic = {topic: docs for topic, docs in results}

        return {"per_topic_docs": per_topic}

    async def _extract_contrast(self, state: ComparisonState) -> dict:
        """提取差异点 — 强制输出结构化对比表。"""
        per_topic = state.get("per_topic_docs", {})
        dimensions = state.get("compare_dimensions", ["综合对比"])

        all_docs = []
        for topic, docs in per_topic.items():
            for d in docs:
                all_docs.append({
                    **d,
                    "metadata": {**(d.get("metadata", {})), "topic": topic},
                })

        docs_text = "\n\n".join(
            f"【{d.get('metadata', {}).get('topic', '')}】"
            f"[{d.get('metadata', {}).get('chapter_title', '')}] "
            f"{d.get('content', '')[:400]}"
            for d in all_docs[:12]
        )

        dims_str = "、".join(dimensions)
        prompt = f"""对比维度：{dims_str}

相关文献：
{docs_text if docs_text else '无相关文献'}

请从以下三个层次提取差异，以 JSON 格式返回：
{{
  "surface": {{"description": "表层差异（情节、人物行为的直接不同）", "items": [{{"dimension": "维度", "detail": "具体差异"}}]}},
  "narrative": {{"description": "中层差异（叙事策略、视角、结构的差异）", "items": [{{"dimension": "维度", "detail": "具体差异"}}]}},
  "deep": {{"description": "深层差异（时代精神、文化观念、宗教思想的演变）", "items": [{{"dimension": "维度", "detail": "具体差异"}}]}}
}}"""

        resp = await self.llm.generate(
            system="你是比较文学研究专家。请从三层差异维度提取文本对比信息。只返回 JSON。",
            user=prompt,
            max_tokens=1536,
            temperature=0.3,
        )
        data = parse_llm_json(resp.content)
        return {"contrasts": [
            {"level": "surface", "label": "表层差异", "data": data.get("surface", {})},
            {"level": "narrative", "label": "中层差异", "data": data.get("narrative", {})},
            {"level": "deep", "label": "深层差异", "data": data.get("deep", {})},
        ]}

    async def _generate_comparison(self, state: ComparisonState) -> dict:
        """生成对比分析报告。"""
        per_topic = state.get("per_topic_docs", {})
        contrasts = state.get("contrasts", [])
        topics = state.get("topics", [])
        dimensions = state.get("compare_dimensions", [])

        all_docs = []
        for topic, docs in per_topic.items():
            all_docs.extend(docs)

        # 构建对比表文本
        contrast_text = ""
        for c in contrasts:
            label = c.get("label", "")
            data = c.get("data", {})
            items = data.get("items", []) if isinstance(data, dict) else []
            if items:
                contrast_text += f"\n### {label}\n"
                for item in items:
                    contrast_text += f"- [{item.get('dimension', '')}] {item.get('detail', '')}\n"

        system, user = self.prompts.render("comparison", {
            "aspect": " vs ".join(topics) if topics else "白蛇传对比分析",
            "chapters": state.get("chapters", ["全文"]),
            "documents": all_docs,
        })
        # 增强 user prompt 加入对比表
        user += f"\n\n## 已提取的结构化差异\n{contrast_text}"

        resp = await self.llm.generate_with_fallback(system, user, max_tokens=8192, temperature=0.5)

        sources = []
        for d in all_docs[:5]:
            metadata = d.get("metadata", {})
            sources.append({
                "chapter_number": metadata.get("chapter_number", ""),
                "chapter_title": metadata.get("chapter_title", ""),
                "excerpt": d.get("content", "")[:200],
                "relevance_score": round(d.get("relevance_score", 0), 2),
            })

        return {
            "comparison_result": resp.content,
            "sources": sources,
            "total_tokens_used": resp.tokens,
            "degraded": resp.degraded,
        }

    # ===== 公共接口 =====

    async def run(self, aspect: str, query: str = "", chapters: list[str] | None = None,
                  thread_id: str = "default") -> dict:
        initial: ComparisonState = {
            "query": query,
            "aspect": aspect,
            "chapters": chapters or [],
            "topics": [],
            "compare_dimensions": [],
            "era_range": [],
            "per_topic_docs": {},
            "contrasts": [],
            "comparison_result": "",
            "sources": [],
            "tool_calls": [],
            "total_tokens_used": 0,
            "degraded": False,
            "thread_id": thread_id,
        }
        config = {"configurable": {"thread_id": thread_id}} if self._checkpointer and thread_id else None
        return await self._graph.ainvoke(initial, config) if config else await self._graph.ainvoke(initial)

    async def run_stream(self, aspect: str, query: str = "", chapters: list[str] | None = None,
                         thread_id: str = "stream") -> AsyncGenerator[dict, None]:
        """流式执行对比分析工作流。"""
        initial: ComparisonState = {
            "query": query,
            "aspect": aspect,
            "chapters": chapters or [],
            "topics": [],
            "compare_dimensions": [],
            "era_range": [],
            "per_topic_docs": {},
            "contrasts": [],
            "comparison_result": "",
            "sources": [],
            "tool_calls": [],
            "total_tokens_used": 0,
            "degraded": False,
            "thread_id": thread_id,
        }

        # Step 1: 解析对比查询
        yield {"type": "status", "data": "Step 1/5: 解析对比查询..."}
        updates = await self._parse_comparison_query(initial)
        initial.update(updates)
        topics = initial.get("topics", [])
        dimensions = initial.get("compare_dimensions", [])
        yield {"type": "status", "data": f"Step 1/5: 对比对象=[{' vs '.join(topics) if topics else aspect}], 维度=[{'、'.join(dimensions[:3]) if dimensions else '综合'}]"}

        # Step 2: 选择文本
        yield {"type": "status", "data": "Step 2/5: 匹配对应朝代文本..."}
        updates = await self._auto_select_texts(initial)
        initial.update(updates)
        chapters = initial.get("chapters", [])
        yield {"type": "status", "data": f"Step 2/5: 选定文本={chapters}"}

        # Step 3: 并行检索
        yield {"type": "status", "data": f"Step 3/5: 并行检索 {len(topics)} 个主题..."}
        updates = await self._retrieve_per_topic(initial)
        initial.update(updates)
        total_docs = sum(len(v) for v in initial.get("per_topic_docs", {}).values())
        yield {"type": "status", "data": f"Step 3/5: 并行检索完成 → 共 {total_docs} 段文献"}

        # Step 4: 提取差异
        yield {"type": "status", "data": "Step 4/5: 提取三层差异（表层/中层/深层）..."}
        updates = await self._extract_contrast(initial)
        initial.update(updates)
        contrasts = initial.get("contrasts", [])
        yield {"type": "status", "data": f"Step 4/5: 提取完成 → {len(contrasts)} 层差异"}

        # Step 5: 流式生成对比报告
        yield {"type": "status", "data": "Step 5/5: 生成对比分析报告..."}

        all_docs = []
        for topic, docs in initial.get("per_topic_docs", {}).items():
            all_docs.extend(docs)

        contrast_text = ""
        for c in contrasts:
            label = c.get("label", "")
            data = c.get("data", {})
            items = data.get("items", []) if isinstance(data, dict) else []
            if items:
                contrast_text += f"\n### {label}\n"
                for item in items:
                    contrast_text += f"- [{item.get('dimension', '')}] {item.get('detail', '')}\n"

        system, user = self.prompts.render("comparison", {
            "aspect": " vs ".join(topics) if topics else aspect,
            "chapters": chapters or ["全文"],
            "documents": all_docs,
        })
        user += f"\n\n## 已提取的结构化差异\n{contrast_text}"

        chunk_count = 0
        async for text in await self.llm.generate_stream(system, user, max_tokens=8192, temperature=0.5):
            chunk_count += len(text)
            yield {"type": "chunk", "data": text}
        yield {"type": "status", "data": f"对比报告完成 ({chunk_count} 字)"}

        # 整理引用
        sources = []
        for d in all_docs[:5]:
            metadata = d.get("metadata", {})
            sources.append({
                "chapter_number": metadata.get("chapter_number", ""),
                "chapter_title": metadata.get("chapter_title", ""),
                "excerpt": d.get("content", "")[:200],
                "relevance_score": round(d.get("relevance_score", 0), 2),
            })
        for src in sources:
            yield {"type": "source", "data": src}

        yield {"type": "done", "data": {
            "workflow": "comparison",
            "topics": topics,
            "contrast_levels": len(contrasts),
            "sources_count": len(sources),
        }}
