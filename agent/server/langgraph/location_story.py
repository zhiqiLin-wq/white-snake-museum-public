"""地标叙事 StateGraph — detect → fetch → retrieve → classify_shift → [analyze_shift] → contextualize → compose。

⚠️ 活跃使用中: 仍被 supervisor.py / skills/pipeline.py / skills/executor.py / summarize_location.py 引用。
Agent Loop 的 ReAct 路径是主入口，但 Skills 系统和 summarize_location 工具仍直接调用此子图。
修改时注意同步更新所有调用方。

升级要点（agent-deep-design Phase 1）：
- 新增 detect_location 节点：精确匹配模糊地名
- 新增 classify_era_shift 条件边：判断是否经历景观移位
- 新增 analyze_era_shift 节点：分析跨朝代文本变化
- 支持 stream 流式输出 + checkpointer 持久化
"""
import json
import logging
from typing import Optional, AsyncGenerator
from langgraph.graph import StateGraph, END

from .state import LocationStoryState
from ..llm.json_utils import parse_llm_json

logger = logging.getLogger(__name__)


class LocationStoryWorkflow:
    """地标叙事生成工作流（升级版）。"""

    def __init__(self, llm, retriever, prompt_registry, node_client=None, checkpointer=None):
        self.llm = llm
        self.retriever = retriever
        self.prompts = prompt_registry
        self.node_client = node_client
        self._checkpointer = checkpointer
        self._graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(LocationStoryState)

        workflow.add_node("detect_location", self._detect_location)
        workflow.add_node("fetch_location_data", self._fetch_location_data)
        workflow.add_node("retrieve_literature", self._retrieve_literature)
        workflow.add_node("extract_characters", self._extract_characters)
        workflow.add_node("classify_era_shift", self._classify_era_shift)
        workflow.add_node("analyze_era_shift", self._analyze_era_shift)
        workflow.add_node("contextualize", self._contextualize)
        workflow.add_node("compose_narrative", self._compose_narrative)

        workflow.set_entry_point("detect_location")
        workflow.add_edge("detect_location", "fetch_location_data")
        workflow.add_edge("fetch_location_data", "retrieve_literature")
        workflow.add_edge("retrieve_literature", "extract_characters")
        workflow.add_edge("extract_characters", "classify_era_shift")

        # ★ 条件边：如果有景观移位 → 进入分析分支；否则跳过
        workflow.add_conditional_edges(
            "classify_era_shift",
            self._route_after_classify,
            {
                "analyze_shift": "analyze_era_shift",
                "skip_shift": "contextualize",
            }
        )
        workflow.add_edge("analyze_era_shift", "contextualize")
        workflow.add_edge("contextualize", "compose_narrative")
        workflow.add_edge("compose_narrative", END)

        kwargs = {}
        if self._checkpointer is not None:
            kwargs["checkpointer"] = self._checkpointer
        return workflow.compile(**kwargs)

    def _route_after_classify(self, state: LocationStoryState) -> str:
        has_shift = state.get("has_era_shift", False)
        return "analyze_shift" if has_shift else "skip_shift"

    async def _detect_location(self, state: LocationStoryState) -> dict:
        """从用户问题中精确提取目标地点（支持模糊匹配）。"""
        user_query = state.get("query", "")
        location_name = state.get("location_name", "")

        # 如果已经指定了明确的地点名，直接使用
        if location_name and location_name not in ("unknown", "未指定"):
            return {"location_name": location_name}

        # 用地名列表做模糊匹配
        from ..utils.location_names import LOCATION_ALIASES
        for canonical, aliases in LOCATION_ALIASES.items():
            for alias in aliases:
                if alias in user_query:
                    return {"location_name": canonical}

        # 如果都没匹配到，用 LLM 提取
        if not location_name:
            prompt = f"""从用户问题中提取目标地点名称。已知的白蛇传相关地点包括：
雷峰塔、断桥、西湖、金山寺、灵隐寺、望江楼、西子湖、青城山、峨眉山、龙虎山、承天寺、卧佛寺。

用户问题：{user_query}

请只返回地点名称，不要其他内容。如果无法确定，返回"未指定"。"""
            resp = await self.llm.generate(
                system="你是地名识别助手。只返回地名。",
                user=prompt,
                max_tokens=64,
                temperature=0.1,
            )
            extracted = resp.content.strip()
            return {"location_name": extracted if extracted else "未指定"}

        return {}

    async def _fetch_location_data(self, state: LocationStoryState) -> dict:
        """从 Node API 获取地标统计数据。"""
        location_name = state["location_name"]
        if self.node_client:
            locations = await self.node_client.get_locations()
            for loc in locations:
                if loc.get("name") == location_name:
                    return {"location_data": loc}

        return {"location_data": {
            "name": location_name,
            "totalMatch": 0,
            "totalWords": 0,
            "keywords": [],
        }}

    async def _retrieve_literature(self, state: LocationStoryState) -> dict:
        """检索与该地标相关的文献片段。"""
        location_name = state["location_name"]
        docs = await self.retriever.retrieve(location_name)
        return {"retrieved_docs": docs}

    async def _extract_characters(self, state: LocationStoryState) -> dict:
        """LLM 提取与该地标相关的角色和事件。"""
        docs = state.get("retrieved_docs", [])
        location_name = state["location_name"]

        docs_text = "\n\n".join(d.get("content", "")[:300] for d in docs[:5])
        prompt = f"""从以下文献片段中，提取与「{location_name}」相关的：
1. 出场角色
2. 发生的事件（按时间顺序）
3. 该地点在故事情节中的作用

文献片段：
{docs_text if docs_text else '无相关文献'}

请以 JSON 格式返回：
{{"characters": ["角色1", ...], "events": [{{"name": "事件名", "description": "简述"}}], "role": "该地点在故事中的作用"}}"""

        resp = await self.llm.generate(
            system="你是白蛇传文学研究专家。请从文献中提取信息。只返回 JSON。",
            user=prompt,
            max_tokens=1024,
            temperature=0.3,
        )
        data = parse_llm_json(resp.content)
        return {
            "characters": data.get("characters", []),
            "events": data.get("events", []),
        }

    async def _classify_era_shift(self, state: LocationStoryState) -> dict:
        """判断该地点是否经历了「景观移位」（如三塔→雷峰塔的变迁）。"""
        location_name = state["location_name"]
        docs = state.get("retrieved_docs", [])

        if not docs:
            return {"has_era_shift": False}

        docs_text = "\n".join(
            f"[{d.get('metadata', {}).get('chapter_title', '')}] {d.get('content', '')[:200]}"
            for d in docs[:6]
        )

        prompt = f"""请判断「{location_name}」在白蛇传文献中是否经历了明显的"景观移位"（即在不同朝代文本中，该地点的名称、功能、或叙事角色发生了显著变化）。

相关文献片段：
{docs_text}

请以 JSON 格式返回：
{{"has_shift": true/false, "reason": "简要说明是否有移位及表现"}}"""

        resp = await self.llm.generate(
            system="你是白蛇传文学地理研究专家。请判断地点是否经历景观移位。只返回 JSON。",
            user=prompt,
            max_tokens=256,
            temperature=0.3,
        )
        data = parse_llm_json(resp.content)
        return {
            "has_era_shift": data.get("has_shift", False),
            "shift_reason": data.get("reason", ""),
        }

    async def _analyze_era_shift(self, state: LocationStoryState) -> dict:
        """分析该地点在不同朝代的文本中如何变化。"""
        location_name = state["location_name"]
        docs = state.get("retrieved_docs", [])

        # 按朝代/章节分组
        era_groups: dict[str, list] = {}
        for d in docs:
            chapter_title = d.get("metadata", {}).get("chapter_title", "未知")
            if chapter_title not in era_groups:
                era_groups[chapter_title] = []
            era_groups[chapter_title].append(d.get("content", "")[:300])

        era_summary = "\n\n".join(
            f"【{title}】\n" + "\n---\n".join(texts[:2])
            for title, texts in era_groups.items()
        )

        prompt = f"""分析「{location_name}」在不同朝代/版本的白蛇传文本中如何变化：

{era_summary}

请分析：
1. 该地点在各文本中的名称是否一致
2. 该地点的叙事功能（镇压/相遇/避难等）是否变化
3. 该地点的意象（神圣/世俗/恐怖/浪漫等）是否转变

以 JSON 格式返回：
{{"shift_analysis": "综合分析（200字以内）", "timeline": [{{"era": "朝代/版本", "name": "地点名称", "function": "叙事功能", "imagery": "意象描述"}}]}}"""

        resp = await self.llm.generate(
            system="你是白蛇传文学地理研究专家。请分析地点在不同文本中的变迁。只返回 JSON。",
            user=prompt,
            max_tokens=1024,
            temperature=0.5,
        )
        data = parse_llm_json(resp.content)
        return {
            "shift_analysis": data.get("shift_analysis", ""),
            "era_timeline": data.get("timeline", []),
        }

    async def _contextualize(self, state: LocationStoryState) -> dict:
        """补充真实世界的历史文化背景。"""
        location_name = state["location_name"]
        prompt = f"""请简要介绍「{location_name}」在中国地理和历史文化中的地位（50-100字）。
不需要涉及白蛇传，只需要真实世界的地理文化信息。"""

        resp = await self.llm.generate(
            system="你是中国文化地理学者。请提供客观的地理文化背景。",
            user=prompt,
            max_tokens=512,
            temperature=0.5,
        )
        return {"cultural_context": resp.content}

    async def _compose_narrative(self, state: LocationStoryState) -> dict:
        """编织成连贯的叙事。"""
        location_name = state["location_name"]
        location_data = state.get("location_data", {})
        docs = state.get("retrieved_docs", [])

        chapters = list(set(
            d.get("metadata", {}).get("chapter_title", "")
            for d in docs
        ))

        system, user = self.prompts.render("location_narrative", {
            "location_name": location_name,
            "total_match": location_data.get("totalMatch", 0),
            "total_words": location_data.get("totalWords", 0),
            "keywords": location_data.get("keywords", []),
            "documents": docs,
        })
        sc = state.get("system_context", "")
        if sc:
            system += f"\n\n{sc}"
        resp = await self.llm.generate_with_fallback(system, user, max_tokens=8192, temperature=0.7)

        sources = []
        for d in docs[:5]:
            metadata = d.get("metadata", {})
            sources.append({
                "chapter_number": metadata.get("chapter_number", ""),
                "chapter_title": metadata.get("chapter_title", ""),
                "excerpt": d.get("content", "")[:200],
                "relevance_score": round(d.get("relevance_score", 0), 2),
            })

        return {
            "narrative": resp.content,
            "sources": sources,
            "total_tokens_used": resp.tokens,
            "degraded": resp.degraded,
        }

    async def run(self, location_name: str, query: str = "", thread_id: str = "default", system_context: str = "") -> dict:
        initial: LocationStoryState = {
            "location_name": location_name,
            "query": query,
            "thread_id": thread_id,
            "system_context": system_context,
            "retrieved_docs": [],
            "characters": [],
            "events": [],
            "cultural_context": "",
            "narrative": "",
            "sources": [],
            "tool_calls": [],
            "total_tokens_used": 0,
            "degraded": False,
            "has_era_shift": False,
        }
        config = {"configurable": {"thread_id": thread_id}} if self._checkpointer and thread_id else None
        return await self._graph.ainvoke(initial, config) if config else await self._graph.ainvoke(initial)

    async def run_stream(self, location_name: str, query: str = "", thread_id: str = "stream", system_context: str = "") -> AsyncGenerator[dict, None]:
        """流式执行地标叙事工作流 — 含实时进度。"""
        config = {"configurable": {"thread_id": thread_id}} if self._checkpointer else {}

        initial: LocationStoryState = {
            "location_name": location_name,
            "query": query,
            "thread_id": thread_id,
            "system_context": system_context,
            "retrieved_docs": [],
            "characters": [],
            "events": [],
            "cultural_context": "",
            "narrative": "",
            "sources": [],
            "tool_calls": [],
            "total_tokens_used": 0,
            "degraded": False,
            "has_era_shift": False,
        }

        # Step 1: 检测地点
        yield {"type": "status", "data": "Step 1/8: 识别目标地点..."}
        updates = await self._detect_location(initial)
        initial.update(updates)
        loc_name = initial["location_name"]
        yield {"type": "status", "data": f"Step 1/8: 目标地点 = {loc_name}"}

        # Step 2: 获取地点数据
        yield {"type": "status", "data": f"Step 2/8: 获取 {loc_name} 统计数据..."}
        updates = await self._fetch_location_data(initial)
        initial.update(updates)

        # Step 3: 检索文献
        yield {"type": "status", "data": f"Step 3/8: 检索 {loc_name} 相关文献..."}
        updates = await self._retrieve_literature(initial)
        initial.update(updates)
        doc_count = len(initial.get("retrieved_docs", []))
        yield {"type": "status", "data": f"Step 3/8: 找到 {doc_count} 段文献"}

        # Step 4: 提取角色
        yield {"type": "status", "data": "Step 4/8: 提取相关角色与事件..."}
        updates = await self._extract_characters(initial)
        initial.update(updates)
        characters = initial.get("characters", [])
        yield {"type": "status", "data": f"Step 4/8: 提取到 {len(characters)} 个角色"}

        # Step 5: 判断景观移位
        yield {"type": "status", "data": "Step 5/8: 判断是否经历景观移位..."}
        updates = await self._classify_era_shift(initial)
        initial.update(updates)
        has_shift = initial.get("has_era_shift", False)
        yield {"type": "status", "data": f"Step 5/8: {'检测到景观移位' if has_shift else '无显著景观移位'}"}

        # Step 6: 分析移位（如果有）
        if has_shift:
            yield {"type": "status", "data": "Step 6/8: 分析跨朝代景观变迁..."}
            updates = await self._analyze_era_shift(initial)
            initial.update(updates)
        else:
            yield {"type": "status", "data": "Step 6/8: 跳过（无景观移位）"}

        # Step 7: 补充文化背景
        yield {"type": "status", "data": f"Step 7/8: 补充 {loc_name} 历史文化背景..."}
        updates = await self._contextualize(initial)
        initial.update(updates)

        # Step 8: 生成叙事
        yield {"type": "status", "data": f"Step 8/8: 生成 {loc_name} 文学地理叙事..."}
        # 流式生成叙事内容
        docs = initial.get("retrieved_docs", [])
        location_data = initial.get("location_data", {})
        shift_analysis = initial.get("shift_analysis", "")
        cultural_context = initial.get("cultural_context", "")

        chapter_set = list(set(d.get("metadata", {}).get("chapter_title", "") for d in docs))
        system, user = self.prompts.render("location_narrative", {
            "location_name": loc_name,
            "total_match": location_data.get("totalMatch", 0),
            "total_words": location_data.get("totalWords", 0),
            "keywords": location_data.get("keywords", []),
            "documents": docs,
            "shift_analysis": shift_analysis,
            "cultural_context": cultural_context,
        })
        sc = initial.get("system_context", "")
        if sc:
            system += f"\n\n{sc}"

        chunk_count = 0
        async for text in await self.llm.generate_stream(system, user, max_tokens=8192, temperature=0.7):
            chunk_count += len(text)
            yield {"type": "chunk", "data": text}
        yield {"type": "status", "data": f"叙事生成完成 ({chunk_count} 字)"}

        # 整理引用
        sources = []
        for d in docs[:5]:
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
            "workflow": "location_story",
            "location": loc_name,
            "has_era_shift": has_shift,
            "sources_count": len(sources),
        }}
