"""[废弃: v4.0 — 由 Agent Loop 替代] 旅行规划 StateGraph — parse_travel_query → gather_attractions → retrieve_literature
→ generate_weather → generate_hotels → generate_transportation → optimize_route → compose_itinerary。

完整功能：天气查询、酒店搜索、交通规划、游览顺序优化。
"""
import json
import logging
from typing import Optional, AsyncGenerator
from langgraph.graph import StateGraph, END

from .state import ItineraryState
from ..llm.json_utils import parse_llm_json

logger = logging.getLogger(__name__)

# 城市 → 景点映射
CITY_ATTRACTIONS = {
    "杭州": [
        {"name": "雷峰塔", "category": "核心景点", "visit_time": "2-3小时", "best_time": "傍晚（雷峰夕照）"},
        {"name": "断桥", "category": "核心景点", "visit_time": "30分钟-1小时", "best_time": "清晨或雪后"},
        {"name": "西湖", "category": "核心景区", "visit_time": "半天-全天", "best_time": "春秋两季"},
        {"name": "灵隐寺", "category": "文化古迹", "visit_time": "2-3小时", "best_time": "上午"},
        {"name": "望江楼", "category": "文化古迹", "visit_time": "1小时", "best_time": "傍晚"},
    ],
    "镇江": [
        {"name": "金山寺", "category": "核心景点", "visit_time": "2-3小时", "best_time": "上午"},
    ],
    "成都": [
        {"name": "青城山", "category": "核心景点", "visit_time": "半天-全天", "best_time": "春夏"},
        {"name": "峨眉山", "category": "核心景点", "visit_time": "1-2天", "best_time": "春秋"},
    ],
}

# 默认杭州景点（未识别城市时）
DEFAULT_ATTRACTIONS = CITY_ATTRACTIONS.get("杭州", []) + CITY_ATTRACTIONS.get("镇江", [])


class ItineraryPlannerWorkflow:
    """旅行规划工作流（完整版）— 天气 · 酒店 · 交通 · 路线优化。"""

    def __init__(self, llm, retriever, prompt_registry, node_client=None, checkpointer=None, mcp_registry=None):
        self.llm = llm
        self.retriever = retriever
        self.prompts = prompt_registry
        self.node_client = node_client
        self._checkpointer = checkpointer
        self.mcp = mcp_registry
        self._graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(ItineraryState)

        workflow.add_node("parse_travel_query", self._parse_travel_query)
        workflow.add_node("gather_attractions", self._gather_attractions)
        workflow.add_node("retrieve_literature", self._retrieve_literature)
        workflow.add_node("generate_weather", self._generate_weather)
        workflow.add_node("generate_hotels", self._generate_hotels)
        workflow.add_node("generate_transportation", self._generate_transportation)
        workflow.add_node("optimize_route", self._optimize_route)
        workflow.add_node("compose_itinerary", self._compose_itinerary)

        workflow.set_entry_point("parse_travel_query")
        workflow.add_edge("parse_travel_query", "gather_attractions")
        workflow.add_edge("gather_attractions", "retrieve_literature")
        workflow.add_edge("retrieve_literature", "generate_weather")
        workflow.add_edge("generate_weather", "generate_hotels")
        workflow.add_edge("generate_hotels", "generate_transportation")
        workflow.add_edge("generate_transportation", "optimize_route")
        workflow.add_edge("optimize_route", "compose_itinerary")
        workflow.add_edge("compose_itinerary", END)

        kwargs = {}
        if self._checkpointer is not None:
            kwargs["checkpointer"] = self._checkpointer
        return workflow.compile(**kwargs)

    # ===== 节点实现 =====

    async def _parse_travel_query(self, state: ItineraryState) -> dict:
        """从用户查询中提取结构化旅行参数（使用快速模型）— 含精确日期。"""
        query = state.get("query", "")
        if not query:
            return {"destination_city": "杭州", "duration_days": 1, "interests": ["白蛇传"], "budget_level": "中档"}

        from datetime import datetime, timedelta
        now = datetime.now()
        today_str = now.strftime("%Y年%m月%d日")
        tomorrow_str = (now + timedelta(days=1)).strftime("%Y年%m月%d日")
        next_week_str = (now + timedelta(days=7)).strftime("%Y年%m月%d日")
        current_season = {
            1:"冬季",2:"冬季",3:"春季",4:"春季",5:"春季",
            6:"夏季",7:"夏季",8:"夏季",9:"秋季",10:"秋季",11:"秋季",12:"冬季"
        }.get(now.month, "")

        prompt = f"""分析以下用户的旅行规划需求，提取关键信息并以 JSON 返回。

## ★ 当前时间参考（必须据此解析相对时间词）
- 今天: {today_str} ({now.strftime('%A')})
- 明天: {tomorrow_str}
- 一周后: {next_week_str}
- 当前季节: {current_season}

用户查询：{query}

返回 JSON 格式：
{{
  "destination_city": "目标城市（杭州/镇江/成都/未指定）",
  "duration_days": 天数（整数，未指定则默认 1），
  "interests": ["兴趣标签1", "兴趣标签2", ...],
  "budget_level": "经济型/中档/高端（未指定默认中档）",
  "travel_style": "深度文化/休闲观光/打卡快游/亲子家庭（未指定默认深度文化）",
  "travel_dates": "具体的计划出行日期（如用户说'明天'则填{tomorrow_str}，未指定则为空字符串）"
}}

兴趣标签可选：白蛇传、佛教文化、自然风光、美食、摄影、古建筑、民俗。
城市识别：提到西湖/雷峰塔/断桥/灵隐→杭州，提到金山/金山寺→镇江，提到青城山/峨眉→成都。"""

        from ..config import settings as app_settings
        fast_model = app_settings.effective_fast_model
        resp = await self.llm.generate(
            system="你是旅行规划助手。请从用户查询中提取旅行参数。只返回 JSON。",
            user=prompt,
            model=fast_model,
            max_tokens=256,
            temperature=0.3,
        )
        data = parse_llm_json(resp.content)
        return {
            "destination_city": data.get("destination_city", "杭州"),
            "duration_days": max(1, int(data.get("duration_days", 1))),
            "interests": data.get("interests", ["白蛇传", "古建筑"]),
            "budget_level": data.get("budget_level", "中档"),
            "travel_style": data.get("travel_style", "深度文化"),
            "travel_dates": data.get("travel_dates", ""),
        }

    async def _gather_attractions(self, state: ItineraryState) -> dict:
        """收集目标城市的景点信息——含真实门票数据。"""
        city = state.get("destination_city", "杭州")
        interests = state.get("interests", [])

        # 从已知数据中获取景点
        attractions = CITY_ATTRACTIONS.get(city, DEFAULT_ATTRACTIONS)

        # 尝试从 Node API 获取补充数据
        location_data = {}
        if self.node_client:
            locations = await self.node_client.get_locations()
            for attr in attractions:
                for loc in locations:
                    if loc.get("name") == attr["name"]:
                        attr["stats"] = loc
            location_data = {"total_locations": len(locations)}

        # ★ 获取真实门票价格（实时 web_search）
        for attr in attractions:
            name = attr["name"]
            ticket_info = None
            if self.mcp:
                ticket_info = await self.mcp.call_tool("get_ticket_price", {
                    "attraction_name": name, "city": city,
                })
            if ticket_info and not ticket_info.get("error") and ticket_info.get("found"):
                attr["ticket_price"] = ticket_info.get("price", "")
                attr["address"] = ticket_info.get("address", "")
                attr["tel"] = ticket_info.get("tel", "")
                attr["open_time"] = ticket_info.get("open_time", "")
                attr["ticket_source"] = ticket_info.get("source_type", "")
            else:
                attr["ticket_price"] = "请电话咨询"
                attr["address"] = ""
                attr["tel"] = ""
                attr["open_time"] = ""

        # 按兴趣过滤和排序
        scored = []
        for attr in attractions:
            score = 0
            name = attr["name"]
            cat = attr.get("category", "")
            # 白蛇传相关加分
            if any(kw in name for kw in ["雷峰塔", "断桥", "金山寺", "灵隐寺", "青城山", "峨眉山"]):
                score += 3
            # 兴趣匹配
            for interest in interests:
                if interest in cat or interest in name:
                    score += 2
            scored.append({**attr, "relevance_score": score})

        scored.sort(key=lambda x: x["relevance_score"], reverse=True)

        return {
            "attractions": scored[:8],  # 最多 8 个景点
            "location_data": location_data,
        }

    async def _retrieve_literature(self, state: ItineraryState) -> dict:
        """为景点检索白蛇传相关文献。"""
        attractions = state.get("attractions", [])
        all_docs = []

        for attr in attractions[:5]:  # 为前 5 个景点检索
            docs = await self.retriever.retrieve(attr["name"], top_k=3)
            for d in docs:
                all_docs.append({
                    **d,
                    "attraction": attr["name"],
                })

        return {"retrieved_docs": all_docs}

    async def _generate_weather(self, state: ItineraryState) -> dict:
        """生成目的地天气与最佳游览时间建议。"""
        city = state.get("destination_city", "杭州")
        travel_dates = state.get("travel_dates", "")

        prompt = f"""请为「{city}」生成一份实用的天气与最佳游览时间建议。

{f'用户计划出行时间：{travel_dates}' if travel_dates else '用户未指定出行时间。'}

请包含以下内容：
1. **四季气候概况**：简要说明该城市四季的气候特征（温度、降水）
2. **最佳游览季节**：推荐最适宜旅游的月份及理由
3. **当前/近期天气趋势**：基于常识给出近期天气预期
4. **出行装备建议**：根据季节推荐衣物和必需品
5. **天气应急预案**：雨天等不利天气下的备选方案

请用中文回答，内容实用具体，约 300-400 字。"""

        resp = await self.llm.generate(
            system="你是专业的旅行气象顾问。每条信息必须标注来源：真实气象数据标注'[来源：气象常识/季节规律]'，基于当前日期推算的信息标注'[来源：系统实时时间]'，不确定的内容标注'[来源：AI推断 — 建议出行前核实]'。",
            user=prompt,
            max_tokens=800,
            temperature=0.5,
        )
        return {"weather_advice": resp.content}

    async def _generate_hotels(self, state: ItineraryState) -> dict:
        """生成酒店/住宿建议——含网络搜索真实酒店数据。"""
        city = state.get("destination_city", "杭州")
        budget = state.get("budget_level", "中档")
        attractions = state.get("attractions", [])
        style = state.get("travel_style", "深度文化")

        # 构建景点区域信息
        attr_names = [a["name"] for a in attractions[:5]]
        attr_areas = "、".join(attr_names)

        # ★ 网络搜索：获取真实酒店数据
        real_hotel_data = ""
        if self.mcp:
            hotel_search = await self.mcp.call_tool("web_search", {
                "query": f"{city} {attr_names[0] if attr_names else ''}附近酒店 {budget} 地址 价格",
                "max_results": 5,
            })
            if hotel_search and not hotel_search.get("error") and hotel_search.get("results"):
                real_hotel_data = "\n\n## ★ 网络搜索到的真实酒店数据（请优先使用以下真实信息）\n"
                for r in hotel_search.get("results", [])[:5]:
                    real_hotel_data += f"- {r.get('title', '')}: {r.get('body', '')[:200]}\n"

        prompt = f"""请为计划前往「{city}」游览白蛇传相关景点的游客提供住宿建议。

**行程参数**：
- 预算水平：{budget}
- 旅行风格：{style}
- 主要景点：{attr_areas}

请包含以下内容：
1. **推荐住宿区域**（2-3个）：说明每个区域的特点、优缺点、适合人群
2. **各档次酒店参考**（★ 必须包含具体名称和地址，方便用户在地图搜索）：
   - 经济型（200-400元/晚）：举2-3家具体酒店，每家的名称、地址（XX路XX号）、参考价格
   - 中档（400-800元/晚）：举2-3家具体酒店，每家的名称、地址、参考价格
   - 高端（800元+/晚）：举1-2家具体酒店，每家的名称、地址、参考价格
3. **住宿选择技巧**：针对白蛇传主题旅行者的特别建议
4. **预订注意事项**：旺季提前预订时间、取消政策等

请用中文回答，根据预算水平重点展开对应档次的建议，约 400-500 字。
★ 重要：酒店必须有具体名称+路牌号地址，不能只写'市中心'等模糊描述。{real_hotel_data}"""

        resp = await self.llm.generate(
            system="你是专业的旅行住宿顾问。★ 来源标注规则：来自搜索结果的真实酒店标注'[来源：DuckDuckGo实时搜索]'，基于你知识库的补充标注'[来源：AI知识库推断 — 建议在预订平台核实]'，价格标注'[来源：搜索参考价/请以实际预订为准]'。每条推荐必须标明来源。",
            user=prompt,
            max_tokens=1000,
            temperature=0.5,
        )
        return {"hotel_advice": resp.content}

    async def _generate_transportation(self, state: ItineraryState) -> dict:
        """生成交通规划。"""
        city = state.get("destination_city", "杭州")
        attractions = state.get("attractions", [])
        duration = state.get("duration_days", 1)

        attr_names = [a["name"] for a in attractions[:5]]
        attr_list = "\n".join(f"- {name}" for name in attr_names)

        # ★ 网络搜索：获取真实交通枢纽数据
        real_transport_data = ""
        if self.mcp:
            transport_search = await self.mcp.call_tool("web_search", {
                "query": f"{city} 火车站 机场 具体地址 到景区交通",
                "max_results": 5,
            })
            if transport_search and not transport_search.get("error") and transport_search.get("results"):
                real_transport_data = "\n\n## ★ 网络搜索到的真实交通枢纽数据（请优先使用以下真实信息）\n"
                for r in transport_search.get("results", [])[:5]:
                    real_transport_data += f"- {r.get('title', '')}: {r.get('body', '')[:200]}\n"

        prompt = f"""请为计划前往「{city}」的游客提供交通出行规划。

**行程参数**：
- 目的地：{city}
- 游览天数：{duration} 天
- 计划景点：
{attr_list}

请包含以下内容：
1. **到达交通**（如适用）：
   - 飞机：最近机场名称及具体地址（XX路XX号）、到市区交通方式及时长
   - 高铁/火车：主要车站全称及具体地址、车站到景区交通
   - 自驾：主要高速出口名称及停车场具体位置
2. **市内交通**：
   - 地铁/公交：具体线路名称+站点名（如'地铁1号线龙翔桥站A口'）
   - 出租车/网约车：大致费用参考
   - 共享单车/步行：适合的区域
3. **景点间接驳**：各景点之间的具体交通方式（包括上车点、下车点名称）
4. **交通费用预估**：{duration}天行程的大致交通预算
5. **实用 Tips**：交通 App、支付方式、高峰避让等

请用中文回答，内容实用具体，约 400-500 字。
★ 重要：所有交通枢纽、站点必须给出全称+地址，方便用户在地图上导航。{real_transport_data}"""

        resp = await self.llm.generate(
            system="你是专业的旅行交通顾问。★ 来源标注规则：来自搜索结果的交通枢纽标注'[来源：DuckDuckGo实时搜索]'，基于你知识库的补充标注'[来源：AI知识库推断 — 建议在地图APP核实]'，票价/费用标注'[来源：AI估算/请以实际为准]'。每条信息必须标明来源。",
            user=prompt,
            max_tokens=1000,
            temperature=0.5,
        )
        return {"transportation_plan": resp.content}

    async def _optimize_route(self, state: ItineraryState) -> dict:
        """智能优化游览顺序。"""
        city = state.get("destination_city", "杭州")
        attractions = state.get("attractions", [])
        duration = state.get("duration_days", 1)
        style = state.get("travel_style", "深度文化")

        # 构建景点详情
        attr_details = []
        for i, attr in enumerate(attractions[:6]):
            visit_time = attr.get("visit_time", "1-2小时")
            best_time = attr.get("best_time", "全天")
            category = attr.get("category", "")
            attr_details.append(
                f"{i+1}. **{attr['name']}** ({category}) — 建议游览{visit_time}，最佳时段：{best_time}"
            )

        attr_text = "\n".join(attr_details)

        prompt = f"""请为以下白蛇传主题旅行设计最优游览路线。

**行程参数**：
- 目的地：{city}
- 天数：{duration} 天
- 旅行风格：{style}
- 候选景点：
{attr_text}

请以 JSON 格式返回优化后的行程安排（每天的具体计划）：

{{
  "daily_plans": [
    {{
      "day": 1,
      "theme": "当日主题",
      "attractions": [
        {{
          "name": "景点名",
          "time_slot": "上午/下午/傍晚",
          "duration": "建议停留时间",
          "reason": "为什么安排在这个时段",
          "tips": "游览小贴士"
        }}
      ],
      "meals": "用餐建议",
      "evening": "晚间活动建议"
    }}
  ],
  "optimization_notes": "总体优化说明（100字以内）"
}}

优化原则：
1. 地理相邻优先（减少交通时间）
2. 最佳时段匹配（雷峰夕照傍晚去，寺庙上午去）
3. 体力节奏（上午重体力景点，下午轻松）
4. 白蛇传故事线串联（按故事情节顺序排列更佳）
5. 预留用餐和休息时间"""

        resp = await self.llm.generate(
            system="你是专业的旅行路线规划师，擅长优化行程顺序。只返回 JSON。",
            user=prompt,
            max_tokens=1500,
            temperature=0.5,
        )
        data = parse_llm_json(resp.content)
        return {
            "optimized_route": data.get("daily_plans", []),
        }

    async def _compose_itinerary(self, state: ItineraryState) -> dict:
        """编织完整旅行攻略（非流式版本 — 供 run() 调用）。"""
        city = state.get("destination_city", "杭州")
        days = state.get("duration_days", 1)
        budget = state.get("budget_level", "中档")
        interests = state.get("interests", [])
        weather = state.get("weather_advice", "")
        hotels = state.get("hotel_advice", "")
        transport = state.get("transportation_plan", "")
        route = state.get("optimized_route", [])
        docs = state.get("retrieved_docs", [])
        attractions = state.get("attractions", [])

        # 构建路线文本
        route_text = ""
        for day_plan in route:
            theme = day_plan.get("theme", "")
            route_text += f"\n### 📅 第{day_plan.get('day', '?')}天：{theme}\n\n"
            route_text += "| 时段 | 景点 | 时长 | 说明 |\n|------|------|------|------|\n"
            for attr in day_plan.get("attractions", []):
                route_text += f"| {attr.get('time_slot', '')} | **{attr.get('name', '')}** | {attr.get('duration', '')} | {attr.get('reason', '')} |\n"
            route_text += f"\n🍽️ **用餐**：{day_plan.get('meals', '当地特色餐厅')}\n"
            route_text += f"\n🌙 **晚间**：{day_plan.get('evening', '自由活动')}\n\n---\n"

        lit_context = ""
        if docs:
            lit_context = "\n\n## 📖 相关文献背景\n\n"
            for d in docs[:5]:
                meta = d.get("metadata", {})
                lit_context += f"- **{meta.get('chapter_title', '未知')}** — {d.get('content', '')[:150]}...\n"

        prompt = f"""请为「{city}」白蛇传主题旅行生成一份完整、专业的旅行攻略。

## 行程参数
- 目的地：{city}
- 天数：{days} 天
- 预算：{budget}
- 风格：{state.get('travel_style', '深度文化')}
- 兴趣：{'/'.join(interests) if interests else '白蛇传文化'}

## 优化路线
{route_text}

{lit_context}

## 天气建议
{weather}

## 住宿建议
{hotels}

## 交通建议
{transport}

请整合以上所有信息，生成一份完整的旅行攻略。格式要求：

# 🗺️ {city}白蛇传主题{days}日深度游攻略

## 📋 行程概览
（简要概述整个行程的亮点和主题）

## 🌤️ 天气与出行准备
（根据具体出行日期给出针对性天气建议 + 装备清单）

## 🗺️ 每日详细行程
（★ 每个景点必须给出：具体地址/位置描述，最近的地铁站/公交站名称，门票价格，建议游览时长）

## 🏨 住宿推荐
（★ 每处住宿必须给出：酒店全称 + 具体地址（XX路XX号）+ 参考价格）

## 🚗 交通指南
（★ 所有交通枢纽必须给出全称和地址：机场全称+地址、火车站全称+地址、地铁线路号+站名）

## 💰 预算参考
（按经济型/中档/高端给出大致总预算）

## 📌 实用贴士
（5-8条针对白蛇传主题旅行的特别建议）

## 📖 白蛇传文化背景
（简要介绍各景点与白蛇传的联系）

★ 核心要求：文中所有地点（景点、酒店、车站、餐厅等）必须给出可在百度地图/高德地图上直接搜索到的具体名称和地址。不要使用"市中心""景区附近"等模糊描述。"""

        system = (
            "你是专业的文化旅游攻略撰写专家。★ 必须遵守来源标注规则：\n"
            "- 门票价格 → 标注'[来源：DuckDuckGo实时搜索 / URL]'\n"
            "- 酒店名称地址 → 标注'[来源：实时搜索]'或'[来源：AI推荐 — 请在地图核实]'\n"
            "- 交通枢纽 → 标注'[来源：实时搜索]'或'[来源：AI知识 — 请在地图核实]'\n"
            "- 天气建议 → 标注'[来源：季节规律/系统时间]'\n"
            "- 文献背景 → 标注'[来源：《白蛇传文献选集》]'\n"
            "- 无法核实的信息 → 标注'[来源：AI推断 — 建议出行前确认]'\n"
            "所有推荐地点必须包含具体名称和路牌号地址。使用 Markdown 格式。"
        )
        sc = state.get("system_context", "")
        if sc:
            system += f"\n\n{sc}"
        resp = await self.llm.generate_with_fallback(system, prompt, max_tokens=3072, temperature=0.7)

        sources = []
        for attr in attractions[:5]:
            sources.append({
                "chapter_number": attr.get("category", ""),
                "chapter_title": attr.get("name", ""),
                "excerpt": f"建议游览{attr.get('visit_time', '1-2小时')}，最佳时段：{attr.get('best_time', '全天')}",
                "relevance_score": round(attr.get("relevance_score", 0.8), 2),
            })

        return {
            "itinerary": resp.content,
            "sources": sources,
            "total_tokens_used": resp.tokens,
            "degraded": resp.degraded,
        }

    # ===== 公共接口 =====

    async def run(self, query: str, thread_id: str = "default", system_context: str = "") -> dict:
        """非流式执行。"""
        initial: ItineraryState = {
            "query": query,
            "thread_id": thread_id,
            "system_context": system_context,
            "destination_city": "",
            "travel_dates": "",
            "duration_days": 1,
            "interests": [],
            "budget_level": "中档",
            "travel_style": "深度文化",
            "attractions": [],
            "location_data": {},
            "retrieved_docs": [],
            "weather_advice": "",
            "hotel_advice": "",
            "transportation_plan": "",
            "optimized_route": [],
            "itinerary": "",
            "sources": [],
            "total_tokens_used": 0,
            "degraded": False,
        }
        config = {"configurable": {"thread_id": thread_id}} if self._checkpointer and thread_id else None
        result = await self._graph.ainvoke(initial, config) if config else await self._graph.ainvoke(initial)
        return result

    async def run_stream(self, query: str, thread_id: str = "stream", system_context: str = "") -> AsyncGenerator[dict, None]:
        """流式执行完整旅行规划 — 含详细实时进度。"""
        initial: ItineraryState = {
            "query": query,
            "thread_id": thread_id,
            "system_context": system_context,
            "destination_city": "",
            "travel_dates": "",
            "duration_days": 1,
            "interests": [],
            "budget_level": "中档",
            "travel_style": "深度文化",
            "attractions": [],
            "location_data": {},
            "retrieved_docs": [],
            "weather_advice": "",
            "hotel_advice": "",
            "transportation_plan": "",
            "optimized_route": [],
            "itinerary": "",
            "sources": [],
            "total_tokens_used": 0,
            "degraded": False,
        }

        # === Step 1: 解析旅行需求 ===
        yield {"type": "status", "data": "🗺️ Step 1/7: 解析旅行需求..."}
        updates = await self._parse_travel_query(initial)
        initial.update(updates)
        city = initial.get("destination_city", "杭州")
        days = initial.get("duration_days", 1)
        interests = initial.get("interests", [])
        budget = initial.get("budget_level", "中档")
        yield {"type": "status", "data": f"✅ Step 1/7: 目的地={city}, {days}天, 兴趣={'/'.join(interests[:3])}, {budget}"}

        # === Step 2: 收集景点 ===
        yield {"type": "status", "data": "📍 Step 2/7: 匹配白蛇传相关景点..."}
        updates = await self._gather_attractions(initial)
        initial.update(updates)
        attractions = initial.get("attractions", [])
        attr_names = [a["name"] for a in attractions[:5]]
        yield {"type": "status", "data": f"✅ Step 2/7: 匹配到 {len(attractions)} 个景点 ({' → '.join(attr_names)})"}

        # === Step 3: 检索文献 ===
        yield {"type": "status", "data": "📚 Step 3/7: 为景点检索白蛇传文献..."}
        updates = await self._retrieve_literature(initial)
        initial.update(updates)
        doc_count = len(initial.get("retrieved_docs", []))
        yield {"type": "status", "data": f"✅ Step 3/7: 检索到 {doc_count} 段相关文献"}

        # === Step 4: 天气建议 ===
        yield {"type": "status", "data": f"🌤️ Step 4/7: 生成 {city} 天气与季节建议..."}
        updates = await self._generate_weather(initial)
        initial.update(updates)
        yield {"type": "status", "data": "✅ Step 4/7: 天气建议已生成"}

        # === Step 5: 酒店建议 ===
        yield {"type": "status", "data": f"🏨 Step 5/7: 生成 {budget} 住宿建议..."}
        updates = await self._generate_hotels(initial)
        initial.update(updates)
        yield {"type": "status", "data": "✅ Step 5/7: 酒店建议已生成"}

        # === Step 6: 交通规划 ===
        yield {"type": "status", "data": "🚗 Step 6/7: 规划交通出行方案..."}
        updates = await self._generate_transportation(initial)
        initial.update(updates)
        yield {"type": "status", "data": "✅ Step 6/7: 交通方案已生成"}

        # === Step 7: 路线优化 + 完整攻略流式生成 ===
        yield {"type": "status", "data": "🔀 Step 7/7: 优化游览顺序并生成完整攻略..."}
        updates = await self._optimize_route(initial)
        initial.update(updates)
        route = initial.get("optimized_route", [])
        day_count = len(route) if route else days
        yield {"type": "status", "data": f"✅ Step 7/7: 路线优化完成 ({day_count}天行程) → 开始生成..."}

        # ---- 流式生成完整攻略 ----
        weather = initial.get("weather_advice", "")
        hotels = initial.get("hotel_advice", "")
        transport = initial.get("transportation_plan", "")
        docs = initial.get("retrieved_docs", [])

        # 构建路线文本
        route_text = ""
        for day_plan in route:
            theme = day_plan.get("theme", "")
            route_text += f"\n### 📅 第{day_plan.get('day', '?')}天：{theme}\n\n"
            route_text += "| 时段 | 景点 | 时长 | 说明 |\n"
            route_text += "|------|------|------|------|\n"
            for attr in day_plan.get("attractions", []):
                route_text += f"| {attr.get('time_slot', '')} | **{attr.get('name', '')}** | {attr.get('duration', '')} | {attr.get('reason', '')} |\n"
            route_text += f"\n🍽️ **用餐**：{day_plan.get('meals', '当地特色餐厅')}\n"
            route_text += f"\n🌙 **晚间**：{day_plan.get('evening', '自由活动')}\n\n---\n"

        # 构建文献背景
        lit_context = ""
        if docs:
            lit_context = "\n\n## 📖 相关文献背景\n\n"
            for d in docs[:5]:
                meta = d.get("metadata", {})
                lit_context += f"- **{meta.get('chapter_title', '未知')}** — {d.get('content', '')[:150]}...\n"

        prompt = f"""请为「{city}」白蛇传主题旅行生成一份完整、专业的旅行攻略。

## 行程参数
- 目的地：{city}
- 天数：{days} 天
- 预算：{budget}
- 风格：{initial.get('travel_style', '深度文化')}
- 兴趣：{'/'.join(interests) if interests else '白蛇传文化'}

## 优化路线
{route_text}

{lit_context}

## 天气建议
{weather}

## 住宿建议
{hotels}

## 交通建议
{transport}

请整合以上所有信息，生成一份完整的旅行攻略。格式要求：

# 🗺️ {city}白蛇传主题{'-' if days else ''}{days}日深度游攻略

## 📋 行程概览
（简要概述整个行程的亮点和主题）

## 🌤️ 天气与出行准备
（根据具体出行日期给出针对性天气建议 + 装备清单）

## 🗺️ 每日详细行程
（★ 每个景点必须给出：具体地址/位置描述，最近的地铁站/公交站名称，门票价格，建议游览时长）

## 🏨 住宿推荐
（★ 每处住宿必须给出：酒店全称 + 具体地址（XX路XX号）+ 参考价格）

## 🚗 交通指南
（★ 所有交通枢纽必须给出全称和地址：机场全称+地址、火车站全称+地址、地铁线路号+站名）

## 💰 预算参考
（按经济型/中档/高端给出大致总预算）

## 📌 实用贴士
（5-8条针对白蛇传主题旅行的特别建议）

## 📖 白蛇传文化背景
（简要介绍各景点与白蛇传的联系）

★ 核心要求：文中所有地点（景点、酒店、车站、餐厅等）必须给出可在百度地图/高德地图上直接搜索到的具体名称和地址。不要使用"市中心""景区附近"等模糊描述。"""

        system = (
            "你是专业的文化旅游攻略撰写专家。★ 必须遵守来源标注规则：\n"
            "- 门票价格 → 标注'[来源：DuckDuckGo实时搜索 / URL]'\n"
            "- 酒店名称地址 → 标注'[来源：实时搜索]'或'[来源：AI推荐 — 请在地图核实]'\n"
            "- 交通枢纽 → 标注'[来源：实时搜索]'或'[来源：AI知识 — 请在地图核实]'\n"
            "- 天气 → 标注'[来源：季节规律/系统时间]'\n"
            "- 文献 → 标注'[来源：《白蛇传文献选集》]'\n"
            "- 无法核实 → 标注'[来源：AI推断 — 建议出行前确认]'\n"
            "所有地点必须有具体地址，方便在地图搜索。使用 Markdown 格式。"
        )
        sc = initial.get("system_context", "")
        if sc:
            system += f"\n\n{sc}"
        chunk_count = 0
        async for text in await self.llm.generate_stream(system, prompt, max_tokens=3072, temperature=0.7):
            chunk_count += len(text)
            yield {"type": "chunk", "data": text}
        yield {"type": "status", "data": f"🎉 攻略生成完成 ({chunk_count} 字)"}

        # 整理引用来源（兼容前端 SourceCitation 接口）
        sources = []
        attractions = initial.get("attractions", [])
        for attr in attractions[:5]:
            sources.append({
                "chapter_number": attr.get("category", ""),
                "chapter_title": attr.get("name", ""),
                "excerpt": f"建议游览{attr.get('visit_time', '1-2小时')}，最佳时段：{attr.get('best_time', '全天')}",
                "relevance_score": round(attr.get("relevance_score", 0.8), 2),
            })
        for src in sources:
            yield {"type": "source", "data": src}

        yield {"type": "done", "data": {
            "workflow": "itinerary",
            "city": city,
            "days": days,
            "attractions_count": len(attractions),
            "sources_count": len(sources),
            "features": ["weather", "hotels", "transportation", "route_optimization"],
        }}
