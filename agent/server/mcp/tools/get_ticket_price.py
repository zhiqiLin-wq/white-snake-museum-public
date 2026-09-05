"""MCP 工具 — 获取景点门票价格（实时网络搜索）。
每次查询都通过 DuckDuckGo 实时搜索最新公开数据，不使用任何本地缓存。
Fast Fail: 仅捕获已知具体异常。
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "get_ticket_price",
    "description": (
        "获取景点最新门票价格、地址、开放时间——每次实时搜索，不依赖过期缓存。"
        "搜索数据来自公开网页（景区官网、政府旅游网站、OTA平台）。"
        "返回票价、具体地址、电话、开放时间、搜索来源URL。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "attraction_name": {
                "type": "string",
                "description": "景点名称，如 '雷峰塔'、'金山寺'、'灵隐寺'",
            },
            "city": {
                "type": "string",
                "description": "所在城市（辅助搜索用），如 '杭州'",
            },
        },
        "required": ["attraction_name"],
    },
}


async def handler(
    attraction_name: str,
    city: str = "",
    web_search_handler=None,
) -> dict:
    """实时搜索门票价格。"""
    if web_search_handler is None:
        return {
            "attraction": attraction_name,
            "found": False,
            "error": "网络搜索不可用",
        }

    search_query = f"{attraction_name} 门票价格 开放时间 地址"
    if city:
        search_query = f"{city} {search_query}"

    search_result = await web_search_handler(query=search_query, max_results=5)

    results = search_result.get("results", [])
    sources = search_result.get("source_urls", [])

    if not results:
        return {
            "attraction": attraction_name,
            "found": False,
            "message": f"实时搜索未找到'{attraction_name}'的门票信息，建议用户电话咨询景区",
        }

    # 提取最可能的票价和地址信息
    all_text = " ".join(r.get("body", "") + " " + r.get("title", "") for r in results)

    return {
        "attraction": attraction_name,
        "found": True,
        "source_type": "DuckDuckGo 实时搜索",
        "source_label": f"[来源：DuckDuckGo实时搜索 / {len(sources)} 个来源]",
        "search_query": search_query,
        "search_results": results,
        "source_urls": sources,
        "raw_text_snippet": all_text[:2000],
        "message": (
            "以上为本次实时搜索的最新公开数据。"
            "请优先采信景区官网、政府旅游网站（.gov.cn）、知名OTA平台（携程/美团）的信息。"
            "如多个来源价格不一致，标注差异并建议用户出行前电话确认。"
            "★ 每条信息必须标注来源URL。"
        ),
    }
