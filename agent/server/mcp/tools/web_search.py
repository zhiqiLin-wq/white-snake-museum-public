"""MCP 工具 — DuckDuckGo 实时网络搜索（免费、无限额、无需API Key）。
用于获取景点门票、酒店价格、餐厅信息等实时公开数据。
Fast Fail: 仅捕获具体异常 ImportError + duckduckgo_search 的已知错误类型。
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "web_search",
    "description": (
        "实时网络搜索——使用DuckDuckGo免费搜索获取最新公开信息。"
        "用于查询景点门票价、酒店价格、餐厅推荐、交通枢纽地址等实时数据。"
        "搜索结果来自公开网页，AI会从中提取准确信息。"
        "示例查询: '雷峰塔 门票价格 2026'、'杭州西湖区全季酒店 地址 价格'、'杭州东站 具体地址'"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词。越具体越好，如'雷峰塔 门票 多少钱 2026'",
            },
            "max_results": {
                "type": "integer",
                "description": "最大返回结果数，默认5",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


async def handler(query: str, max_results: int = 5) -> dict:
    """执行网络搜索。"""
    from duckduckgo_search import DDGS
    from duckduckgo_search.exceptions import (
        DuckDuckGoSearchException,
        RatelimitException,
        TimeoutException,
    )

    results = []
    search_errors = []
    sources = []

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "href": r.get("href", ""),
                })
    except RatelimitException as e:
        search_errors.append(f"搜索频率限制: {e}（稍后重试）")
    except TimeoutException as e:
        search_errors.append(f"搜索超时: {e}")
    except DuckDuckGoSearchException as e:
        search_errors.append(f"搜索失败: {e}")

    # 提取引用来源
    for r in results:
        href = r.get("href", "")
        if href and href not in sources:
            sources.append(href)

    return {
        "query": query,
        "source_label": "[来源：DuckDuckGo实时搜索]",
        "results_count": len(results),
        "results": results,
        "source_urls": sources[:max_results],
        "errors": search_errors if search_errors else [],
        "usage_note": "请在使用这些信息时标注'[来源：DuckDuckGo实时搜索 / 网页URL]'，尤其是价格、地址等可能随时间变化的数据。",
    }
