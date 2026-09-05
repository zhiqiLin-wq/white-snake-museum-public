"""MCP 工具 — 获取地标信息。"""
import logging
import httpx

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "get_location_info",
    "description": (
        "获取地标的预计算文献统计：文献匹配数(total_match)、关键词、记录数、"
        "3 条原文摘录。数据来自研究资料库的记录级匹配计数（非实时全文检索）。"
        "仅支持白蛇传已知地点（如 '金山寺''断桥''雷峰塔'），未知地标返回错误；"
        "需要该地名在全文中的逐处出场定位请用 entity_appearances。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "location_name": {
                "type": "string",
                "description": "地标名称（如 '金山寺'、'断桥'、'雷峰塔' 等）",
            },
        },
        "required": ["location_name"],
    },
}


async def handler(location_name: str, node_client=None) -> dict:
    """获取地标信息。"""
    if node_client is None:
        return {"error": "Node API 客户端未初始化"}

    try:
        locations = await node_client.get_locations()
        for loc in locations:
            if loc.get("name") == location_name:
                return {
                    "name": loc.get("name"),
                    "total_match": loc.get("totalMatch", 0),
                    "total_words": loc.get("totalWords", 0),
                    "keywords": loc.get("keywords", [])[:10],
                    "record_count": len(loc.get("records", [])),
                    "sample_excerpts": [r.get("摘要", "") for r in loc.get("records", [])[:3]],
                }
        return {"error": f"未找到地标: {location_name}"}
    except (RuntimeError, httpx.HTTPError, ConnectionError) as e:
        return {"error": f"获取地标信息失败: {str(e)}"}
