"""MCP 工具 — 获取景点门票价格 + 详细信息。
调用高德 POI 详情 API 获取 deep_info 中的票价数据。
"""
import logging
from typing import Optional

from ...data.amap_client import AmapClient, AmapAPIError, POI_SCENIC

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "get_attraction_ticket",
    "description": (
        "获取指定景点的真实门票价格、开放时间、地址、电话。"
        "先搜索景点POI，再查详情获取票价等深层信息。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "attraction_name": {
                "type": "string",
                "description": "景点名称，如 '雷峰塔'、'金山寺'",
            },
            "city": {
                "type": "string",
                "description": "所在城市，如 '杭州'、'镇江'",
            },
        },
        "required": ["attraction_name", "city"],
    },
}


async def handler(
    attraction_name: str,
    city: str,
    amap_client: Optional[AmapClient] = None,
) -> dict:
    """获取景点门票等详细信息。"""
    if not amap_client:
        return {"error": "高德地图 API 未配置", "attraction": attraction_name}

    # Step 1: 搜索景点 POI
    results = await amap_client.search_poi(
        keywords=attraction_name,
        city=city,
        poi_type=POI_SCENIC,
        offset=1,
    )

    if not results:
        return {
            "attraction": attraction_name,
            "found": False,
            "message": f"在'{city}'未搜索到景点'{attraction_name}'",
        }

    top = results[0]

    # Step 2: 查详情（有些数据只在 detail 接口返回）
    detail = None
    if top.raw_data.get("id"):
        detail = await amap_client.get_detail(top.raw_data["id"])

    info = {
        "attraction": attraction_name,
        "found": True,
        "name": detail.name if detail else top.name,
        "address": detail.address if detail else top.address,
        "tel": detail.tel if detail else top.tel,
        "ticket_price": (
            detail.ticket_price or
            top.ticket_price or
            "暂未获取到门票价，请电话咨询"
        ),
        "open_time": (
            detail.open_time or
            top.open_time or
            "暂未获取到，请电话咨询"
        ),
        "rating": detail.rating if detail else top.rating,
        "location": detail.location if detail else top.location,
    }

    if info["tel"]:
        info["ticket_price"] += f"（确认: {info['tel']}）"

    return info
