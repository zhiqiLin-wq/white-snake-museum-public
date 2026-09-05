"""MCP 工具 — 高德地图真实 POI 搜索（景点/酒店/餐厅/交通枢纽）。
返回真实名称、具体地址、电话、票价/价格区间、评分。
Fast Fail: 仅捕获 httpx.HTTPError + AmapAPIError。
"""
import logging
from typing import Optional

from ...data.amap_client import AmapClient, AmapAPIError, POI_SCENIC, POI_HOTEL, POI_RESTAURANT, POI_TRANSPORT

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "search_real_poi",
    "description": (
        "通过高德地图 API 搜索真实POI（景点、酒店、餐厅、交通枢纽）。"
        "返回具体名称、路牌号地址、电话、票价/价格区间、评分、开放时间。"
        "用于替代 AI 凭空生成的模糊推荐，给用户可在地图上直接搜索的真实地点信息。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "string",
                "description": "搜索关键词。如 '雷峰塔'、'全季酒店'、'杭州菜'、'火车站'",
            },
            "city": {
                "type": "string",
                "description": "城市名。如 '杭州'、'成都'",
            },
            "poi_category": {
                "type": "string",
                "enum": ["scenic", "hotel", "restaurant", "transport"],
                "description": "POI类别: scenic=景点, hotel=酒店, restaurant=餐厅, transport=交通枢纽",
            },
            "limit": {
                "type": "integer",
                "description": "返回数量，默认5",
                "default": 5,
            },
        },
        "required": ["keywords", "city"],
    },
}

_POI_TYPE_MAP = {
    "scenic": POI_SCENIC,
    "hotel": POI_HOTEL,
    "restaurant": POI_RESTAURANT,
    "transport": POI_TRANSPORT,
}


async def handler(
    keywords: str,
    city: str,
    poi_category: str = "scenic",
    limit: int = 5,
    amap_client: Optional[AmapClient] = None,
) -> dict:
    """搜索真实 POI。amap_client 从全局注入。"""
    if not amap_client:
        return {"error": "高德地图 API 未配置（缺少 AMAP_API_KEY）", "results": []}

    poi_type = _POI_TYPE_MAP.get(poi_category, "")
    results = await amap_client.search_poi(
        keywords=keywords,
        city=city,
        poi_type=poi_type,
        offset=limit,
    )

    items = []
    for r in results:
        item = {
            "name": r.name,
            "address": r.address,
            "tel": r.tel,
            "rating": r.rating,
            "location": r.location,
            "open_time": r.open_time,
        }
        if poi_category == "scenic":
            item["ticket_price"] = r.ticket_price or "请电话咨询"
        elif poi_category == "hotel":
            item["star"] = r.hotel_star
            item["price_range"] = r.hotel_price_range or "请电话咨询"
        elif poi_category == "restaurant":
            item["avg_cost"] = r.ticket_price or r.hotel_price_range or "请电话咨询"
        elif poi_category == "transport":
            item["poi_type"] = r.poi_type
        items.append(item)

    return {
        "source": "高德地图 POI 实时数据",
        "category": poi_category,
        "city": city,
        "count": len(items),
        "results": items,
    }
