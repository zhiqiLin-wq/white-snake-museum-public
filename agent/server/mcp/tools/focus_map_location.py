"""MCP 工具 -- 聚焦地图到指定地点并触发飞行动画 + 溯源卡片。

通过 SSE map_highlight 事件驱动前端 mapStore.addHighlight，
触发 MapView.processFlyToQueue() 执行 flyTo 并打开溯源卡片。
"""
import logging

from ._text_utils import GEO_LOCATION_COORDS

logger = logging.getLogger(__name__)

# 12 个可飞行的地理地标坐标（单一真相源在 _text_utils，与前端 coordinates.ts 一致）
LOCATION_COORDS: dict[str, tuple[float, float]] = GEO_LOCATION_COORDS

TOOL_DEF = {
    "name": "focus_map_location",
    "description": (
        "聚焦地图到指定地点，触发地图飞行动画并自动打开该地点的溯源卡片。"
        "当用户要求定位到/飞到/聚焦/跳转到/导航到某个地点时调用此工具。"
        "支持 12 个白蛇传核心地点：雷峰塔/西湖/金山寺/断桥/峨眉山/青城山/"
        "承天寺/卧佛寺/望江楼/龙虎山/灵隐寺/西子湖。"
        "例如：「定位到雷峰塔」「飞到断桥」「聚焦西湖」。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "location_name": {
                "type": "string",
                "description": "地点名称，必须是12个白蛇传核心地点之一",
            },
        },
        "required": ["location_name"],
    },
}


def handler(location_name: str) -> dict:
    """查找地点坐标并返回 fly_to 指令。

    fast fail: 未知地点 → 返回 error。
    """
    coords = LOCATION_COORDS.get(location_name)
    if coords is None:
        valid = "、".join(sorted(LOCATION_COORDS.keys()))
        return {"error": f"未找到地点「{location_name}」。白蛇传核心地点包括：{valid}"}

    lat, lng = coords
    return {
        "name": location_name,
        "lat": lat,
        "lng": lng,
        "action": "fly_to",
    }
