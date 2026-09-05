"""MCP 工具 — 控制前端地图朝代时间图层。

通过 SSE map_layer_update 事件驱动前端 mapStore.setTimeLayer / toggleTimeLayer / resetLayers，
切换地图上地点标记的朝代过滤状态。
"""
from typing import Union

TOOL_DEF = {
    "name": "set_map_dynasty_layer",
    "description": (
        "切换地图朝代图层，过滤地图上显示的地点标记。"
        "当用户要求查看特定朝代的地图标记时调用此工具。"
        "例如：「显示清代的地图标记」「只看唐宋的地点」「显示所有朝代」。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "dynasty": {
                "oneOf": [
                    {"type": "string", "enum": ["唐", "宋", "明", "清", "all"]},
                    {"type": "array",
                     "items": {"type": "string", "enum": ["唐", "宋", "明", "清"]}},
                ],
                "description": (
                    "要显示的朝代。单个字符串（如'清'）或字符串数组（如['唐','宋']）。"
                    "'all' 表示全部朝代。"
                ),
            },
            "action": {
                "type": "string",
                "enum": ["set", "toggle", "reset"],
                "description": "set=单选替换选中朝代, toggle=追加/取消切换, reset=恢复全部朝代。默认'set'。",
                "default": "set",
            },
        },
        "required": ["dynasty"],
    },
}

VALID_DYNASTIES = frozenset({"唐", "宋", "明", "清", "all"})
VALID_ACTIONS = frozenset({"set", "toggle", "reset"})


def handler(dynasty: Union[str, list[str]], action: str = "set") -> dict:
    """切换地图时间图层。dynasty 可以是单个朝代字符串或字符串数组。"""
    # 规范化并校验
    if isinstance(dynasty, str):
        dynasties = [dynasty]
    else:
        dynasties = list(dynasty)

    for d in dynasties:
        if d not in VALID_DYNASTIES:
            return {"error": f"Invalid dynasty: {d}. Valid values: {sorted(VALID_DYNASTIES)}"}

    if action not in VALID_ACTIONS:
        return {"error": f"Invalid action: {action}. Valid values: {sorted(VALID_ACTIONS)}"}

    return {
        "type": "map_layer_update",
        "dynasty": dynasty,
        "action": action,
    }
