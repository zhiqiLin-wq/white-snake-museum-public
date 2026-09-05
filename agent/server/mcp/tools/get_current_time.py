"""MCP 工具 — 获取当前日期时间和季节信息。"""
from datetime import datetime, timedelta

TOOL_DEF = {
    "name": "get_current_time",
    "description": (
        "获取当前精确日期时间、星期、季节信息。"
        "用于回答'今天/明天/后天/周末/下周'等时间相关问题，"
        "以及判断当前季节以给出合理的旅游建议。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "days_offset": {
                "type": "integer",
                "description": "偏移天数。0=今天, 1=明天, -1=昨天, 7=下周同一天。默认 0。",
                "default": 0,
            },
        },
    },
}

# 农历月份→中国传统季节（大致映射）
_MONTH_SEASON = {
    1: "冬季（深冬）", 2: "冬季（晚冬）", 3: "春季（早春）",
    4: "春季（仲春）", 5: "春季（暮春）", 6: "夏季（初夏）",
    7: "夏季（盛夏）", 8: "夏季（晚夏）", 9: "秋季（初秋）",
    10: "秋季（仲秋）", 11: "秋季（深秋）", 12: "冬季（初冬）",
}


async def handler(days_offset: int = 0) -> dict:
    """返回当前（或偏移后）的日期时间信息。"""
    target = datetime.now() + timedelta(days=days_offset)
    month = target.month
    weekday = target.strftime("%A")
    weekday_cn = {
        "Monday": "周一", "Tuesday": "周二", "Wednesday": "周三",
        "Thursday": "周四", "Friday": "周五", "Saturday": "周六", "Sunday": "周日",
    }.get(weekday, weekday)

    return {
        "source_label": "[来源：系统实时时间]",
        "date_iso": target.strftime("%Y-%m-%d"),
        "date_display": f"{target.year}年{target.month}月{target.day}日",
        "weekday": weekday_cn,
        "time": target.strftime("%H:%M"),
        "season": _MONTH_SEASON.get(month, "未知"),
        "is_weekend": target.weekday() >= 5,
        "year": target.year,
        "month": target.month,
        "day": target.day,
        "days_offset": days_offset,
    }
