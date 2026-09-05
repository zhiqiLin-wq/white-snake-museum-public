"""entity_appearances — 实体出场轨迹定位工具。

回答"XX 在哪些章节哪些段落出场"类问题：全文逐字定位实体的每一处
出现位置（章节/段落坐标 + 上下文摘录），按章节分组返回。
结果经 SSE 映射复用前端搜索结果列表的跳转按钮。
"""
import logging

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "entity_appearances",
    "description": (
        "定位实体在白蛇传全文中的每一处出场位置（章节+段落坐标+上下文摘录），"
        "按章节分组返回。口径：全文逐字匹配（与 count_occurrences 同源），"
        "实体未在语料中出现时返回 0 并提示可能为版本别名。"
        "适用于'XX在哪些章节出场''XX首次出现在哪''XX出场轨迹'类问题；"
        "只要数字分布不要位置时用 count_occurrences 更轻量。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "entity": {"type": "string", "description": "实体名称，如 '白素贞'、'雷峰塔'"},
        },
        "required": ["entity"],
    },
}

# 单章最多返回的出场条数（防止高频实体撑爆 LLM 上下文；总数不受影响）
MAX_PER_CHAPTER = 15


async def handler(entity: str, retriever=None) -> dict:
    """定位实体全部出场位置（字面计数口径）。

    fast fail: retriever 调用错误向上传播。
    0 命中时给出与 count_occurrences 一致的别名提示。
    """
    if retriever is None:
        return {"error": "检索器未初始化"}

    stats = await retriever.find_appearances(entity)
    total = stats["total"]

    # 语料库不可用（索引重建中/集合为空）：明确报错，禁止把"查不了"误报成"未出场"
    if not stats.get("corpus_available", True):
        return {
            "error": (
                "语料索引当前不可用（可能正在后台重建），无法定位出场位置。"
                "请稍后重试本工具；在此之前不要断言该实体'未出场'，"
                "可改用 search_literature（基于 Node 全文检索）临时定位。"
            ),
        }

    if total == 0:
        message = (
            f"「{entity}」在白蛇传语料全文中字面出现 0 次，无出场记录。"
            "该实体可能是其他版本的译名/别名（同一人物在不同版本文本中名称不同），"
            "建议：1) 尝试该实体的其他常见写法或别名重新定位；"
            "2) 向用户如实说明本语料中未收录该名称，不要编造位置。"
        )
        return {
            "title": f"'{entity}' 出场轨迹",
            "entity": entity,
            "data_basis": "literal_match",
            "total_occurrences": 0,
            "chapter_groups": [],
            "message": message,
        }

    # 按章节分组，组内按 (paragraph_index, 原始顺序) 排列。
    # chapter_number 已由 retriever 转为 int；'' 为研究文献（无正文详情页）
    groups: dict = {}
    for app in stats["appearances"]:
        key = app["chapter_number"] if app["chapter_number"] != "" else "research"
        g = groups.setdefault(key, {
            "chapter_number": app["chapter_number"],
            "chapter_title": app["chapter_title"],
            "dynasty": app["dynasty"],
            "count": 0,
            "appearances": [],
        })
        g["count"] += 1
        if len(g["appearances"]) < MAX_PER_CHAPTER:
            g["appearances"].append(app)

    for g in groups.values():
        g["truncated_in_chapter"] = g["count"] > len(g["appearances"])

    chapter_groups = [
        groups[k] for k in sorted(groups.keys(), key=lambda k: (1, 0) if k == "research" else (0, k))
    ]

    return {
        "title": f"'{entity}' 出场轨迹（全文逐字定位）",
        "entity": entity,
        "data_basis": "literal_match",
        "total_occurrences": total,
        "truncated": stats["truncated"],
        "note": (
            f"共 {total} 处字面出现，已按章节分组；"
            + (f"单章最多展示 {MAX_PER_CHAPTER} 条摘录。" if stats["truncated"] or any(
                g["truncated_in_chapter"] for g in chapter_groups) else "")
        ),
        "chapter_groups": chapter_groups,
    }
