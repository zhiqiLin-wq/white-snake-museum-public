"""MCP 工具 -- 删除指定章节的标注。

支持按类别删除（如"删除所有地点标注"）或删除全部标注（如"清空本章标注"）。

关键设计:
- 标注主要存储在前端 localStorage，后端仅做路由转发。
- 返回 deleted=true + 空 annotations/marginalia 列表，
  由 supervisor/agent_loop 转为 SSE annotation_data 事件清空前端状态。
- categories 参数用于前端过滤保留不需要删除的类别（仅 user 标注支持）。
"""

import logging

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "delete_annotations",
    "description": (
        "删除指定章节的标注。可按类别删除（如'删除所有地点标注'），"
        "或删除全部标注（如'清空本章标注'）。"
        "返回删除确认信号，触发前端清空对应标注。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "chapter_number": {
                "type": "integer",
                "description": "章节编号",
            },
            "categories": {
                "type": "array",
                "items": {"type": "string"},
                "description": "要删除的标注类别列表。如 ['location'] 删除所有地点标注。不传则删除全部。",
            },
            "delete_all": {
                "type": "boolean",
                "description": "是否删除该章全部标注。与 categories 互斥，delete_all 优先。",
            },
        },
        "required": ["chapter_number"],
    },
}


async def handler(
    chapter_number: int,
    categories: list[str] | None = None,
    delete_all: bool = False,
    node_client=None,
    user_id: str = "",
) -> dict:
    """返回删除信号，由上层（supervisor/agent_loop）转为 SSE 事件。

    同时调用 Node.js /api/annotations DELETE 端点实际删除数据库中的记录。
    """
    if chapter_number <= 0:
        return {"error": f"无效的章节编号: {chapter_number}"}

    if node_client is None:
        return {"error": "Node API 客户端未初始化，无法验证章节是否存在"}

    # Fast fail: 验证章节存在
    ch = await node_client.get_literature_by_chapter(chapter_number)
    if ch is None:
        return {"error": f"未找到第 {chapter_number} 章，无法执行删除操作"}

    # Normalize: delete_all 优先于 categories
    if delete_all:
        categories = None

    # Normalize categories to list or None
    valid_categories = {"person", "location", "event", "term", "motif", "custom"}
    if categories is not None:
        if not isinstance(categories, list):
            return {"error": f"categories 必须是数组，收到: {type(categories).__name__}"}
        filtered = [c for c in categories if c in valid_categories]
        if not filtered:
            return {"error": f"未指定有效类别。支持的类别: {', '.join(sorted(valid_categories))}"}
        categories = filtered

    # 安全检查: 如果既未指定 delete_all 也未指定 categories，拒绝操作。
    # LLM 必须显式传入 delete_all=true 或指定 categories 列表。
    # 防止 LLM 幻觉调用 delete_annotations(chapter_number=N) 意外清空全部标注。
    if not delete_all and categories is None:
        return {
            "error": (
                "必须指定 delete_all=true（删除全部）或 categories 列表（按类别删除）。"
                f"支持的类别: {', '.join(sorted(valid_categories))}"
            )
        }

    logger.info(
        f"delete_annotations: chapter={chapter_number} "
        f"delete_all={delete_all} categories={categories}"
    )

    # 实际调用 Node.js API 删除数据库记录
    # clean_agent=True: 同时删除 agent 来源标注（用户说"删除标注"通常指删 agent 生成的）
    try:
        db_result = await node_client.delete_annotations(
            chapter_number=chapter_number,
            categories=categories,
            delete_all=delete_all,
            clean_agent=True,
            user_id=user_id,
        )
        logger.info(
            f"delete_annotations DB: deleted={db_result.get('deleted', 0)} "
            f"annotations={db_result.get('deletedAnnotations', 0)} "
            f"marginalia={db_result.get('deletedMarginalia', 0)}"
        )
    except Exception as e:
        logger.error(f"delete_annotations Node API 调用失败: {e}")
        return {"error": f"删除标注失败: Node API 不可达 ({str(e)[:200]})"}

    return {
        "chapter_number": chapter_number,
        "deleted": True,
        "delete_all": delete_all or categories is None,
        "categories": categories,
        "annotations": [],
        "marginalia": [],
    }
