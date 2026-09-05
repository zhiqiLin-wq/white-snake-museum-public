"""B-012: get_paragraph_context — 获取指定段落的前后文。"""
import logging

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "get_paragraph_context",
    "description": "获取指定段落的前后 N 段上下文，用于理解段落在原文中的语境。",
    "inputSchema": {
        "type": "object",
        "properties": {
            "chapter_number": {
                "type": "integer",
                "description": "章节编号",
            },
            "paragraph_index": {
                "type": "integer",
                "description": "段落索引（从 0 开始）",
            },
            "context_size": {
                "type": "integer",
                "description": "前后各取 N 段，默认 2",
                "default": 2,
            },
        },
        "required": ["chapter_number", "paragraph_index"],
    },
}


async def handler(chapter_number: int, paragraph_index: int,
                  context_size: int = 2, node_client=None) -> dict:
    """获取指定段落的前后文。

    fast fail: 参数越界直接 return error，不捕获异常——让 Node API 错误向上传播。
    """
    if node_client is None:
        return {"error": "Node API 客户端未初始化"}

    ch = await node_client.get_literature_by_chapter(chapter_number)
    if ch is None:
        return {"error": f"未找到第 {chapter_number} 章"}

    from ._text_utils import split_paragraphs

    paragraphs = split_paragraphs(ch.get("content", ""))

    if paragraph_index < 0 or paragraph_index >= len(paragraphs):
        return {"error": f"段落索引 {paragraph_index} 超出范围 (0-{len(paragraphs) - 1})"}

    start = max(0, paragraph_index - context_size)
    end = min(len(paragraphs), paragraph_index + context_size + 1)

    context_before = [
        {"index": i, "text": paragraphs[i]}
        for i in range(start, paragraph_index)
    ]
    context_after = [
        {"index": i, "text": paragraphs[i]}
        for i in range(paragraph_index + 1, end)
    ]

    return {
        "target_paragraph": {
            "chapter_number": chapter_number,
            "paragraph_index": paragraph_index,
            "text": paragraphs[paragraph_index],
        },
        "context_before": context_before,
        "context_after": context_after,
        "total_paragraphs_in_chapter": len(paragraphs),
    }
