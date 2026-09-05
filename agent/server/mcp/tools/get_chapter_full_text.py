"""B-011: get_chapter_full_text — 获取指定章的完整文本。"""
import logging
import re

from ._text_utils import infer_dynasty_by_chapter

logger = logging.getLogger(__name__)

CHINESE_NUMBERS: dict[str, int] = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
}

TOOL_DEF = {
    "name": "get_chapter_full_text",
    "description": "获取指定章节的完整文本内容并触发前端自动跳转到该章节。返回该章的全部段落。当用户要求打开/查看/跳转到某章或某段时必须调用此工具。",
    "inputSchema": {
        "type": "object",
        "properties": {
            "chapter_number": {
                "type": "integer",
                "description": "章节编号，如 1 表示第一章（唐代），2 表示第二章（宋代）",
            },
            "paragraph_index": {
                "type": "integer",
                "description": "可选。段落索引（0-based，即用户说的'第N段'对应 N-1）。传入后前端会滚动到指定段落。",
            },
        },
        "required": ["chapter_number"],
    },
}


async def handler(chapter_number: int, paragraph_index: int = None, node_client=None, retriever=None) -> dict:
    """获取指定章的完整文本。

    fast fail: 只捕获已知可恢复的异常（Node API 不可达），其余向上传播。
    """
    if node_client is None:
        return {"error": "Node API 客户端未初始化"}

    ch = await node_client.get_literature_by_chapter(chapter_number)
    if ch is None:
        return {"error": f"未找到第 {chapter_number} 章"}

    title = ch.get("title", "")
    content = ch.get("content", "")
    dynasty = infer_dynasty_by_chapter(chapter_number)

    # 按空行分隔段落，与前端 sourceTreeStore.splitParagraphs 保持一致。
    # 只返回 paragraphs（含段落索引），不再同时返回整章 content——
    # 两者内容完全重复，双份序列化会让结果超出 LLM 截断预算，
    # 导致靠后段落在 ReAct 循环中对 LLM 不可见（标注定位失败）。
    content = re.sub(r'\r\n|\r', '\n', content)
    paragraphs_raw = [p.strip() for p in content.split('\n\n') if p.strip()]
    paragraphs = [{"index": i, "text": p} for i, p in enumerate(paragraphs_raw)]

    result = {
        "chapter_number": ch.get("chapterNumber", chapter_number),
        "title": title,
        "dynasty": dynasty,
        "paragraphs": paragraphs,
        "paragraph_count": len(paragraphs),
    }
    if paragraph_index is not None:
        result["paragraph_index"] = paragraph_index
    return result
