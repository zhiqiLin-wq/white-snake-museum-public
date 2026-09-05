"""MCP 工具 — 全文搜索文献。

调用 Node 后端 /api/literature/search 进行精确字符串匹配，
返回段落级偏移（paragraphMatchPositions）供前端 TextReaderView 渲染高亮。
"""
import logging

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "search_literature",
    "description": (
        "在《白蛇传》7 章文献正文（唐至现当代选集，不含研究文献）中"
        "逐字精确搜索关键词，返回所有匹配段落及匹配词在段落中的精确起止位置。"
        "当用户要求查找某个词/短语/原句的所有出现、或引用某句原文需要定位出处时调用。"
        "精确字面匹配（未出现即返回 0）；语义相关的模糊搜索请用 search_paragraphs。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词（中文）",
            },
        },
        "required": ["query"],
    },
}


async def handler(query: str, node_client=None) -> dict:
    """调用 Node 后端全文搜索 API，返回 SearchResultsEvent 格式数据。

    fast fail: node_client 未初始化或 HTTP 错误直接抛出，由 registry.call_tool 捕获。
    """
    if node_client is None:
        raise RuntimeError("Node API 客户端未初始化")

    if len(query.strip()) < 2:
        raise ValueError("搜索关键词不能少于 2 个字符")

    return await node_client.search_literature(query)
