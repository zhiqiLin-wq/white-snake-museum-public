"""B-010: search_paragraphs — 按条件过滤检索段落。

支持按朝代/地点/人物过滤的语义搜索。
"""
import logging

from ...rag.retriever import chapter_number_to_int

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "search_paragraphs",
    "description": (
        "在白蛇传文献（含 7 章正文与研究文献）中执行语义搜索，返回按相关度排序的"
        "匹配段落，支持朝代/地点/人物过滤。适合意思相近、表述不精确的模糊查询"
        "（如搜索'许仙被法海关押'能命中'金山寺蟒蛇索夫'）。"
        "注意：结果可能含研究文献条目（isResearch=true，前端无详情页不可跳转）；"
        "逐字精确匹配某关键词请用 search_literature。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词或自然语言查询",
            },
            "dynasty_filter": {
                "type": "string",
                "description": "按朝代过滤：唐 | 宋 | 明 | 清，不填则搜索全部朝代",
            },
            "location_filter": {
                "type": "string",
                "description": "按地点名过滤，如 '雷峰塔'、'断桥'",
            },
            "character_filter": {
                "type": "string",
                "description": "按人物名过滤，如 '白素贞'、'许仙'、'法海'",
            },
            "top_k": {
                "type": "integer",
                "description": "返回结果数量，默认 10",
                "default": 10,
            },
        },
        "required": ["query"],
    },
}


async def handler(query: str, dynasty_filter: str = "",
                  location_filter: str = "", character_filter: str = "",
                  top_k: int = 10, retriever=None) -> dict:
    """按条件过滤检索段落。

    dynasty_filter: metadata 级硬过滤（chunk metadata 含 dynasty 字段）。
    location_filter / character_filter: content 层后置过滤——chunker 未提取
    地点/人物实体到 metadata，故无法走 metadata 过滤，改为对召回结果做子串匹配。
    """
    if retriever is None:
        return {"error": "检索器未初始化"}

    # dynasty 走 metadata 硬过滤
    filters = {}
    if dynasty_filter:
        filters["dynasty"] = dynasty_filter

    # location/character 走 content 后置过滤：扩大召回以补偿过滤导致的结果减少
    has_post_filter = bool(location_filter or character_filter)
    from ...rag.config import rag_config
    fetch_k = min(top_k, rag_config.search_top_k)
    if has_post_filter:
        fetch_k = min(fetch_k * 3, rag_config.search_top_k)

    docs = await retriever.search(query, top_k=fetch_k,
                                  filters=filters if filters else None)

    # content 层后置过滤：保留同时含 location_filter 与 character_filter 的结果
    if has_post_filter:
        def _content_match(doc):
            content = doc.get("content", "") or ""
            if location_filter and location_filter not in content:
                return False
            if character_filter and character_filter not in content:
                return False
            return True
        docs = [d for d in docs if _content_match(d)][:top_k]

    # 输出结构与前端 SearchResultsView 期望的 Node 版 search_literature 格式对齐
    # (query/totalMatches/results[].chapterNumber 等 camelCase 字段)
    results = []
    for rank, d in enumerate(docs, start=1):
        meta = d.get("metadata", {})
        dynasty = meta.get("dynasty", "")
        # 章节号转 int（中文数字 '三'→3）；研究文献（'研究'/空）归为 ''，
        # 避免 Number('三') = NaN 导致前端跳转打开空白章节
        results.append({
            "rank": rank,
            "chapterNumber": chapter_number_to_int(meta.get("chapter_number", "")),
            "chapterTitle": meta.get("chapter_title", ""),
            "paragraphIndex": meta.get("paragraph_index", 0),
            "excerpt": d.get("content", "")[:300],
            "relevanceScore": round(d.get("relevance_score", d.get("score", 0)), 3),
            # 研究文献（无朝代归属）在前端没有详情页，标记后前端隐藏"跳转到原文"按钮
            "isResearch": not dynasty,
            "metadata": {
                "dynasty": dynasty,
            },
        })

    return {
        "query": query,
        "totalMatches": len(results),
        "results": results,
        "sortOptions": ["相关度"],
        "filtersApplied": {
            **filters,
            **({"location": location_filter} if location_filter else {}),
            **({"character": character_filter} if character_filter else {}),
        },
    }
