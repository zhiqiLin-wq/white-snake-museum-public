"""MCP 工具 — 跨章节比较。"""
import logging

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "compare_chapters",
    "description": (
        "跨章节主题对比分析：按指定维度（如'角色塑造''场景描写''主题表达'）"
        "对比多个章节的处理差异，返回 LLM 综合对比 + 各章相关段落。"
        "⚠️ 口径说明：基于语义检索召回的相关段落 + LLM 对比工作流，耗时 10 秒以上，"
        "适合明确的多章节(3章以上)横向对比请求。"
        "仅两章的结构化元素对比请用 dynasty_compare_elements（更快）；"
        "两段文本的精确对比请用 compare_passages。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "chapters": {
                "type": "array",
                "items": {"type": "string"},
                "description": "要对比的章节号列表（字符串数字，如 ['1', '3', '5']，对应唐/明/现当代）",
            },
            "aspect": {
                "type": "string",
                "description": "对比的维度（如 '角色塑造'、'场景描写'、'主题表达'）",
            },
        },
        "required": ["chapters", "aspect"],
    },
}


async def handler(chapters: list[str], aspect: str, llm=None, retriever=None) -> dict:
    """执行跨章节对比。"""
    if llm is None or retriever is None:
        return {"error": "LLM 或检索器未初始化"}

    per_chapter_docs = {}
    for ch in chapters:
        docs = await retriever.retrieve(f"{aspect} 第{ch}章")
        per_chapter_docs[ch] = [
            {
                "content": d.get("content", "")[:300],
                "chapter_title": d.get("metadata", {}).get("chapter_title", ""),
            }
            for d in docs
        ]

    # 用 LLM 生成对比摘要
    from ...langgraph.comparison import ComparisonWorkflow
    wf = ComparisonWorkflow(llm, retriever, None)
    result = await wf.run(aspect=aspect, chapters=chapters)

    return {
        "aspect": aspect,
        "chapters": chapters,
        "comparison": result.get("comparison_result", ""),
        "per_chapter_docs": per_chapter_docs,
        "sources": result.get("sources", []),
    }
