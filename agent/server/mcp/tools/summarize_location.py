"""MCP 工具 — 生成地标叙事。"""
import logging

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "summarize_location",
    "description": "为指定地标生成文学化叙事介绍，结合文献原文摘录和历史文化背景。⚠️ 内部调用 LLM，耗时数秒；只要文献统计数据时用 get_location_info。",
    "inputSchema": {
        "type": "object",
        "properties": {
            "location_name": {
                "type": "string",
                "description": "地标名称（如 '雷峰塔'、'西湖'、'金山寺'）",
            },
            "style": {
                "type": "string",
                "description": "叙事风格：'scholarly'(学术) / 'storytelling'(故事) / 'concise'(简洁)",
                "default": "scholarly",
            },
        },
        "required": ["location_name"],
    },
}


async def handler(location_name: str, style: str = "scholarly", llm=None, retriever=None, node_client=None) -> dict:
    """生成地标叙事。"""
    if llm is None or retriever is None:
        return {"error": "LLM 或检索器未初始化"}

    from ...langgraph.location_story import LocationStoryWorkflow

    wf = LocationStoryWorkflow(llm, retriever, None, node_client)
    result = await wf.run(location_name)

    return {
        "location_name": location_name,
        "style": style,
        "narrative": result.get("narrative", ""),
        "characters": result.get("characters", []),
        "events": result.get("events", []),
        "sources": result.get("sources", []),
    }
