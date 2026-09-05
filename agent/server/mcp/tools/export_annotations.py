"""U-16: export_annotations — Agent 导出标注请求的路由工具。

用户通过 Agent Panel 说出导出意图（如「导出第4章标注为 CSV」）时，
Agent 调用此工具将参数结构化为 SSE export_ready 事件。
前端收到事件后使用当前已加载的标注数据在本地生成文件并触发下载。

此工具不直接读取 SQLite — 标注数据在用户会话期间已由前端从 SQLite
(user.db) 加载到 Pinia store。工具仅负责意图解析和参数传递。
如需实现完整的服务端导出（从 SQLite 直接生成文件），请参考设计文档
docs/sqlite-storage-design.md 中的 SQL 查询示例。
"""
import logging

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "export_annotations",
    "description": (
        "当用户要求导出标注数据时调用此工具。"
        "从用户自然语言指令中解析导出参数（章节编号、格式、来源范围）。"
        "支持的导出格式: csv / json / html。"
        "来源范围: user(仅用户标注) / agent(仅Agent标注) / all(全部标注)。"
        "调用后前端会自动生成文件并触发浏览器下载，Agent 只需简短确认即可。"
        "章节编号约定: 用户说的「第4章」对应 chapter_numbers=[4]，"
        "「第1章和第3章」对应 [1,3]，「全部章节」对应所有章编号的列表。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "chapter_numbers": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "要导出的章节编号列表。例如 [4] 表示仅第4章，[1,2,3] 表示前三章。"
                    "如果用户说「全部章节」且无法确定具体数量，传入空数组 [] 或猜测常见范围 [1,2,3,4,5,6,7,8]。"
                ),
            },
            "format": {
                "type": "string",
                "enum": ["csv", "json", "html"],
                "description": "导出文件格式。用户说 CSV/表格 时用 csv，JSON/结构化数据 时用 json，HTML/网页 时用 html。默认为 csv。",
            },
            "source": {
                "type": "string",
                "enum": ["user", "agent", "all"],
                "description": (
                    "标注来源范围。用户说「我的标注」「我标的」→ user；"
                    "「Agent 标注」「AI 标的」→ agent；未指定或说「全部」「所有」→ all。默认为 all。"
                ),
            },
        },
        "required": ["chapter_numbers"],
    },
}


async def handler(chapter_numbers: list, format: str = "csv",
                  source: str = "all", **kwargs) -> dict:
    """接收 Agent 解析的导出参数，返回结构化数据。

    返回的 dict 会被 agent_loop._format_workspace_event() 转为
    build_sse_event(SSEEventType.EXPORT_READY, **result)，
    最终以 SSE export_ready 事件流向前端。
    """
    chapter_count = len(chapter_numbers)
    chapter_desc = (
        f"第{'、'.join(str(n) for n in chapter_numbers)}章"
        if chapter_count <= 5
        else f"{chapter_count}个章节"
    )

    format_labels = {"csv": "CSV", "json": "JSON", "html": "HTML"}
    source_labels = {"user": "仅用户标注", "agent": "仅Agent标注", "all": "全部标注"}

    logger.info(
        f"export_annotations: chapters={chapter_numbers} format={format} source={source}"
    )

    return {
        "chapter_numbers": chapter_numbers,
        "format": format,
        "source": source,
        "message": (
            f"已接收导出请求：{chapter_desc}，"
            f"格式 {format_labels.get(format, format.upper())}，"
            f"范围 {source_labels.get(source, source)}。"
            f"文件正在准备下载..."
        ),
    }
