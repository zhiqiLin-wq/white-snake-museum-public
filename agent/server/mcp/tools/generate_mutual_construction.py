"""v10 MCP 工具 -- 景观-文本互构分析: 分析指定地点的景观变迁与文本叙事之间的双向影响。

设计文档: B.4 节 -- Agent 工具 generate_mutual_construction 按需生成
使用 landscape_info.json 获取物理景观数据，通过 LLM 生成双向影响描述。
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "generate_mutual_construction",
    "description": "分析指定地点的景观-文本互构关系，生成景观到文本和文本到景观的双向影响描述(各80-120字)，以及各朝代的叙事特征描述和关联母题。⚠️ 内部调用 LLM，耗时数秒，适合深入的景观文化分析请求。",
    "inputSchema": {
        "type": "object",
        "properties": {
            "location_name": {
                "type": "string",
                "description": "地点名称，如'雷峰塔'、'西湖'、'金山寺'、'断桥'等",
            },
        },
        "required": ["location_name"],
    },
}


async def handler(
    location_name: str,
    *,
    node_client,
    llm,
    landscape_data: dict | None = None,
) -> dict:
    """分析景观-文本互构关系。

    fast fail: node_client/llm 不可用 -> 直接抛出异常。
    landscape_data 缺失 -> 降级处理（仅生成文本侧分析）。
    """
    if node_client is None:
        return {"error": "Node API 客户端未初始化"}
    if llm is None:
        return {"error": "LLM 未初始化"}

    # Step 1: 获取该地点的所有 literaryRecords
    from .get_location_traceability import handler as trace_handler
    trace_result = await trace_handler(
        location_names=[location_name],
        node_client=node_client,
    )

    loc_data = trace_result.get(location_name, {})
    literary_records = loc_data.get("literaryRecords", [])

    if not literary_records:
        return {
            "locationName": location_name,
            "mutualConstruction": {
                "landscapeToText": f"文献中未找到{location_name}的相关记录。",
                "textToLandscape": f"文献中未找到{location_name}的相关记录。",
            },
            "descriptionStyles": {},
            "relatedMotifs": [],
        }

    # Step 2: 获取该地点的 physicalLandscape 数据
    landscape_info = None
    if landscape_data:
        landscape_info = landscape_data.get(location_name)

    # Step 3-6: 通过 LLM 生成互构分析
    from ...config import settings as app_settings
    model = app_settings.effective_fast_model

    records_summary = _build_records_summary(literary_records)

    landscape_desc = "无物理景观数据"
    if landscape_info:
        physical = landscape_info.get("physicalLandscape", {})
        geo = physical.get("geographicalFeatures", "")
        arch = physical.get("architecturalFeatures", "")
        hist_changes = physical.get("historicalChanges", "")
        landscape_desc = f"地理特征: {geo}\n建筑特征: {arch}\n历史变迁: {hist_changes}"

    system = (
        "你是白蛇传景观-文本互构分析专家。分析指定地点的物理景观与文学文本叙事之间的双向影响关系。\n\n"
        "## 任务\n"
        "1. landscapeToText: 分析该地点的物理景观特征（地理/建筑/历史变迁）如何影响了文本中的叙事方式"
        "（如：景观的视觉特征如何塑造了场景描写？建筑空间的布局如何影响了情节走向？）80-120字。\n"
        "2. textToLandscape: 分析文本叙事如何反过来塑造了该地点的文化景观认知和社会地位"
        "（如：文学作品如何改变人们对这个地方的想象？叙事如何赋予建筑以象征意义？）80-120字。\n"
        "3. descriptionStyles: 为每个朝代生成该地点在文本中的叙事特征描述（每朝代15-25字）。\n"
        "4. relatedMotifs: 识别与该地点关联的叙事母题名称列表（3-8个）。\n\n"
        "## 输出格式\n"
        "只返回 JSON:\n"
        '{"mutualConstruction": {"landscapeToText": "...", "textToLandscape": "..."},'
        '"descriptionStyles": {"唐": "...", "宋": "...", "明": "...", "清": "..."},'
        '"relatedMotifs": ["...", "..."]}\n\n'
        "注意: 若 landscape_info 缺失，landscapeToText 请根据文献记录推断景观特征。"
    )

    user = (
        f"## 地点: {location_name}\n\n"
        f"## 物理景观数据\n{landscape_desc}\n\n"
        f"## 文献记录\n{records_summary}\n\n"
        "请分析该地点的景观-文本互构关系。"
    )

    resp = await llm.generate(system, user, model=model, max_tokens=1024, temperature=0.4, thinking_disabled=True)
    content_raw = resp.content.strip()
    if content_raw.startswith("```"):
        content_raw = content_raw.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        analysis = json.loads(content_raw)
    except json.JSONDecodeError as e:
        logger.error(f"generate_mutual_construction JSON 解析失败: {e}")
        return {
            "locationName": location_name,
            "mutualConstruction": {
                "landscapeToText": f"AI 分析生成失败: {e}",
                "textToLandscape": f"AI 分析生成失败: {e}",
            },
            "descriptionStyles": {},
            "relatedMotifs": [],
            "degraded": True,
        }

    return {
        "locationName": location_name,
        "mutualConstruction": analysis.get("mutualConstruction", {}),
        "descriptionStyles": analysis.get("descriptionStyles", {}),
        "relatedMotifs": analysis.get("relatedMotifs", []),
    }


def _build_records_summary(records: list[dict]) -> str:
    """构建文献记录摘要（供 LLM prompt 使用）。"""
    lines = []
    for r in records:
        dynasty = r.get("dynasty", "?")
        ch_title = r.get("chapterTitle", "")
        excerpt = r.get("excerpt", "")
        lines.append(f"[{dynasty}] {ch_title}: ...{excerpt}...")
    return "\n".join(lines[:30])
