"""B-022: infer_phylogeny — 基于母题共享模式推断文本传承关系（供 EvolutionWorkbench E4 调用）。"""
import json
import logging

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "infer_phylogeny",
    "description": "基于母题共享矩阵推断文本传承关系，生成谱系网络（节点 + 边）。⚠️ 该工具内部调用 LLM，耗时 3-5 秒，适合明确的演变分析请求。",
    "inputSchema": {
        "type": "object",
        "properties": {
            "motif_matrix": {"type": "object", "description": "母题 x 朝代矩阵，E2 的输出"},
            "chapters": {"type": "array", "items": {"type": "object"}, "description": "章节信息列表 [{chapter_number, title, dynasty}]"},
        },
        "required": ["motif_matrix"],
    },
}


async def handler(motif_matrix: dict, chapters: list = None, llm=None) -> dict:
    """推断谱系网络。

    fast fail: LLM JSON 异常 -> degraded。其他错误向上传播。
    """
    if llm is None:
        return {"error": "LLM 未初始化"}

    system = (
        "你是文本谱系学专家。基于母题在朝代之间的共享/转移模式，推断文本的传承关系。\n"
        "每个节点需要指定 type 字段为以下三者之一:\n"
        "- extant: 现存文本（有明确文献依据的章节）\n"
        "- missing: 失传环节（推测存在但文本已失传的中间环节）\n"
        "- external_influence: 外部影响（非文本传承，来自社会/文化的外部因素）\n\n"
        "只返回 JSON: {{\"nodes\": [{{\"id\": \"...\", \"label\": \"...\", \"era\": \"...\", "
        "\"type\": \"extant|missing|external_influence\", \"size\": N}}], "
        "\"edges\": [{{\"from\": \"...\", \"to\": \"...\", \"type\": \"...\", \"weight\": N}}]}}"
    )
    user = f"母题矩阵:\n{json.dumps(motif_matrix, ensure_ascii=False, indent=2)}"
    if chapters:
        user += f"\n\n章节信息:\n{json.dumps(chapters, ensure_ascii=False)}"

    from ...config import settings as app_settings
    from ...observability.tracer import llm_call_context
    model = app_settings.effective_model
    with llm_call_context("mcp:infer_phylogeny", "infer_phylogeny",
                          upstream="classify_changes",
                          downstream="correlate_external"):
        # thinking_disabled=True: max_tokens=4096 + 思考可能耗时数分钟必超工作台超时
        resp = await llm.generate(system, user, model=model, max_tokens=4096, temperature=0.5, thinking_disabled=True)
    content_raw = resp.content.strip()
    if content_raw.startswith("```"):
        content_raw = content_raw.split("\n", 1)[1].rsplit("```", 1)[0]

    from ...llm.json_utils import parse_llm_json
    try:
        data = parse_llm_json(content_raw)
    except json.JSONDecodeError as e:
        logger.error(f"infer_phylogeny JSON 解析失败: {e}")
        return {"error": f"AI 返回了无法解析的谱系结果: {e}", "degraded": True}

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    return {
        "nodes": nodes,
        "edges": [{"from": e.get("from", e.get("source", "")),
                   "to": e.get("to", e.get("target", "")),
                   "type": e.get("type", ""), "weight": e.get("weight", 1)}
                  for e in edges],
        "node_count": len(nodes),
        "edge_count": len(edges),
    }
