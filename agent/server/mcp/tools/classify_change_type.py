"""B-021: classify_change_type — 跨朝代变化类型分类（供 EvolutionWorkbench E3 调用）。

7 种变化类型: emergence, disappearance, substitution, amplification,
diminution, transvaluation, transplantation
"""
import json
import logging

logger = logging.getLogger(__name__)

CHANGE_TYPES = [
    "emergence", "disappearance", "substitution", "amplification",
    "diminution", "transvaluation", "transplantation",
]

TOOL_DEF = {
    "name": "classify_change_type",
    "description": "判断两个朝代之间母题变化的类型，返回 7 种变化类型之一及置信度和解释。⚠️ 该工具内部调用 LLM，耗时 3-5 秒，适合明确的演变分析请求。",
    "inputSchema": {
        "type": "object",
        "properties": {
            "motif_id": {"type": "string", "description": "母题 ID"},
            "dynasty_from": {"type": "string", "description": "起始朝代，如 '明'"},
            "dynasty_to": {"type": "string", "description": "结束朝代，如 '清'"},
            "text_from": {"type": "string", "description": "起始朝代的文本描述"},
            "text_to": {"type": "string", "description": "结束朝代的文本描述"},
        },
        "required": ["motif_id", "dynasty_from", "dynasty_to", "text_from", "text_to"],
    },
}


async def handler(motif_id: str, dynasty_from: str, dynasty_to: str,
                  text_from: str, text_to: str, llm=None) -> dict:
    """分类变化类型。

    fast fail: LLM JSON 异常 -> degraded。type 不在枚举中 -> fallback substitution。
    """
    if llm is None:
        return {"error": "LLM 未初始化"}

    types_desc = {
        "emergence": "新出现", "disappearance": "消失", "substitution": "替代",
        "amplification": "放大", "diminution": "缩小",
        "transvaluation": "价值翻转", "transplantation": "移植",
    }
    types_list = "\n".join(f"- {t}: {d}" for t, d in types_desc.items())

    system = (
        f"你是叙事变化类型学专家。判断两个朝代之间母题变化的类型。\n"
        f"7 种变化类型:\n{types_list}\n\n"
        "只返回 JSON: {{\"change_type\": \"...\", \"confidence\": \"high|medium|low\", \"explanation\": \"...\"}}"
    )
    user = (
        f"母题 ID: {motif_id}\n从 {dynasty_from} 到 {dynasty_to}:\n"
        f"{dynasty_from}代: {text_from}\n{dynasty_to}代: {text_to}"
    )

    from ...config import settings as app_settings
    from ...observability.tracer import llm_call_context
    model = app_settings.effective_fast_model
    with llm_call_context("mcp:classify_change_type", "classify_change",
                          upstream="scan_motif_matrix",
                          downstream="infer_phylogeny"):
        # thinking_disabled=True: 结构化 JSON 任务关思考，避免慢 + 小预算被思考吃空
        resp = await llm.generate(system, user, model=model, max_tokens=512, temperature=0.3, thinking_disabled=True)
    content_raw = resp.content.strip()
    if content_raw.startswith("```"):
        content_raw = content_raw.split("\n", 1)[1].rsplit("```", 1)[0]

    from ...llm.json_utils import parse_llm_json
    try:
        data = parse_llm_json(content_raw)
    except json.JSONDecodeError as e:
        logger.error(f"classify_change_type JSON 解析失败: {e}")
        return {"error": f"AI 返回了无法解析的分类结果: {e}", "degraded": True}

    change_type = data.get("change_type", "substitution")
    if change_type not in CHANGE_TYPES:
        change_type = "substitution"

    return {
        "motif_id": motif_id,
        "dynasty_from": dynasty_from,
        "dynasty_to": dynasty_to,
        "change_type": change_type,
        "confidence": data.get("confidence", "medium"),
        "explanation": data.get("explanation", ""),
    }
