"""B-020: decompose_motifs — 将追踪目标分解为叙事母题（供 EvolutionWorkbench E1 调用）。"""
import asyncio
import json
import logging
import time as _time

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "decompose_motifs",
    "description": "将追踪目标分解为 3-7 个叙事母题，每个区分 invariant core 和 variable surface。⚠️ 该工具内部调用 LLM，耗时 3-5 秒，适合明确的演变分析请求；简单检索/统计类问题请勿使用。",
    "inputSchema": {
        "type": "object",
        "properties": {
            "tracking_target": {"type": "string", "description": "追踪目标，如 '雷峰塔'、'白素贞'"},
            "target_type": {"type": "string", "description": "目标类型: location / character / theme / object / motif", "default": "location"},
        },
        "required": ["tracking_target", "target_type"],
    },
}


async def handler(tracking_target: str, target_type: str = "location",
                  retriever=None, llm=None) -> dict:
    """将追踪目标分解为叙事母题。

    fast fail: LLM JSON 格式异常 -> degraded。
    retriever 失败只影响上下文丰富度，不影响分解逻辑。
    """
    if llm is None:
        return {"error": "LLM 未初始化"}

    # 检索相关文献（非关键——失败不影响分解）
    # 单独限 12s：曾发生 Dense 检索+重排序耗时 ~38s，吃光工作台 45s 预算，
    # 导致 LLM 调用只剩 6.3s 被取消 → 预制模板兜底。检索慢就放弃上下文。
    docs_text = ""
    if retriever:
        try:
            _t0 = _time.monotonic()
            docs = await asyncio.wait_for(retriever.retrieve(tracking_target), timeout=12.0)
            logger.info(
                f"decompose_motifs 检索完成: {len(docs)} docs, "
                f"{(_time.monotonic() - _t0) * 1000:.0f}ms"
            )
            docs_text = "\n".join(
                f"[{d.get('metadata', {}).get('chapter_title', '')}@{d.get('metadata', {}).get('dynasty', '')}] "
                f"{d.get('content', '')[:300]}"
                for d in docs
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"decompose_motifs 检索超时 (>12s)，跳过上下文继续分解 "
                f"tracking_target={tracking_target}"
            )
        except Exception as e:
            logger.warning(f"decompose_motifs 检索失败（非关键，继续分解）: {e}")

    type_labels = {"location": "地点", "character": "人物", "theme": "主题", "object": "物象", "motif": "母题"}
    type_label = type_labels.get(target_type, target_type)

    system = (
        f"你是白蛇传叙事学分析专家。将{type_label}'{tracking_target}'分解为 3-7 个叙事母题。\n"
        "每个母题需区分:\n"
        "- core (invariant core): 跨朝代保持不变的叙事核心\n"
        "- variations (variable surface): 在不同朝代的表现变化\n\n"
        "另外，建议 1-3 个可能遗漏的母题（suggested_additions），每个需包含 id/name/type/core。\n"
        "只返回 JSON: {{\"motifs\": [{{\"id\": \"M1\", \"name\": \"...\", \"type\": \"...\", "
        "\"core\": \"...\", \"variations\": [], \"icon\": \"...\", "
        "\"highlights\": []}}], "
        "\"suggested_additions\": [{{\"id\": \"S1\", \"name\": \"...\", \"type\": \"invariant|variable\", \"core\": \"...\"}}]}}"
    )
    user = f"追踪目标: {tracking_target}\n目标类型: {type_label}\n"
    if docs_text:
        user += f"相关文献:\n{docs_text}"

    from ...config import settings as app_settings
    from ...observability.tracer import llm_call_context
    model = app_settings.effective_model
    with llm_call_context("mcp:decompose_motifs", "decompose_motifs",
                          upstream="decompose_motifs_node",
                          downstream="confirm_motifs"):
        # thinking_disabled=True: 结构化 JSON 任务关思考。默认思考模式曾致
        # 45s 工作台超时 → 预制模板兜底（decompose 目标越大思考越久）
        resp = await llm.generate(system, user, model=model, max_tokens=2048, temperature=0.7, thinking_disabled=True)
    content_raw = resp.content.strip()
    if content_raw.startswith("```"):
        content_raw = content_raw.split("\n", 1)[1].rsplit("```", 1)[0]

    from ...llm.json_utils import parse_llm_json
    try:
        data = parse_llm_json(content_raw)
    except json.JSONDecodeError as e:
        logger.error(f"decompose_motifs JSON 解析失败: {e}")
        return {"error": f"AI 返回了无法解析的母题分解结果: {e}", "degraded": True}

    motifs = data.get("motifs", [])
    if len(motifs) < 3:
        logger.warning(f"decompose_motifs 仅生成 {len(motifs)} 个母题，少于预期的 3-7 个")

    return {
        "tracking_target": tracking_target,
        "target_type": target_type,
        "motifs": motifs,
        "suggested_additions": data.get("suggested_additions", []),
    }
