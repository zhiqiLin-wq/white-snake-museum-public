"""B-023: correlate_external — 关键变化点关联外部历史背景（供 EvolutionWorkbench E5 调用）。"""
import asyncio
import json
import logging

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "correlate_external",
    "description": (
        "将母题变化点与外部历史背景关联：输入变化描述列表，逐条分析其与"
        "指定朝代的社会变迁/文化思潮/政治环境的关联，结合 web_search 检索史料佐证。"
        "⚠️ 输入的 change_descriptions 是变化描述（如'白娘子形象从妖变为痴情仙女'），"
        "通常来自 decompose_motifs/classify_change_type 的分析结果，而非用户原话。"
        "每条变化内部调用 web_search + LLM，多条并行但整体耗时较长，供演化分析使用。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "change_descriptions": {
                "type": "array", "items": {"type": "string"},
                "description": "变化描述列表",
            },
            "dynasty": {"type": "string", "description": "变化发生的朝代，如 '清'"},
        },
        "required": ["change_descriptions", "dynasty"],
    },
}


async def handler(change_descriptions: list, dynasty: str,
                  llm=None, mcp_registry=None) -> dict:
    """关联外部历史背景。

    fast fail: 单条变化 LLM 失败 -> 该条 relevance=low，不阻塞整体。
    web_search 失败 -> 跳过外部参考（非关键路径）。
    """
    if llm is None:
        return {"error": "LLM 未初始化"}

    from ...config import settings as app_settings
    fast_model = app_settings.effective_fast_model

    system_base = (
        f"你是历史语境分析专家。分析以下文本变化与{dynasty}代历史背景的关联。\n"
        "从社会变迁、文化思潮、政治环境、宗教发展等角度分析。\n"
        "只返回 JSON: {{\"external_event\": \"...\", \"relevance\": \"high|medium|low\", "
        "\"explanation\": \"...\"}}"
    )

    async def _analyze_one_change(change: str) -> dict:
        """分析单条变化（内部串行: web_search -> LLM），供 asyncio.gather 并行调用。"""
        web_results = []
        source_url = ""
        if mcp_registry:
            try:
                ws_result = await mcp_registry.call_tool("web_search", {
                    "query": f"白蛇传 {change[:50]} {dynasty}代 历史背景",
                    "max_results": 3,
                })
                if ws_result and "result" in ws_result:
                    web_data = ws_result["result"]
                    if isinstance(web_data, dict):
                        web_results = web_data.get("results", [])
            except (KeyError, TypeError, ConnectionError, RuntimeError) as e:
                logger.warning(f"correlate_external web_search 失败（非关键）: {e}")

        if web_results:
            first = web_results[0]
            source_url = first.get("url", "")

        user = f"文本变化: {change}\n朝代: {dynasty}"
        if web_results:
            user += f"\n\n外部参考: {web_results[0].get('title', '')} - {web_results[0].get('body', '')[:200]}"

        try:
            from ...observability.tracer import llm_call_context
            with llm_call_context("mcp:correlate_external", "correlate_external",
                                  upstream="infer_phylogeny",
                                  downstream="formulate_hypothesis"):
                # thinking_disabled=True: 结构化 JSON 任务关思考，防止超时兜底
                resp = await llm.generate(system_base, user, model=fast_model, max_tokens=512, temperature=0.5, thinking_disabled=True)
            content_raw = resp.content.strip()
            if content_raw.startswith("```"):
                content_raw = content_raw.split("\n", 1)[1].rsplit("```", 1)[0]
            from ...llm.json_utils import parse_llm_json
            data = parse_llm_json(content_raw)
            return {
                "change": change,
                "external_event": data.get("external_event", ""),
                "relevance": data.get("relevance", "low"),
                "source_url": source_url,
                "explanation": data.get("explanation", ""),
            }
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"correlate_external 单条分析失败（非关键）: {e}")
            return {
                "change": change,
                "external_event": "",
                "relevance": "low",
                "source_url": source_url,
            }

    raw_results = await asyncio.gather(
        *[_analyze_one_change(c) for c in change_descriptions],
        return_exceptions=True,
    )
    correlations = [r for r in raw_results if isinstance(r, dict) and not isinstance(r, BaseException)]

    return {"correlations": correlations, "dynasty": dynasty}
