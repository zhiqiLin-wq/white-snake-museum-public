"""chapter_summary — 章节情节骨架工具。

回答"第X章讲了什么"类快问：LLM 基于章节全文生成情节要点列表
（按段落锚定）+ 人物/地点/主题标签。进程内缓存，同一章节只调一次 LLM。
"""
import json
import logging
import re

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "chapter_summary",
    "description": (
        "生成指定章节的情节骨架：一句话概括 + 按段落锚定的情节要点 + "
        "主要人物/地点/主题标签。适用于'第X章讲了什么''帮我快速了解某版本剧情'"
        "'这章哪里讲到XX'类问题。⚠️ 首次调用内部生成摘要需数秒（同章节有缓存，"
        "重复调用即时返回）。需要引用具体原文句子请用 search_literature "
        "或 get_chapter_full_text。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "chapter_number": {"type": "integer", "description": "章节号（1-7）"},
        },
        "required": ["chapter_number"],
    },
}

# 喂给 LLM 的章节文本上限（语料为节选，7 章均 < 1 万字，覆盖绝大多数；超出时截断）
_MAX_CONTENT = 6000
_MAX_PARAS_PER_SUMMARY = 20

# 进程内缓存：chapter_number -> 摘要 dict（服务重启后自动重建）
_CACHE: dict[int, dict] = {}

_SYSTEM = (
    "你是白蛇传文本研究助手。基于给定章节文本生成情节骨架，只依据原文，不编造。"
    "输出 JSON（不要 markdown 代码块）：\n"
    '{"one_line": "一句话概括本章剧情（30字内）",\n'
    ' "plot_points": [{"paragraph_index": 段落号(整数,与输入前缀一致), '
    '"summary": "该段情节要点（40字内）"}],\n'
    ' "characters": ["主要人物"], "locations": ["主要地点"], "themes": ["主题关键词"]}\n'
    "plot_points 覆盖全部重要情节段，按段落顺序排列。"
)


async def handler(chapter_number: int, *, node_client=None, llm=None) -> dict:
    """生成章节情节骨架（LLM + 进程内缓存）。

    fast fail: node_client/llm 调用错误向上传播；
    JSON 解析失败返回明确 error（含 LLM 原始输出片段便于排查），不降级造假。
    """
    if node_client is None:
        return {"error": "Node API 客户端未初始化"}
    if llm is None:
        return {"error": "LLM 未初始化"}

    cached = _CACHE.get(chapter_number)
    if cached is not None:
        return {**cached, "cached": True}

    ch = await node_client.get_literature_by_chapter(chapter_number)
    if not ch:
        return {"error": f"章节 {chapter_number} 不存在（有效章节 1-7）"}

    content = (ch.get("content", "") or "").replace("\r\n", "\n").replace("\r", "\n")
    paras = [p.strip() for p in content.split("\n\n") if p.strip()]
    paras = paras[:_MAX_PARAS_PER_SUMMARY]
    if not paras:
        return {"error": f"章节 {chapter_number} 内容为空"}

    # 段落号前缀（0-based，与 get_paragraph_context / entity_appearances 一致）
    numbered = "\n\n".join(f"[段{i}] {p}" for i, p in enumerate(paras))
    truncated = False
    if len(numbered) > _MAX_CONTENT:
        numbered = numbered[:_MAX_CONTENT]
        truncated = True

    user = (
        f"章节 {chapter_number}《{ch.get('title', '')}》"
        f"{'（文本过长已截断，请基于可见部分）' if truncated else ''}：\n\n{numbered}"
    )
    resp = await llm.generate(
        _SYSTEM, user, max_tokens=2048, temperature=0.3, thinking_disabled=True
    )
    raw = resp.content.strip()
    data = _parse_llm_json(raw)
    if data is None or not isinstance(data.get("plot_points"), list):
        logger.error("chapter_summary JSON 解析失败: %s", raw[:200])
        return {
            "error": (
                "摘要生成失败（LLM 输出格式异常），请重试一次；"
                "若反复失败请勿编造情节，改用 get_chapter_full_text 让用户自行阅读原文。"
            ),
        }

    result = {
        "title": f"第{chapter_number}章《{ch.get('title', '')}》情节骨架",
        "chapter_number": chapter_number,
        "chapter_title": ch.get("title", ""),
        "one_line": str(data.get("one_line", ""))[:80],
        "plot_points": [
            {
                "paragraph_index": int(pp.get("paragraph_index", -1)),
                "summary": str(pp.get("summary", ""))[:120],
            }
            for pp in data["plot_points"][:_MAX_PARAS_PER_SUMMARY]
            if isinstance(pp, dict)
        ],
        "characters": [str(c) for c in (data.get("characters") or [])][:10],
        "locations": [str(c) for c in (data.get("locations") or [])][:10],
        "themes": [str(c) for c in (data.get("themes") or [])][:8],
        "cached": False,
    }
    _CACHE[chapter_number] = result
    return result


def _parse_llm_json(raw: str) -> dict | None:
    """解析 LLM JSON：直接解析 → 去围栏/提取大括号段 → 修尾逗号。"""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        candidate = re.sub(r",\s*([}\]])", r"\1", m.group(0))
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return None
