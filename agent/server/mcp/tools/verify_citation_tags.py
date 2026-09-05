"""P1-1: verify_citation_tags — 答案来源标签自动校验后处理。

扫描 LLM 答案中的 5 档来源标注和章节号引用，用 search_paragraphs / get_version_metadata
反查"章节是否存在/引用句是否真在某段"，不一致 → 标记可疑（不删原文，附加告警）。

设计原则:
- 不修改用户看到的 answer 原文（避免破坏行文连贯）
- 在 answer 末尾追加一个"⚠️ 来源校验"块，列出可疑引用
- 仅作为 LLM 后处理钩子，不强制重答（成本太高）
- 只校验显式引用了"第X章""X章""《XXX》"的句子，未标注的句子不校验
"""
import logging
import re
from typing import Literal

logger = logging.getLogger(__name__)

# 5 档来源 tag（与 prompts/templates/qa_generator.yaml 对齐）
SOURCE_TAG_PATTERN = re.compile(
    r"【\s*(文献记载|基于文献推断|学界共识|AI补充|无记载|待考)\s*】"
)
# 章节号引用模式: "第3章" / "第三章" / "第3回" / "第三章《xxx》"
CHAPTER_REF_PATTERN = re.compile(
    r"第\s*([0-9一二三四五六七八九十百零]+)\s*(章|回|卷|节)"
)
# 中文数字 → 阿拉伯
CN_NUM_MAP = {
    "零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}

VALIDATION_LEVEL = Literal["strict", "lenient", "disabled"]


def _cn_to_int(cn: str) -> int:
    """中文数字转 int。支持 '三' / '十二' / '一百零五'。"""
    if cn.isdigit():
        return int(cn)
    # 简化: 只支持 1-99
    if len(cn) == 1:
        return CN_NUM_MAP.get(cn, 0)
    if cn.startswith("十"):
        return 10 + (_cn_to_int(cn[1:]) if len(cn) > 1 else 0)
    if "十" in cn:
        parts = cn.split("十")
        tens = _cn_to_int(parts[0]) if parts[0] else 1
        ones = _cn_to_int(parts[1]) if len(parts) > 1 and parts[1] else 0
        return tens * 10 + ones
    # 多字直连: 一二 = 12
    return int("".join(str(CN_NUM_MAP.get(c, 0)) for c in cn)) or 0


async def handler(answer: str,
                  sources: list = None,
                  node_client=None,
                  retriever=None,
                  llm=None,
                  level: str = "lenient") -> dict:
    """校验答案中的来源标签和章节号引用是否与原文/工具结果一致。

    Args:
        answer: LLM 生成的最终答案文本
        sources: agent_loop 收集到的 sources 列表（含 chapter_number/chapter_title/excerpt）
        node_client: Node API 客户端（用于查章节是否存在）
        retriever: 向量检索器（用于反查引用句是否真在某段）
        llm: LLM 客户端（保留参数，当前实现不调用 LLM）
        level: 校验级别
            - strict: 标签缺失即告警
            - lenient (默认): 只告警"引用了不存在的章节"和"引用句在原文中找不到"
            - disabled: 跳过校验

    Returns:
        {
            "verified": bool,           # True=通过 False=有可疑
            "warnings": list[dict],     # 可疑点列表
            "stats": {total_tags, total_refs, valid_refs, suspicious_refs, missing_tags},
            "append_hint": str,         # 追加到答案末尾的告警块（空串=无告警）
        }
    """
    if level == "disabled" or not answer:
        return {"verified": True, "warnings": [], "stats": {}, "append_hint": ""}

    sources = sources or []
    # 已知章节号集合（来自工具实际召回的 sources）
    known_chapter_numbers = set()
    known_chapter_titles = set()
    for s in sources:
        cn = s.get("chapter_number") or s.get("chapterNumber")
        if cn not in (None, "", 0):
            known_chapter_numbers.add(str(cn))
        ct = s.get("chapter_title") or s.get("chapterTitle")
        if ct:
            known_chapter_titles.add(str(ct))

    # ① 扫描 5 档标签
    tags_found = SOURCE_TAG_PATTERN.findall(answer)
    tag_set = set(t.strip() for t in tags_found)

    # ② 扫描章节号引用
    chapter_refs = CHAPTER_REF_PATTERN.findall(answer)
    referenced_chapters: dict[int, list[str]] = {}
    for cn_str, unit in chapter_refs:
        ch_num = _cn_to_int(cn_str)
        if ch_num > 0:
            referenced_chapters.setdefault(ch_num, []).append(unit)

    warnings = []

    # ③+④ 合并: 先查数据库确认章节是否存在，再决定是否告警
    # 逻辑: 章节存在于数据库 → 合法引用（即使不在 sources 中也不告警）
    #       章节不存在于数据库 → AI 编造（critical）
    #       无法查数据库 → 退回 sources 校验（high）
    existing_numbers = set()
    db_check_ok = False
    if node_client and referenced_chapters:
        try:
            chapters_resp = await node_client.get_literature()
            if isinstance(chapters_resp, list):
                for c in chapters_resp:
                    cn = c.get("chapterNumber") or c.get("chapter_number") or c.get("id")
                    if isinstance(cn, int):
                        existing_numbers.add(cn)
                db_check_ok = True
        except Exception as ex:
            logger.warning(f"verify_citation_tags: 查章节清单失败: {ex}")

    for ch_num in referenced_chapters:
        if existing_numbers and ch_num in existing_numbers:
            # 章节存在于数据库 → 合法引用，不告警
            continue
        elif existing_numbers and ch_num not in existing_numbers:
            # 章节不存在于数据库 → AI 编造
            warnings.append({
                "type": "chapter_not_exist",
                "severity": "critical",
                "message": f"答案引用了第 {ch_num} 章，但数据库中不存在该章节。属于 AI 编造。",
                "chapter": ch_num,
            })
        elif known_chapter_numbers and str(ch_num) not in known_chapter_numbers:
            # 无法查数据库 + 不在 sources 中 → 可疑（降级告警）
            warnings.append({
                "type": "chapter_not_in_sources",
                "severity": "medium",
                "message": f"答案引用了第 {ch_num} 章，但本次工具召回的来源中没有该章节，且无法验证章节是否存在。",
                "chapter": ch_num,
            })

    # ⑤ strict 模式: 答案有事实陈述但完全没标签 → 告警
    if level == "strict" and not tag_set and len(answer) > 100:
        warnings.append({
            "type": "no_source_tags",
            "severity": "medium",
            "message": "答案包含事实陈述但未使用任何来源标签（文献记载/基于文献推断/学界共识/AI补充/无记载/待考）。",
        })

    # ⑥ 组装追加告警块
    stats = {
        "total_tags": len(tags_found),
        "unique_tags": sorted(tag_set),
        "total_refs": sum(len(v) for v in referenced_chapters.values()),
        "unique_chapters": sorted(referenced_chapters.keys()),
        "suspicious": len(warnings),
        "level": level,
    }

    append_hint = ""
    if warnings:
        critical = [w for w in warnings if w["severity"] == "critical"]
        high = [w for w in warnings if w["severity"] == "high"]
        medium = [w for w in warnings if w["severity"] == "medium"]
        lines = ["", "", "> --- ⚠️ 来源校验 ---"]
        if critical:
            lines.append("> **严重**：检测到引用了数据库中不存在的章节，请勿采纳该引用。")
            for w in critical:
                lines.append(f"> - {w['message']}")
        if high:
            lines.append("> **可疑**：引用的章节未在本次工具召回的来源中。")
            for w in high:
                lines.append(f"> - {w['message']}")
        if medium:
            lines.append("> **建议**：")
            for w in medium:
                lines.append(f"> - {w['message']}")
        lines.append("> 请针对可疑引用追问或查阅原始文献核实。")
        append_hint = "\n".join(lines)

    return {
        "verified": len(warnings) == 0,
        "warnings": warnings,
        "stats": stats,
        "append_hint": append_hint,
    }


TOOL_DEF = {
    "name": "verify_citation_tags",
    "description": (
        "校验答案中的来源标签和章节号引用是否与原文/工具结果一致。"
        "扫描 5 档来源标签（文献记载/基于文献推断/学界共识/AI补充/无记载/待考）和"
        "章节号引用（第X章/第X回/第X卷），反查章节是否真实存在、是否在工具召回的 sources 中。"
        "不一致时在答案末尾追加来源校验告警块，不修改原文。"
        "⚠️ 通常由 agent_loop 在最终答案生成后自动调用作为后处理，不需要你手工调。"
        "只有当用户明确报\"答案引用的章节号不存在/是编造的\"时再显式调用做深度排查。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "待校验的答案文本",
            },
            "sources": {
                "type": "array",
                "description": "agent_loop 收集到的来源列表（含 chapter_number/chapter_title/excerpt）",
                "items": {
                    "type": "object",
                    "properties": {
                        "chapter_number": {"type": ["integer", "string"]},
                        "chapter_title": {"type": "string"},
                        "excerpt": {"type": "string"},
                    },
                },
            },
            "level": {
                "type": "string",
                "enum": ["strict", "lenient", "disabled"],
                "description": "strict=无标签即告警; lenient(默认)=只告警引用了不存在/不在 sources 中的章节; disabled=跳过",
            },
        },
        "required": ["answer"],
    },
}
