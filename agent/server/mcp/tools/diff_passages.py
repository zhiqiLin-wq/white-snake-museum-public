"""P2-7: diff_passages — 版本异文 token 级对比工具。

对两个章节/段落做逐字级 diff，返回每个差异块的位置、左文本、右文本、变化类型。
使用 difflib.SequenceMatcher 做字符级对齐，输出结构化差异列表。

与 compare_passages（比较文学分析 LLM）和 compare_chapters（章节级总结）不同：
本工具做的是"逐字对齐"——让用户看到改了哪几个字、加了什么、删了什么。
"""
import difflib
import logging
import re
from typing import List

logger = logging.getLogger(__name__)

CHANGE_TYPES = {
    "replace": "替换",
    "delete": "删除",
    "insert": "新增",
    "equal": "相同",
}


TOOL_DEF = {
    "name": "diff_passages",
    "description": (
        "对两个段落/章节做逐字级异文对比，返回每个差异块的位置、左文本、右文本、变化类型。"
        "用于白蛇传不同版本间的异文比对（如同一情节在唐/宋/明/清版本的文字差异）。"
        "返回结构化 JSON: [{position, leftText, rightText, changeType, changeLabel}]。"
        "典型场景：用户问「第3章和第8章的断桥情节文字差异」「同一情节在不同版本怎么改的」"
        "「逐字对比两个段落」。与 compare_passages（文学分析）互补：本工具是字级 diff，那个是语义级对比。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "left_chapter": {
                "type": "integer",
                "description": "左侧章节号（如 3=宋代第3章）",
            },
            "right_chapter": {
                "type": "integer",
                "description": "右侧章节号（如 8=明代第1章）",
            },
            "left_paragraph": {
                "type": "integer",
                "description": "左侧段落索引（0-based）。不传=整章对比",
            },
            "right_paragraph": {
                "type": "integer",
                "description": "右侧段落索引（0-based）。不传=整章对比",
            },
            "left_text": {
                "type": "string",
                "description": "直接传入左侧文本（优先于 chapter/paragraph 查原文）",
            },
            "right_text": {
                "type": "string",
                "description": "直接传入右侧文本（优先于 chapter/paragraph 查原文）",
            },
            "granularity": {
                "type": "string",
                "enum": ["char", "word"],
                "description": "对比粒度: char=字符级（默认，适合古文）; word=词级（按空格/标点分词，适合现代文）",
            },
            "context_window": {
                "type": "integer",
                "description": "差异块周围保留的相同字符数（默认 10，用于展示上下文）",
            },
        },
    },
}


def _tokenize(text: str, granularity: str = "char") -> list[str]:
    """按字符或词分词。"""
    if granularity == "word":
        # 中文按标点/空格分词
        tokens = re.findall(r'''[^\s，。！？；：、""''（）()【】《》\n]+|[^\w]''', text)
        return [t for t in tokens if t]
    return list(text)


def _diff_texts(left: str, right: str, granularity: str = "char",
                context_window: int = 10) -> list[dict]:
    """做 diff 并返回结构化差异列表。"""
    left_tokens = _tokenize(left, granularity)
    right_tokens = _tokenize(right, granularity)

    matcher = difflib.SequenceMatcher(None, left_tokens, right_tokens, autojunk=False)
    diffs = []
    position = 0  # 在左文本中的累积位置

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        left_chunk = "".join(left_tokens[i1:i2])
        right_chunk = "".join(right_tokens[j1:j2])

        if tag == "equal":
            position += (i2 - i1)
            continue

        # 取上下文
        ctx_start = max(0, i1 - context_window)
        ctx_end = min(len(left_tokens), i2 + context_window)
        left_context = "".join(left_tokens[ctx_start:ctx_end])
        right_ctx_start = max(0, j1 - context_window)
        right_ctx_end = min(len(right_tokens), j2 + context_window)
        right_context = "".join(right_tokens[right_ctx_start:right_ctx_end])

        diffs.append({
            "position": position + 1,  # 1-based
            "leftText": left_chunk or "(空)",
            "rightText": right_chunk or "(空)",
            "leftContext": left_context,
            "rightContext": right_context,
            "changeType": tag,
            "changeLabel": CHANGE_TYPES.get(tag, tag),
            "leftLength": len(left_chunk),
            "rightLength": len(right_chunk),
        })

        if tag in ("replace", "delete"):
            position += (i2 - i1)

    return diffs


async def handler(left_chapter: int = 0,
                  right_chapter: int = 0,
                  left_paragraph: int = -1,
                  right_paragraph: int = -1,
                  left_text: str = "",
                  right_text: str = "",
                  granularity: str = "char",
                  context_window: int = 10,
                  node_client=None) -> dict:
    """MCP handler: 逐字级异文对比。"""
    # ① 获取文本
    left = left_text if left_text else ""
    right = right_text if right_text else ""

    if not left and left_chapter > 0 and node_client:
        try:
            ch = await node_client.get_literature_by_chapter(left_chapter)
            if ch is None:
                return {"error": f"未找到第 {left_chapter} 章"}
            content = re.sub(r"\r\n|\r", "\n", ch.get("content", ""))
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            if left_paragraph >= 0 and left_paragraph < len(paragraphs):
                left = paragraphs[left_paragraph]
            else:
                left = content
        except Exception as ex:
            return {"error": f"获取左章节失败: {ex}"}

    if not right and right_chapter > 0 and node_client:
        try:
            ch = await node_client.get_literature_by_chapter(right_chapter)
            if ch is None:
                return {"error": f"未找到第 {right_chapter} 章"}
            content = re.sub(r"\r\n|\r", "\n", ch.get("content", ""))
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            if right_paragraph >= 0 and right_paragraph < len(paragraphs):
                right = paragraphs[right_paragraph]
            else:
                right = content
        except Exception as ex:
            return {"error": f"获取右章节失败: {ex}"}

    if not left and not right:
        return {"error": "未提供文本（left_text/right_text 空 且章节号无法定位）"}
    if not left:
        return {"error": "左侧文本为空"}
    if not right:
        return {"error": "右侧文本为空"}

    # ② 做 diff
    diffs = _diff_texts(left, right, granularity=granularity or "char",
                        context_window=context_window or 10)

    # ③ 统计
    stats = {
        "leftLength": len(left),
        "rightLength": len(right),
        "totalDiffs": len(diffs),
        "replaceCount": sum(1 for d in diffs if d["changeType"] == "replace"),
        "deleteCount": sum(1 for d in diffs if d["changeType"] == "delete"),
        "insertCount": sum(1 for d in diffs if d["changeType"] == "insert"),
    }
    stats["similarityRatio"] = round(
        difflib.SequenceMatcher(None, _tokenize(left, granularity),
                                 _tokenize(right, granularity)).ratio(), 4
    )

    # ④ 生成 human-readable diff preview
    diff_preview_lines = []
    for d in diffs[:30]:  # 最多展示 30 条
        arrow = "→" if d["changeType"] == "replace" else ("−" if d["changeType"] == "delete" else "+")
        diff_preview_lines.append(f"[{d['position']}] {arrow} {d['leftText']} → {d['rightText']}")
    diff_preview = "\n".join(diff_preview_lines)

    return {
        "diffs": diffs,
        "stats": stats,
        "diffPreview": diff_preview,
        "leftSource": f"第{left_chapter}章 段{left_paragraph}" if left_chapter and not left_text else "直接传入",
        "rightSource": f"第{right_chapter}章 段{right_paragraph}" if right_chapter and not right_text else "直接传入",
    }
