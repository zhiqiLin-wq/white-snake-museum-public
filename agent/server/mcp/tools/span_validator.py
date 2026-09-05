"""Span Validator — 6 级回退链验证标注字符偏移。

阶段 0 — P0-04.

对 LLM 返回的 (entity, start_char, end_char) 进行验证。
精确匹配通过则保留。失败则逐级回退修正。无法修正则丢弃。

6 级回退链:
  Level 1: 精确匹配 (high confidence) — text[start:end] == entity
  Level 2: 窗口搜索 +-50字符 (medium confidence) — text.find(entity, start-50, end+50)
  Level 3: 全文搜索 (low confidence) — text.find(entity)
  Level 4: Compact 匹配去空白 (low confidence) — 去空白后 find → 映射回原始坐标
  Level 5: difflib 模糊匹配 (low confidence, ratio >= 0.7) — SequenceMatcher 滑动窗口
  Level 6: 丢弃 (not_found)

⚠️ 本模块同时暴露为 MCP 工具 `span_validator`，LLM 可直接调用来检查标注坐标。
也可作为内部库函数由 annotate_passage / compare_and_annotate 调用（函数级兜底）。
"""
import difflib
import logging
import re
from typing import Literal

logger = logging.getLogger(__name__)

# 类型定义
ValidationStatus = Literal["valid", "corrected", "discarded"]
ConfidenceLevel = Literal["high", "medium", "low", "not_found"]


class SpanValidationResult:
    """Span 验证结果。"""
    __slots__ = ("status", "start_char", "end_char", "confidence", "correction_method")

    def __init__(
        self,
        status: ValidationStatus,
        start_char: int,
        end_char: int,
        confidence: ConfidenceLevel,
        correction_method: str | None = None,
    ):
        self.status = status
        self.start_char = start_char
        self.end_char = end_char
        self.confidence = confidence
        self.correction_method = correction_method

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": self.confidence,
            "correction_method": self.correction_method,
        }


def validate_span(
    paragraph_text: str,
    entity: str,
    start_char: int,
    end_char: int,
) -> SpanValidationResult:
    """验证并修正 span 偏移。

    Args:
        paragraph_text: 段落原文（已 trim，与前端渲染文本一致）
        entity: LLM 声称的实体文本
        start_char: LLM 返回的起始偏移 (0-based, 段内)
        end_char: LLM 返回的结束偏移 (0-based, 段内)

    Returns:
        SpanValidationResult: 包含 status/start_char/end_char/confidence/correction_method

    Raises:
        ValueError: paragraph_text 为空 或 entity 为空/仅空白
    """
    # 输入验证 (fast fail)
    if not paragraph_text:
        raise ValueError("paragraph_text must not be empty")
    if not entity or not entity.strip():
        raise ValueError("entity must not be empty or whitespace-only")

    text_len = len(paragraph_text)

    # Level 1: 精确匹配
    if 0 <= start_char < end_char <= text_len:
        if paragraph_text[start_char:end_char] == entity:
            return SpanValidationResult(
                "valid", start_char, end_char, "high"
            )

    # Level 2: 窗口搜索 (+-50 字符)
    if 0 <= start_char < text_len:
        window_start = max(0, start_char - 50)
        window_end = min(text_len, max(end_char, start_char) + 50)
        pos = paragraph_text.find(entity, window_start, window_end)
        if pos >= 0:
            return SpanValidationResult(
                "corrected", pos, pos + len(entity), "medium",
                f"window_search:[{window_start}:{window_end}]"
            )

    # Level 3: 全文搜索
    pos = paragraph_text.find(entity)
    if pos >= 0:
        return SpanValidationResult(
            "corrected", pos, pos + len(entity), "low",
            "global_search"
        )

    # Level 4: Compact 匹配（去空白）
    entity_compact = entity.replace(" ", "").replace("　", "")
    text_compact = paragraph_text.replace(" ", "").replace("　", "")
    compact_pos = text_compact.find(entity_compact, max(0, start_char - 60))
    if compact_pos >= 0:
        orig_pos = _compact_to_original_pos(paragraph_text, compact_pos)
        if orig_pos >= 0:
            return SpanValidationResult(
                "corrected", orig_pos, orig_pos + len(entity), "low",
                "compact_match"
            )
    compact_pos = text_compact.find(entity_compact)
    if compact_pos >= 0:
        orig_pos = _compact_to_original_pos(paragraph_text, compact_pos)
        if orig_pos >= 0:
            return SpanValidationResult(
                "corrected", orig_pos, orig_pos + len(entity), "low",
                "compact_match_global"
            )

    # Level 5: difflib 模糊匹配
    fuzzy_result = _fuzzy_match(paragraph_text, entity)
    if fuzzy_result is not None:
        f_start, f_end, f_ratio = fuzzy_result
        return SpanValidationResult(
            "corrected", f_start, f_end, "low",
            f"fuzzy_match_ratio={f_ratio:.2f}"
        )

    # Level 6: 丢弃
    logger.warning(
        "Span validation DISCARDED: entity=%r start=%d end=%d text_len=%d",
        entity[:50], start_char, end_char, text_len
    )
    return SpanValidationResult(
        "discarded", -1, -1, "not_found"
    )


def _compact_to_original_pos(original_text: str, compact_pos: int) -> int:
    """将紧凑文本（去除空白后）中的位置映射回原始文本位置。

    逐字符遍历原始文本，跳过空白字符，计数非空白字符直到达到 compact_pos。
    返回 -1 表示映射失败。
    """
    count = 0
    for i, ch in enumerate(original_text):
        if ch not in (" ", "　"):
            if count == compact_pos:
                return i
            count += 1
    return -1


def _fuzzy_match(
    text: str, entity: str, min_ratio: float = 0.7
) -> tuple[int, int, float] | None:
    """使用 difflib.SequenceMatcher 滑动窗口模糊匹配。

    Args:
        text: 段落原文
        entity: 待匹配实体文本
        min_ratio: 最低相似度阈值

    Returns:
        (start, end, ratio) | None
    """
    if len(entity) < 3 or len(entity) > len(text):
        return None

    best_ratio = 0.0
    best_pos = -1

    for i in range(len(text) - len(entity) + 1):
        # 窗口大小 = entity 长度 +- 2 字符容差
        window_end = min(i + len(entity) + 2, len(text))
        window = text[i:window_end]
        for j in range(len(window) - len(entity) + 1):
            candidate = window[j:j + len(entity)]
            ratio = difflib.SequenceMatcher(None, entity, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_pos = i + j

    if best_ratio >= min_ratio and best_pos >= 0:
        return (best_pos, best_pos + len(entity), best_ratio)
    return None


# ============================================================================
#  MCP 工具封装：span_validator
#  供 Agent 检查自己/历史标注的坐标准确性，也供前端"标注体检"按钮调用。
# ============================================================================

TOOL_DEF = {
    "name": "span_validator",
    "description": (
        "验证一个或多个标注的字符偏移是否与原文精确匹配。"
        "当实体偏移不正确时自动通过 6 级回退链（精确→窗口→全文→去空白→模糊→丢弃）修正。"
        "⚠️ annotate_passage/compare_and_annotate 写出前系统自动调用作为兜底（无需主动调用）。"
        "仅在检查已有标注质量（如'第2章地点标注是否准确'）、或用户报标注标错字/标到空白时再显式调用。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "chapter_number": {
                "type": "integer",
                "description": "章节号，0 表示由 paragraph_text 直接传全文（无需查询章节）",
            },
            "paragraph_index": {
                "type": "integer",
                "description": "当检查单段标注时传入的段落索引（0-based）。chapter_number>0 时用于定位段落原文",
            },
            "paragraph_text": {
                "type": "string",
                "description": "直接传入的段落原文。优先使用；为空时从 chapter_number+paragraph_index 查原文",
            },
            "annotations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "实体原文（LLM 声称的标注目标词/句）"},
                        "start_char": {"type": "integer", "description": "段内起始偏移 0-based"},
                        "end_char": {"type": "integer", "description": "段内结束偏移 0-based（exclusive，同 Python 切片）"},
                        "id": {"type": "string", "description": "可选：标注 ID，原样返回便于前端识别"},
                    },
                    "required": ["text", "start_char", "end_char"],
                },
                "description": "待校验的标注列表（段内偏移）",
            },
        },
        "required": ["annotations"],
    },
}


async def handler(chapter_number: int = 0,
                  paragraph_index: int = 0,
                  paragraph_text: str = "",
                  annotations: list = None,
                  node_client=None) -> dict:
    """MCP handler: 校验并修正一批段内标注坐标。"""
    if not annotations:
        return {"error": "annotations 不能为空"}

    # ① 拿段落文本
    target_text = paragraph_text if paragraph_text else ""
    if not target_text and chapter_number > 0:
        if node_client is None:
            return {"error": "node_client 未注入，无法按章节查原文"}
        ch = await node_client.get_literature_by_chapter(chapter_number)
        if ch is None:
            return {"error": f"未找到第 {chapter_number} 章"}
        content = re.sub(r"\r\n|\r", "\n", ch.get("content", ""))
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if not (0 <= paragraph_index < len(paragraphs)):
            return {"error": f"paragraph_index={paragraph_index} 越界 [0, {len(paragraphs)-1}]"}
        target_text = paragraphs[paragraph_index]

    if not target_text:
        return {"error": "未提供原文（paragraph_text 空 且 chapter_number/paragraph_index 无法定位）"}

    results = []
    valid_count = 0
    corrected_count = 0
    discarded_count = 0

    for ann in (annotations or []):
        entity = (ann.get("text") or "").strip()
        s = int(ann.get("start_char", 0) or 0)
        e = int(ann.get("end_char", 0) or 0)
        ann_id = ann.get("id") or ""
        if not entity:
            discarded_count += 1
            results.append({
                "id": ann_id, "text": "", "status": "discarded",
                "confidence": "not_found", "reason": "entity 为空",
                "start_char": -1, "end_char": -1,
            })
            continue
        try:
            r = validate_span(target_text, entity, s, e)
        except ValueError as ex:
            discarded_count += 1
            results.append({
                "id": ann_id, "text": entity, "status": "discarded",
                "confidence": "not_found", "reason": f"输入异常: {ex}",
                "start_char": -1, "end_char": -1,
            })
            continue
        rd = r.to_dict()
        rd["id"] = ann_id
        rd["text"] = entity
        if rd["status"] == "valid":
            valid_count += 1
        elif rd["status"] == "corrected":
            corrected_count += 1
        else:
            discarded_count += 1
        results.append(rd)

    return {
        "summary": {
            "total": len(results),
            "valid": valid_count,
            "corrected": corrected_count,
            "discarded": discarded_count,
            "textLen": len(target_text),
            "chapterNumber": chapter_number,
            "paragraphIndex": paragraph_index,
        },
        "results": results,
    }
