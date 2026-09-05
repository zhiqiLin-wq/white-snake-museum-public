"""find_shared_passages — 跨章节共享文本片段定位工具（文本传承证据）。

对两个章节的段落提取公共文本片段（最长公共子串），找出"直接沿用的
情节表述/共享套语"，为版本演变对比提供段落级原文证据。
算法：8-gram 集合交集预筛（排除无共享文字的段落对）→
SequenceMatcher.get_matching_blocks 提取公共片段，纯 Python 无新依赖。
"""
import difflib
import logging

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "find_shared_passages",
    "description": (
        "对白蛇传两个章节的段落提取公共文本片段（最长公共子串），"
        "定位'直接沿用的情节表述/共享文字'，为'两个版本哪些情节是直接沿用的'"
        "'找出两章雷同/差异'类问题提供段落级原文证据。"
        "基于字符序列精确比对（非语义匹配）：语料为节选且改写幅度大时可能无片段，"
        "此时返回'无共享片段'结论——这本身是有价值的分析结果。"
        "向用户转述时请引用片段原文并注明所在章节与段落号。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "chapter_a": {"type": "integer", "description": "第一个章节号（1-7）"},
            "chapter_b": {"type": "integer", "description": "第二个章节号（1-7）"},
            "min_span_length": {
                "type": "integer",
                "description": "公共片段最小长度（字符），默认 12；调大只留长片段，调小捕捉更多共享表述",
                "default": 12,
            },
        },
        "required": ["chapter_a", "chapter_b"],
    },
}

# 片段分级阈值（字符数）
_LONG_SPAN = 40    # >= 40 字视为"长片段沿用"（可能整句/整段直接搬用）
_NGRAM = 8         # 预筛 n-gram 长度：存在 8 连字相同才进入精确比对
# 段落切分上限保护（控制两两比较规模）
_MAX_PARAS_PER_CHAPTER = 80
_EXCERPT_LEN = 100
_MAX_SPANS = 60


async def handler(chapter_a: int, chapter_b: int, min_span_length: int = 12,
                  node_client=None) -> dict:
    """定位两个章节的共享文本片段。

    fast fail: node_client 调用错误向上传播；章节不存在/相同返回明确错误。
    """
    if node_client is None:
        return {"error": "node_client 未初始化"}
    if chapter_a == chapter_b:
        return {"error": "两个章节号相同，无法比较（请提供不同的章节号）"}

    ch_a, ch_b = await node_client.get_literature_by_chapter(chapter_a), \
        await node_client.get_literature_by_chapter(chapter_b)
    if not ch_a or not ch_b:
        missing = chapter_a if not ch_a else chapter_b
        return {"error": f"章节 {missing} 不存在（有效章节 1-7）"}

    def _split(ch: dict) -> list[str]:
        content = (ch.get("content", "") or "").replace("\r\n", "\n").replace("\r", "\n")
        paras = [p.strip() for p in content.split("\n\n") if p.strip()]
        return paras[:_MAX_PARAS_PER_CHAPTER]

    def _ngrams(text: str) -> set[str]:
        return {text[i:i + _NGRAM] for i in range(len(text) - _NGRAM + 1)}

    paras_a, paras_b = _split(ch_a), _split(ch_b)

    spans = []
    seen_spans: set[str] = set()
    for i, pa in enumerate(paras_a):
        grams_a = _ngrams(pa)
        for j, pb in enumerate(paras_b):
            # 预筛：无 8 连字交集的段落对直接跳过（避免 O(n*m) 精确比对）
            if not (grams_a & _ngrams(pb)):
                continue
            sm = difflib.SequenceMatcher(None, pa, pb, autojunk=False)
            for blk in sm.get_matching_blocks():
                if blk.size < min_span_length:
                    continue
                span = pa[blk.a:blk.a + blk.size]
                # 同一片段可能出现在多对段落间，去重（保留首次出现的段落对）
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                ctx_a = pa[max(0, blk.a - 20):blk.a + blk.size + 20]
                ctx_b = pb[max(0, blk.b - 20):blk.b + blk.size + 20]
                spans.append({
                    "paragraph_a": i,
                    "paragraph_b": j,
                    "span": span,
                    "span_length": blk.size,
                    "level": "长片段沿用" if blk.size >= _LONG_SPAN else "共享表述",
                    "context_a": ctx_a,
                    "context_b": ctx_b,
                })

    spans.sort(key=lambda s: -s["span_length"])
    truncated = len(spans) > _MAX_SPANS
    spans = spans[:_MAX_SPANS]

    def _chapter_info(ch: dict, num: int) -> dict:
        return {
            "chapter_number": num,
            "title": ch.get("title", ""),
            "paragraph_count": len(_split(ch)),
        }

    result = {
        "title": f"第{chapter_a}章 × 第{chapter_b}章 共享文本片段定位",
        "data_basis": "literal_comparison",
        "metric_label": f"公共文本片段（最长公共子串，≥{min_span_length} 字）",
        "chapter_a": _chapter_info(ch_a, chapter_a),
        "chapter_b": _chapter_info(ch_b, chapter_b),
        "min_span_length": min_span_length,
        "span_count": len(seen_spans),
        "truncated": truncated,
        "shared_spans": spans,
    }
    if not spans:
        result["verdict"] = (
            f"两章之间未发现 ≥ {min_span_length} 字的连续相同片段，"
            "说明两个版本无直接文字沿承（或改写/翻译幅度极大，已无原文级重合），"
            "属独立创作或深度重构。建议：结合 dynasty_compare_elements 做"
            "主题/母题层面的演变对比。"
        )
    else:
        long_n = sum(1 for s in spans if s["level"] == "长片段沿用")
        longest = spans[0]["span_length"]
        result["verdict"] = (
            f"发现 {len(seen_spans)} 处共享文本片段（其中 ≥{_LONG_SPAN} 字的长片段 {long_n} 处，"
            f"最长 {longest} 字），可直接作为版本间文字沿承的证据，"
            "转述时请引用片段原文并注明段落位置。"
        )
    return result
