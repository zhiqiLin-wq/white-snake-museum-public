"""标注单元切分（抽样与真实验收共用，保证对齐）。

生产等价：先空行 \n\n 切段（sourceTree.splitParagraphs / annotate_passage._split_paragraphs），
再对超长段落做句子级二次切分，保证每个标注单元长度合理（≤ max_chars）。
"""

import re

DEFAULT_MAX_CHARS = 500

_SENT_SPLIT_RE = re.compile(r'(?<=[。！？!?；;])')


def split_paragraphs(content: str) -> list[str]:
    """空行切段，规范化换行（复刻生产 splitParagraphs）。"""
    normalized = re.sub(r"\r\n|\r", "\n", content)
    return [p.strip() for p in normalized.split("\n\n") if p.strip()]


def _chunk_long(text: str, max_chars: int) -> list[str]:
    """对超长段落做句子级切分：按换行+句末标点切句，再贪心合并到 ≤ max_chars。"""
    if len(text) <= max_chars:
        return [text]

    lines = re.split(r"\n+", text)
    sentences: list[str] = []
    for line in lines:
        for s in _SENT_SPLIT_RE.split(line):
            s = s.strip()
            if s:
                sentences.append(s)

    chunks: list[str] = []
    buf = ""
    for s in sentences:
        if len(buf) + len(s) <= max_chars:
            buf += s
        else:
            if buf:
                chunks.append(buf)
            if len(s) > max_chars:
                # 单句超长则硬切
                while len(s) > max_chars:
                    chunks.append(s[:max_chars])
                    s = s[max_chars:]
                buf = s
            else:
                buf = s
    if buf:
        chunks.append(buf)
    return chunks


def segment_content(content: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """把整章 content 切成标注单元列表（空行切 + 超长段句子级切）。"""
    units: list[str] = []
    for para in split_paragraphs(content):
        units.extend(_chunk_long(para, max_chars))
    return units
