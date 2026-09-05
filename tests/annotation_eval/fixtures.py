"""标注评测 fixtures：固定章节正文 + 金标准。

金标准由 build_golden() 程序化生成，避免人工手数 start/end 出错：
  对每个 (实体, 类别) 在每段里用 str.find 找出全部字面出现位置，
  得到的 span 天然满足 text[start:end] == entity。

注意: 金标准只包含"字面出现的实体"，与 annotate_passage 规则引擎 / 流水线
的 person+location 已知实体策略一致，保证精确匹配（δ=0）有可预期的上限。
"""

# 每段一句，避免段内跨行；实体已知且在 _ENTITY_KB 词表内
CHAPTER_001_TITLE = "评测样例章"

CHAPTER_001_PARAGRAPHS = [
    "许宣出涌金门，到西湖边。",
    "只见白娘子与青青在船上等候。",
    "许宣问道：娘子何故在此？",
    "法海自金山寺而来，手持钵盂与禅杖。",
    "许宣与白娘子同往断桥，许宣心中欢喜。",
    "白娘子为报恩而来，此乃宿缘。",
]

# 金标准实体清单：person/location 用精确字面，term/motif 也标字面词。
# 类别需与 annotation_knowledge_base._ENTITY_KB 的 known 列表对齐，否则
# 规则引擎/流水线规则匹配会漏，导致 recall 无法解释。
GOLDEN_ENTITIES = [
    ("许宣", "person"),
    ("白娘子", "person"),
    ("青青", "person"),
    ("法海", "person"),
    ("涌金门", "location"),
    ("西湖", "location"),
    ("金山寺", "location"),
    ("断桥", "location"),
    ("钵盂", "term"),
    ("禅杖", "term"),
    ("报恩", "motif"),
]


def build_chapter() -> dict:
    """返回 Node client get_literature_by_chapter 同构的章节对象。"""
    return {
        "chapterNumber": 1,
        "title": CHAPTER_001_TITLE,
        # 生产里 content 是 \n\n 分隔的全文，这里重建
        "content": "\n\n".join(CHAPTER_001_PARAGRAPHS),
    }


def build_paragraph_list() -> list[dict]:
    """返回流水线需要的段落列表 [{index, text}]。"""
    return [{"index": i, "text": t} for i, t in enumerate(CHAPTER_001_PARAGRAPHS)]


def _find_all(text: str, entity: str) -> list[int]:
    """返回 entity 在 text 中的全部起始位置（不重叠）。"""
    positions = []
    search_start = 0
    while True:
        pos = text.find(entity, search_start)
        if pos < 0:
            break
        positions.append(pos)
        search_start = pos + 1
    return positions


def build_golden() -> list[dict]:
    """构建金标准标注列表。"""
    golden = []
    for para_index, text in enumerate(CHAPTER_001_PARAGRAPHS):
        for entity, category in GOLDEN_ENTITIES:
            for start in _find_all(text, entity):
                golden.append({
                    "paragraph_index": para_index,
                    "entity": entity,
                    "category": category,
                    "start_char": start,
                    "end_char": start + len(entity),
                })
    return golden


def paragraph_texts() -> dict[int, str]:
    return {i: t for i, t in enumerate(CHAPTER_001_PARAGRAPHS)}


def chapter_content() -> str:
    return "\n\n".join(CHAPTER_001_PARAGRAPHS)
