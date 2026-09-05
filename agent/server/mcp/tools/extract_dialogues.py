"""extract_dialogues — 角色对话提取工具。

在白蛇传语料中提取指定角色的对话原文。
通过对话标记模式（X道：/X说：/X曰：/X云： 等）定位角色发言，
返回对话原文及上下文，支持跨版本对比同一角色的发言风格。
"""
import json
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "extract_dialogues",
    "description": (
        "在白蛇传全文中提取指定角色的对话原文。"
        "通过对话标记模式（X道/X说/X曰/X云/X言 等）定位角色发言位置，"
        "返回对话原文及上下文。适用于'白娘子说了什么''法海的台词''许仙的对话'类问题。"
        "支持跨版本对比同一角色的发言风格和语气变化。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "character": {
                "type": "string",
                "description": "角色名称，如 '白素贞'、'法海'、'许仙'",
            },
            "chapter_number": {
                "type": "string",
                "description": "限定章节编号（如 '三'），留空则搜索全部章节",
                "default": "",
            },
        },
        "required": ["character"],
    },
}

MAX_DIALOGUES = 25

# 对话标记模式：角色名 + 道说曰云言 + 可选冒号
# 不使用前导 lookbehind（中文无空格分隔，lookbehind 会误杀所有匹配）
DIALOGUE_PATTERN = re.compile(
    r'({char})(笑道|说道|怒道|叹道|叫道|喊道|问道|答道|道|说|曰|云|言)(：|:)?'
)

# 对话内容延伸模式：引号或冒号后的文本
CONTENT_PATTERN = re.compile(r'[：:]\s*[""''「『]?([^\n。！？]{2,200}[。！？]?)')

_CORPUS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "rag_eval" / "ground_truth" / "corpus.jsonl"
)


def _load_corpus() -> list[dict]:
    """读取语料快照。"""
    try:
        chunks = []
        with open(_CORPUS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks
    except Exception as e:
        logger.error(f"读取语料快照失败: {e}")
        return []


def _extract_dialogues(text: str, character: str) -> list[dict]:
    """从文本中提取角色对话。"""
    results = []
    # 构建正则：角色名 + 对话标记
    pattern = DIALOGUE_PATTERN.pattern.replace("{char}", re.escape(character))
    for m in re.finditer(pattern, text):
        start = m.start()
        # 提取对话内容：从标记后开始，取到句号/感叹号/问号
        after_marker = m.end()
        # 向后查找对话内容（最多 300 字符）
        content_end = min(len(text), after_marker + 300)
        raw_content = text[after_marker:content_end]
        # 提取第一个句子（到句号/感叹号/问号为止）
        sentence_end = re.search(r'[。！？\n]', raw_content)
        if sentence_end:
            dialogue_content = raw_content[:sentence_end.end()].strip()
        else:
            dialogue_content = raw_content.strip()[:150]

        # 向前取上下文（最多 50 字符）
        context_start = max(0, start - 50)
        context_before = text[context_start:start]

        if len(dialogue_content) >= 2:
            results.append({
                "marker": m.group(0),
                "dialogue": dialogue_content,
                "context": context_before[-50:] + m.group(0) + dialogue_content,
            })

    return results


async def handler(character: str, chapter_number: str = "", **kwargs) -> dict:
    """提取角色对话。"""
    if not character or len(character.strip()) < 2:
        return {"error": "角色名称不能少于 2 个字符"}

    chunks = _load_corpus()
    if not chunks:
        return {"error": "语料快照不可读，无法提取对话。"}

    # 过滤研究文献
    chunks = [c for c in chunks if c.get("metadata", {}).get("source_type") == "primary_literature"]

    # 按章节过滤
    if chapter_number:
        chunks = [c for c in chunks if c.get("metadata", {}).get("chapter_number") == chapter_number]

    # 构建角色别名列表（同一角色的不同称呼）
    # 在白蛇传中，同一角色常有多个称呼
    aliases = _get_aliases(character)

    all_dialogues = []
    for chunk in chunks:
        text = chunk.get("text", "")
        meta = chunk.get("metadata", {})

        # 对角色名和每个别名都做匹配
        for name in aliases:
            dialogues = _extract_dialogues(text, name)
            for d in dialogues:
                d["character"] = name
                d["chapter_number"] = meta.get("chapter_number", "")
                d["chapter_title"] = meta.get("chapter_title", "")
                d["dynasty"] = meta.get("dynasty", "")
                d["paragraph_index"] = meta.get("paragraph_index", -1)
                all_dialogues.append(d)

        if len(all_dialogues) >= MAX_DIALOGUES:
            break

    if not all_dialogues:
        return {
            "title": f"'{character}' 对话提取",
            "character": character,
            "aliases_tried": aliases,
            "total": 0,
            "dialogues": [],
            "message": (
                f"未在语料中找到 '{character}' 的对话。"
                "该角色可能是其他版本的别名（同一人物在不同版本文本中名称不同），"
                "建议尝试其他称呼重新搜索。"
            ),
        }

    # 去重（同一对话可能被多个别名匹配到）
    seen = set()
    unique_dialogues = []
    for d in all_dialogues:
        key = d["dialogue"][:30]
        if key not in seen:
            seen.add(key)
            unique_dialogues.append(d)
        if len(unique_dialogues) >= MAX_DIALOGUES:
            break

    return {
        "title": f"'{character}' 对话提取",
        "character": character,
        "aliases_tried": aliases,
        "total": len(unique_dialogues),
        "truncated": len(all_dialogues) >= MAX_DIALOGUES,
        "dialogues": unique_dialogues,
        "note": (
            f"共提取 {len(unique_dialogues)} 条对话。"
            + ("结果已截断。" if len(all_dialogues) >= MAX_DIALOGUES else "")
        ),
    }


def _get_aliases(character: str) -> list[str]:
    """获取角色的常见别名。"""
    ALIAS_MAP = {
        "白素贞": ["白素贞", "白娘子", "白氏", "白珍娘", "白云仙姑", "娘子"],
        "许仙": ["许仙", "许宣", "汉文", "许汉文", "官人", "许生"],
        "法海": ["法海", "裴头陀", "裴文德", "妖僧", "禅师"],
        "小青": ["小青", "青青", "青儿", "丫鬟", "青蛇"],
        "许仕林": ["许仕林", "许士麟", "梦蛟", "许梦蛟"],
    }
    for key, aliases in ALIAS_MAP.items():
        if character in aliases or character == key:
            return aliases
    return [character]
