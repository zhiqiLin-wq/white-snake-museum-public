"""get_version_metadata — 版本元数据查询工具。

返回白蛇传语料中各版本/章节的元数据信息（作者、朝代、体裁、文献来源等）。
供 Agent 在需要回答版本背景、作者信息、版本间关系等问题时调用，
减少 LLM 幻觉——这些信息是事实性的，不应由 LLM 编造。
"""
import logging

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "get_version_metadata",
    "description": (
        "查询白蛇传语料中各版本/章节的元数据信息（作者、朝代、体裁、文献来源、章节标题）。"
        "适用于'白蛇传有哪些版本''方成培是什么朝代的''雷峰塔传奇的作者是谁'类问题。"
        "返回结构化的版本信息列表，确保作者/朝代/体裁等事实性信息准确无误。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "chapter_number": {
                "type": "string",
                "description": "章节编号（如 '一'~'七'），留空返回全部版本",
                "default": "",
            },
        },
        "required": [],
    },
}

# 白蛇传语料版本元数据（与 corpus.jsonl 对齐）
VERSION_METADATA = [
    {
        "chapter_number": "一",
        "chapter_title": "一、溯源 唐：无名氏 《西湖三塔记》（文言小说）",
        "dynasty": "唐",
        "genre": "classical_chinese",
        "author": "无名氏",
        "work_title": "西湖三塔记",
        "source_file": "白蛇传文献选集.txt",
        "source_type": "primary_literature",
        "notes": "白蛇传故事雏形，记奚宣赞镇压西湖三怪（白蛇/乌鸡/獭）的故事，与后世白蛇传的核心人物和情节已有对应。",
    },
    {
        "chapter_number": "二",
        "chapter_title": "二、雏形 宋：无 名氏 《西湖三塔记》（话本）",
        "dynasty": "宋",
        "genre": "huaben",
        "author": "无名氏",
        "work_title": "西湖三塔记",
        "source_file": "白蛇传文献选集.txt",
        "source_type": "primary_literature",
        "notes": "宋元话本体，较唐代版本叙事更丰富，出现白娘娘、奚宣赞等核心人物雏形。",
    },
    {
        "chapter_number": "三",
        "chapter_title": "三、成熟 明：冯梦龙 《白娘子永镇雷峰塔》（拟话本）",
        "dynasty": "明",
        "genre": "imitation_huaben",
        "author": "冯梦龙",
        "work_title": "白娘子永镇雷峰塔",
        "source_file": "白蛇传文献选集.txt",
        "source_type": "primary_literature",
        "notes": "白蛇传故事成熟期标志，确立白娘子/许宣/法海三角关系和雷峰塔镇妖核心情节。收入《警世通言》。",
    },
    {
        "chapter_number": "四",
        "chapter_title": "四、成熟期 清乾隆中期：方成培 雷峰塔传奇（戏曲）",
        "dynasty": "清",
        "genre": "opera_libretto",
        "author": "方成培",
        "work_title": "雷峰塔传奇",
        "source_file": "白蛇传文献选集.txt",
        "source_type": "primary_literature",
        "notes": "传奇戏曲体，增加端阳/求草/水斗/断桥等经典折子，白娘子形象由妖向人的转变关键版本。",
    },
    {
        "chapter_number": "五",
        "chapter_title": "五、流变 清：玉山堂主人 《雷峰塔传奇》（白话神魔小说）",
        "dynasty": "清",
        "genre": "chapter_vernacular",
        "author": "玉山堂主人",
        "work_title": "雷峰塔奇传（白话神魔小说）",
        "source_file": "白蛇传文献选集.txt",
        "source_type": "primary_literature",
        "notes": "章回白话小说体，在方成培传奇基础上扩展为长篇小说，增加许仕林祭塔等情节。",
    },
    {
        "chapter_number": "六",
        "chapter_title": "六、流变 清末民初：佚名 《义妖传》（弹词）",
        "dynasty": "清末民初",
        "genre": "tanci",
        "author": "佚名",
        "work_title": "义妖传",
        "source_file": "白蛇传文献选集.txt",
        "source_type": "primary_literature",
        "notes": "弹词体，以七言唱词为主，民间说唱形式的白蛇传，语言更为通俗化。",
    },
    {
        "chapter_number": "七",
        "chapter_title": "七、流变期：现当代 田汉《白蛇传》（京剧）",
        "dynasty": "现当代",
        "genre": "script",
        "author": "田汉",
        "work_title": "白蛇传（京剧剧本）",
        "source_file": "白蛇传文献选集.txt",
        "source_type": "primary_literature",
        "notes": "京剧剧本，田汉改编，强化白娘子反抗精神和爱情主题，是现代最流行的白蛇传舞台版本。",
    },
]


async def handler(chapter_number: str = "", **kwargs) -> dict:
    """返回版本元数据。"""
    if chapter_number:
        results = [v for v in VERSION_METADATA if v["chapter_number"] == chapter_number]
        if not results:
            return {
                "error": f"未找到章节编号 '{chapter_number}'。可用章节: 一~七",
            }
    else:
        results = VERSION_METADATA

    return {
        "title": "白蛇传版本元数据",
        "total": len(results),
        "versions": results,
    }
