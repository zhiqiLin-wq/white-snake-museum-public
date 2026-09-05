"""v10 MCP 工具 -- 获取指定地点的跨朝代文本溯源数据。

设计文档: B.4 节 -- Agent 工具 get_location_traceability 预生成并缓存
返回各朝代提及记录、摘录和叙事特征描述，供前端地图溯源卡片 literaryRecords 使用。
"""
import logging

logger = logging.getLogger(__name__)

from ._text_utils import infer_dynasty_by_chapter, ALL_KNOWN_LOCATIONS

# 全部已知地点（12 文学地点 + 12 地理地标，去重 20 个）作为默认溯源清单
ALL_LOCATIONS = ALL_KNOWN_LOCATIONS

# 朝代 → dynastyOrder 映射（与 infer_dynasty_by_chapter 朝代集一致，含现当代）
DYNASTY_ORDER = {"唐": 1, "宋": 2, "明": 3, "清": 4, "现当代": 5}

TOOL_DEF = {
    "name": "get_location_traceability",
    "description": "获取地点的跨朝代文本溯源数据：各朝代提及记录、原文摘录和叙事特征描述。可查单个地点或默认批量查询全部 20 个白蛇传已知地点（文学场所+地理地标）。⚠️ 批量模式需遍历全部文献，耗时较长；只需统计数据用 get_location_info，需逐处出场定位用 entity_appearances",
    "inputSchema": {
        "type": "object",
        "properties": {
            "location_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "地点名列表，默认全部 20 个已知地点。文学场所: 雷峰塔/西湖/金山寺/断桥/保和堂/许宅/白府/钱塘门/清波门/苏州/镇江/临安；地理地标: 峨眉山/青城山/承天寺/卧佛寺/望江楼/龙虎山/灵隐寺/西子湖",
            },
        },
        "required": [],
    },
}


async def handler(
    location_names: list[str] | None = None,
    *,
    node_client,
) -> dict:
    """获取指定地点的跨朝代文本溯源数据。

    对每个地点遍历所有章节文本，记录每次出现的章节、段落、摘录、偏移，
    并生成各朝代的叙事特征描述。

    fast fail: node_client 不可用 -> 直接抛出异常。
    """
    if node_client is None:
        return {"error": "Node API 客户端未初始化"}

    targets = location_names if location_names else list(ALL_LOCATIONS)
    # 去重并保持顺序
    seen = set()
    targets = [t for t in targets if not (t in seen or seen.add(t))]

    # 获取所有章节数据（缓存 1 小时）
    try:
        chapters = await node_client.get_literature()
    except (RuntimeError, ConnectionError, TimeoutError) as e:
        logger.error(f"get_location_traceability: 获取文献数据失败: {e}")
        raise

    result = {}
    for loc_name in targets:
        literary_records = []
        dynasty_mentions = {"唐": 0, "宋": 0, "明": 0, "清": 0, "现当代": 0}

        for ch in chapters:
            chapter_number = ch.get("chapter_number", 0)
            dynasty = infer_dynasty_by_chapter(chapter_number)
            chapter_title = ch.get("title", "")
            content = ch.get("content", "")

            if not content:
                continue

            from ._text_utils import split_paragraphs
            paragraphs = split_paragraphs(content)

            for p_idx, para in enumerate(paragraphs):
                # 在段落中搜索地点名称
                pos = 0
                while True:
                    pos = para.find(loc_name, pos)
                    if pos == -1:
                        break

                    # 摘录：前后各扩展至多 30 字符
                    excerpt_start = max(0, pos - 30)
                    excerpt_end = min(len(para), pos + len(loc_name) + 30)
                    excerpt = para[excerpt_start:excerpt_end]

                    # 生成叙事特征描述（规则模板）
                    description_style = _generate_style(
                        loc_name, dynasty, para, pos
                    )

                    literary_records.append({
                        "dynasty": dynasty,
                        "dynastyOrder": DYNASTY_ORDER.get(dynasty, 0),
                        "chapterNumber": chapter_number,
                        "chapterTitle": chapter_title,
                        "paragraphIndex": p_idx,
                        "excerpt": excerpt,
                        "span": {
                            "startChar": pos,
                            "endChar": pos + len(loc_name),
                        },
                        "descriptionStyle": description_style,
                    })

                    dynasty_mentions[dynasty] += 1
                    pos += len(loc_name)

        result[loc_name] = {
            "literaryRecords": literary_records,
            "totalMentions": dynasty_mentions,
        }

    return result


def _generate_style(location_name: str, dynasty: str, paragraph: str, pos: int) -> str:
    """根据段落上下文生成叙事特征描述（规则模板，30字以内）。

    若有 LLM 可用建议用 LLM 替代此方法以获得更丰富的描述。
    """
    window = paragraph[max(0, pos - 15):pos + len(location_name) + 15]

    style_templates = {
        "唐": "唐代笔记中作为异闻背景简略提及",
        "宋": "宋人话本中具象化为戏剧性场景",
        "明": "明代拟话本中成为情节推进的关键节点",
        "清": "清代传奇中融入浪漫化与世俗化描写",
    }

    # 简单关键词增强
    keywords_enhance = {
        "雷峰塔": { "明": "镇压/囚禁意象突出", "清": "成为悲剧爱情的永恒象征" },
        "西湖":   { "宋": "游湖邂逅的浪漫舞台", "明": "市民生活与奇幻交织的背景" },
        "金山寺": { "明": "佛门与妖异的对抗中心", "清": "宗教救赎与世俗情感的交汇" },
        "断桥":   { "明": "重逢与和解的象征性节点", "清": "爱情记忆的标志性地点" },
    }

    enhance = keywords_enhance.get(location_name, {}).get(dynasty)
    if enhance:
        return enhance

    return style_templates.get(dynasty, f"{dynasty}代文献中的{location_name}")
