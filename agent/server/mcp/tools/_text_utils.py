"""段落切分与文本坐标映射共享工具。

口径对齐链（全部与生产一致）:
- 段落切分: 复现前端 splitParagraphs / recompute_paragraph_index.split_paragraphs_with_spans
  （\r\n→\n 规范化，按 \n\n 分段，trim，非空）
- compact→original 坐标映射: 用于 LLM 返回片段在原文中字面 find 不到时的容错定位，
  把"去空白后文本"里的命中偏移还原回原文坐标。逻辑参考 compare_and_annotate 的实现。
"""


def split_paragraphs(content: str) -> list[str]:
    """规范化换行并按双换行切分段落。

    与前端 sourceTreeStore.splitParagraphs 保持一致:
    先 \r\n / \r → \n，再按 \n\n 分段，trim，丢弃空段。
    返回段落列表（不含索引，索引即列表下标）。
    """
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    return [p.strip() for p in content.split("\n\n") if p.strip()]


def compact_to_orig_start(original: str, compact_start: int) -> int | None:
    """将"去空白后文本"的偏移映射回原文偏移。

    遍历 original，跳过空白字符（空格/换行/全角空格），数到第 compact_start
    个非空白字符即落点。找不到返回 None。
    """
    count = 0
    for i, ch in enumerate(original):
        if ch in (" ", "\n", "\r", "\u3000"):
            continue
        if count == compact_start:
            return i
        count += 1
    return None


def original_end_from_compact_len(original: str, orig_start: int, compact_len: int) -> int:
    """已知原文起点 orig_start 与"去空白后长度"，算原文终点（exclusive）。

    从 orig_start 起数 compact_len 个非空白字符，返回其右侧边界。
    """
    count = 0
    for i in range(orig_start, len(original)):
        if original[i] not in (" ", "\n", "\r", "\u3000"):
            count += 1
        if count == compact_len:
            return i + 1
    return orig_start + compact_len


# 朝代映射单一真相源：基于白蛇传文献选集.txt 实际章节结构
_DYNASTY_BY_CHAPTER = {
    1: "唐",      # 一、萌芽期 唐
    2: "宋",      # 二、萌芽期 宋
    3: "明",      # 三、成熟期 明
    4: "清",      # 四、成熟期 清乾隆中期
    5: "清",      # 五、流变 清
    6: "清",      # 六、流变 清
    7: "现当代",  # 七、流变期 现当代
}


def infer_dynasty_by_chapter(chapter_number: int) -> str:
    """根据章节编号推断朝代。

    单一真相源，供 dynasty_compare_elements / get_location_traceability /
    get_chapter_full_text 共用，消除三套本地映射的同章不同朝代矛盾。
    """
    return _DYNASTY_BY_CHAPTER.get(chapter_number, "未知")


# ============================================================================
# 地点清单单一真相源
# ============================================================================
# 文学地点（12 个）：文本溯源用——白蛇传文献中出现的地名/场所，
# 供 get_location_traceability 默认批量查询。
LITERARY_LOCATIONS = [
    "雷峰塔", "西湖", "金山寺", "断桥",
    "保和堂", "许宅", "白府", "钱塘门",
    "清波门", "苏州", "镇江", "临安",
]

# 地理地标（12 个）：地图聚焦用——有 WGS-84 坐标（与前端 coordinates.ts
# rawCoords 保持一致），供 focus_map_location 飞行动画。
GEO_LOCATION_COORDS = {
    "雷峰塔": (30.2409, 120.1512),
    "西湖":   (30.2741, 120.1551),
    "金山寺": (32.2106, 119.4485),
    "断桥":   (30.2608, 120.1442),
    "峨眉山": (29.564, 103.388),
    "青城山": (30.918, 103.607),
    "承天寺": (38.467, 106.273),
    "卧佛寺": (25.112, 99.169),
    "望江楼": (30.628, 104.089),
    "龙虎山": (28.108, 117.016),
    "灵隐寺": (30.2408, 120.0957),
    "西子湖": (30.26, 120.16),
}

# 全部已知地点（文学 ∪ 地理，去重保序）：
# 供 get_location_traceability 覆盖地理地标溯源（如峨眉山为白蛇修炼地）。
ALL_KNOWN_LOCATIONS = list(dict.fromkeys(LITERARY_LOCATIONS + list(GEO_LOCATION_COORDS.keys())))
