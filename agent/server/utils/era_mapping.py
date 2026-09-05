"""共享朝代—章节映射常量。

供 comparison.py、evolution_workbench.py 统一引用。
(evolution_explorer.py 已在 U03-06 删除)
"""
from typing import Dict, List

# 朝代 -> 章节编号映射（基于白蛇传文献选集.txt 实际章节结构）
ERA_CHAPTER_MAP: Dict[str, List[str]] = {
    "唐": ["一"],
    "宋": ["二"],
    "明": ["三"],
    "清": ["四", "五", "六"],
    "现当代": ["七"],
}

# 朝代 -> 文本标题映射（基于白蛇传文献选集.txt 实际 7 篇核心文本）
ERA_TEXT_MAP: Dict[str, List[str]] = {
    "唐": ["唐代《李黄》"],
    "宋": ["宋代《西湖三塔记》"],
    "明": ["明代《白娘子永镇雷峰塔》"],
    "清": ["清代方成培《雷峰塔传奇》", "清代玉山堂主人《雷峰塔传奇》", "清代陈遇乾《义妖传》"],
    "现当代": ["现当代田汉《白蛇传》"],
}

# 所有朝代列表（按时间顺序）
ALL_ERAS: List[str] = ["唐", "宋", "元", "明", "清", "现当代"]

# 有文献记载的朝代（元朝缺失）
ERAS_WITH_LITERATURE: List[str] = ["唐", "宋", "明", "清"]
