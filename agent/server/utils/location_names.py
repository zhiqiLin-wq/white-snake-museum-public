"""白蛇传相关地名映射表 — 向后兼容层。

★ 此文件保留以兼容旧代码引用。新代码请使用 entity_names 模块。
   from server.utils.entity_names import LOCATION_ALIASES, resolve_location_from_query
"""

# 从 entity_names 重导出以保持向后兼容
from .entity_names import (
    LOCATION_ALIASES,
    resolve_location_from_query,
    resolve_character_from_query,
    resolve_entity_from_query,
)

# 保留旧的 ALIAS_TO_CANONICAL 引用 (兼容旧代码)
ALIAS_TO_CANONICAL = {}

def _lazy_init():
    """惰性初始化反向索引 (兼容旧代码)。"""
    global ALIAS_TO_CANONICAL
    if not ALIAS_TO_CANONICAL:
        for canonical, aliases in LOCATION_ALIASES.items():
            for alias in aliases:
                ALIAS_TO_CANONICAL[canonical] = canonical

# 旧的 resolve_location 函数 (仅按精确名查找)
def resolve_location(name: str) -> str:
    """将用户输入的地名解析为规范名称 (向后兼容)。"""
    _lazy_init()
    return ALIAS_TO_CANONICAL.get(name, name)
