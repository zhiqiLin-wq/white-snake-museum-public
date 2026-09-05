"""技能注册中心 + 触发匹配。"""
import re
import logging

logger = logging.getLogger(__name__)


class SkillRegistry:
    """管理所有已加载的技能，提供触发匹配。"""

    def __init__(self):
        self._skills: dict[str, dict] = {}

    def register(self, skill_def: dict):
        name = skill_def.get("name", "")
        if name:
            self._skills[name] = skill_def

    def match(self, user_input: str) -> list[dict]:
        """匹配用户输入，返回命中的技能列表（按优先级）。"""
        matched = []
        for name, skill in self._skills.items():
            triggers = skill.get("triggers", [])
            for pattern in triggers:
                try:
                    if re.search(pattern, user_input):
                        matched.append(skill)
                        break
                except re.error:
                    logger.warning(f"技能 '{name}' 的正则无效: {pattern}")
        return matched

    def get(self, name: str) -> dict | None:
        return self._skills.get(name)

    def list_all(self) -> list[dict]:
        return [
            {
                "name": s.get("name", ""),
                "display_name": s.get("display_name", ""),
                "description": s.get("description", ""),
                "category": s.get("category", ""),
                "icon": s.get("icon", "📖"),
                "triggers": s.get("triggers", []),
            }
            for s in self._skills.values()
        ]

    def unregister(self, name: str):
        self._skills.pop(name, None)
