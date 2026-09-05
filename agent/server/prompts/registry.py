"""集中式 Prompt 管理 — YAML 模板 + Jinja2 渲染 + 热重载。"""
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PromptRegistry:
    """管理所有 Prompt 模板。"""

    def __init__(self, templates_dir: Path):
        self._dir = templates_dir
        self.templates: dict[str, dict] = {}
        self._load_all()

    def _load_all(self):
        if not self._dir.exists():
            logger.warning(f"Prompt 模板目录不存在: {self._dir}")
            return
        try:
            import yaml
            for yaml_file in self._dir.glob("*.yaml"):
                try:
                    with open(yaml_file, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                    if data and "name" in data:
                        self.templates[data["name"]] = data
                except Exception as e:
                    logger.warning(f"加载 prompt 模板失败: {yaml_file} — {e}")
        except ImportError:
            logger.warning("pyyaml 未安装，无法加载 Prompt 模板")

    def render(self, name: str, variables: dict[str, Any]) -> tuple[str, str]:
        """渲染 prompt → (system_prompt, user_prompt)。"""
        tpl = self.templates.get(name)
        if not tpl:
            return "", f"Prompt 模板 '{name}' 未找到"

        try:
            from jinja2 import Environment, BaseLoader
            jinja = Environment(loader=BaseLoader())

            system = jinja.from_string(tpl.get("system", "")).render(**variables)
            user = jinja.from_string(tpl.get("user", "")).render(**variables)
            return system, user
        except ImportError:
            # 无 jinja2 时的简单替换
            system = tpl.get("system", "")
            user = tpl.get("user", "")
            for k, v in variables.items():
                placeholder = f"{{{{ {k} }}}}"
                system = system.replace(placeholder, str(v))
                user = user.replace(placeholder, str(v))
            return system, user

    def reload(self):
        """热重载所有模板（不重启服务）。"""
        self.templates.clear()
        self._load_all()
        logger.info(f"Prompt 模板已重载: {len(self.templates)} 个")

    def list_templates(self) -> list:
        return [{"name": name, "version": tpl.get("version", 1)} for name, tpl in self.templates.items()]
