"""YAML 技能文件加载器。"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SkillLoader:
    """扫描 definitions/ 目录，解析 YAML 技能定义。"""

    def __init__(self, definitions_dir: Path):
        self._dir = definitions_dir

    def load_all(self) -> list[dict]:
        """加载所有 .yaml 技能定义。"""
        skills = []
        if not self._dir.exists():
            logger.warning(f"技能定义目录不存在: {self._dir}")
            return skills

        try:
            import yaml
        except ImportError:
            logger.warning("pyyaml 未安装")
            return skills

        for yaml_file in self._dir.glob("*.yaml"):
            if yaml_file.name.startswith("_"):
                continue  # 跳过模板文件
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if data and "name" in data:
                    skills.append(data)
                    logger.info(f"加载技能: {data['name']}")
            except Exception as e:
                logger.warning(f"加载技能文件失败: {yaml_file} — {e}")

        return skills

    def load_one(self, filename: str) -> dict | None:
        """加载单个技能文件。"""
        filepath = self._dir / filename
        if not filepath.exists():
            return None
        try:
            import yaml
            with open(filepath, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"加载技能失败: {filepath} — {e}")
            return None
