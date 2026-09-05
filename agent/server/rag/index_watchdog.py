"""源文件变更检测 + 自动触发索引重建。"""
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class IndexWatchdog:
    """监控源文件 hash 变更，决定是否需要重建索引。"""

    HASH_FILE_NAME = "source_hashes.json"

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.hash_file = Path(__file__).resolve().parent.parent.parent / "data" / self.HASH_FILE_NAME
        self.last_hashes: dict[str, str] = self._load_hashes()

    def _load_hashes(self) -> dict:
        if self.hash_file.exists():
            try:
                return json.loads(self.hash_file.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _compute_hashes(self) -> dict[str, str]:
        hashes = {}
        if not self.data_dir.exists():
            return hashes
        for f in self.data_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in (".txt", ".docx", ".xlsx", ".xls"):
                try:
                    hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
                except Exception as e:
                    logger.warning(f"无法计算 hash: {f.name} — {e}")
        return hashes

    def needs_rebuild(self) -> bool:
        """对比当前 hash 与上次记录的 hash。"""
        current = self._compute_hashes()
        if not self.last_hashes:
            # 首次运行 → 保存 hash + 需要重建
            self._save_hashes(current)
            logger.info("首次运行，需要构建索引")
            return True
        if current != self.last_hashes:
            changed = set(current.items()) ^ set(self.last_hashes.items())
            logger.info(f"检测到源文件变更: {changed}")
            self._save_hashes(current)
            return True
        return False

    def _save_hashes(self, hashes: dict):
        self.hash_file.parent.mkdir(parents=True, exist_ok=True)
        self.hash_file.write_text(json.dumps(hashes, ensure_ascii=False, indent=2))
        self.last_hashes = hashes
