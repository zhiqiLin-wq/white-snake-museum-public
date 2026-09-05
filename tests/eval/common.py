"""记忆系统评测公共工具。

提供评测子项目共用的路径常量、JSON 读写、以及 D1 将用到的语义去重与
schema 校验框架。D5（纯规则）只用路径与 JSON 读写。
"""

import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = EVAL_DIR / "data"


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def write_json(data: dict, filename: str) -> Path:
    ensure_data_dir()
    path = DATA_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def read_json(filename: str) -> dict:
    path = DATA_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
