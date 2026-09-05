"""LLM 输出 JSON 容错解析。

deepseek 等模型偶发输出带小毛病的 JSON（尾逗号、单引号、缺右括号、
```json 残留等），标准 json.loads 直接失败会导致整条链路降级到模板兜底。
这里提供统一入口：标准解析优先，json_repair 兜底修复小毛病。

行为约定：
- 标准解析成功 -> 原样返回
- 标准失败但 json_repair 修复成功 -> 返回修复结果（记 WARNING 便于 trace 排查）
- 两者都失败 -> 抛出标准解析的原始 JSONDecodeError，调用方降级逻辑不变
"""
import json
import logging

import json_repair

logger = logging.getLogger(__name__)


def parse_llm_json(content: str):
    """解析 LLM 返回的 JSON，json_repair 容错兜底。失败抛 JSONDecodeError。"""
    try:
        return json.loads(content)
    except json.JSONDecodeError as first_err:
        try:
            repaired = json_repair.loads(content)
        except Exception:
            logger.warning(f"json_repair 兜底异常，按解析失败处理: {first_err}")
            raise first_err
        if isinstance(repaired, (dict, list)) and repaired:
            logger.warning(f"json_repair 兜底成功 (标准解析失败: {first_err})")
            return repaired
        raise first_err
