"""B-018: cooccurrence_matrix — 实体共现矩阵工具。"""
import logging

logger = logging.getLogger(__name__)

TOOL_DEF = {
    "name": "cooccurrence_matrix",
    "description": (
        "计算多个实体在白蛇传文本中的共现矩阵（滑动窗口检测共现关系）。"
        "⚠️ 口径说明：基于语义检索召回的段落（每实体取语义最相关的前 50 段）"
        "统计滑窗共现次数，是『相关段落中的共现』而非全文逐段共现矩阵；"
        "实体未在语料中出现时，召回的语义相近段落同样参与统计，解读需谨慎。"
        "向用户转述时请使用『相关段落中的共现』表述。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array", "items": {"type": "string"},
                "description": "实体名称列表，如 ['白素贞', '许仙', '法海']",
                "minItems": 2,
            },
            "window_size": {
                "type": "integer",
                "description": "共现窗口范围（前后 N 段），默认 10",
                "default": 10,
            },
        },
        "required": ["entities"],
    },
}


async def handler(entities: list, window_size: int = 10, retriever=None) -> dict:
    """计算实体共现矩阵（基于语义召回段落）。

    ⚠️ 数据口径：每个实体做语义检索取前 50 个相关段落，按段落位置
    滑窗统计共现——是"相关段落中的共现"，不是全文共现矩阵。
    结果统一标注 data_basis，防止 LLM 向用户误述口径。
    fast fail: retriever/CooccurrenceCalculator 错误向上传播。
    """
    if retriever is None:
        return {"error": "检索器未初始化"}

    from ...data.cooccurrence import CooccurrenceCalculator
    calc = CooccurrenceCalculator(retriever)
    result = await calc.compute(entities, window_size=window_size)
    if isinstance(result, dict) and "error" not in result:
        result.setdefault("title", "实体共现矩阵（相关段落）")
        result["data_basis"] = "semantic_recall"
        result["metric_label"] = "共现次数（基于每实体语义最相关的前 50 段落滑窗统计）"
        result["note"] = (
            "共现统计基于语义检索召回的相关段落（每实体前 50 段），"
            "非全文逐段共现矩阵。"
        )
    return result
