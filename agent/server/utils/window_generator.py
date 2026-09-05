"""滑动窗口生成器 — 将段落列表按窗口大小和重叠量分组。

阶段 1 — P1-03.
用于标注流水线 Pass 1 Discovery 阶段。
"""
from typing import List, Dict, Any


def sliding_windows(
    paragraphs: List[Dict[str, Any]],
    window_size: int = 3,
    overlap: int = 1,
) -> List[Dict[str, Any]]:
    """生成滑动窗口。

    Args:
        paragraphs: 段落列表，每个元素含 index 和 text 字段
        window_size: 每个窗口包含的段落数
        overlap: 相邻窗口间的重叠段落数

    Returns:
        [{"start_idx": int, "paragraphs": [...], "window_id": str}, ...]

    Raises:
        ValueError: overlap >= window_size 或 window_size <= 0
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    if overlap >= window_size:
        raise ValueError(
            f"overlap ({overlap}) must be < window_size ({window_size})"
        )

    if not paragraphs:
        return []

    total = len(paragraphs)
    step = window_size - overlap  # 步长 = 窗口大小 - 重叠
    windows = []

    for i in range(0, total, step):
        window_paras = paragraphs[i:i + window_size]
        if not window_paras:
            break
        windows.append({
            "start_idx": i,
            "paragraphs": window_paras,
            "window_id": f"w{i // step}",
        })

    return windows
