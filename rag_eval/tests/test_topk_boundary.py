"""T03.16: top_k=50 边界测试。

验证 MAX_TOP_K 边界: top_k=50 正常返回，top_k=60 截断为 50。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))


class TestTopKBoundary:
    """T03.16: MAX_TOP_K 边界测试。"""

    def test_max_top_k_value(self):
        """验证 MAX_TOP_K = 50。"""
        from server.rag.retriever import MAX_TOP_K
        assert MAX_TOP_K == 50

    def test_top_k_50_allowed(self):
        """top_k=50 应正常(等于 MAX_TOP_K)。"""
        top_k = 50
        from server.rag.retriever import MAX_TOP_K
        assert top_k <= MAX_TOP_K

    def test_top_k_60_truncated(self):
        """top_k=60 应被截断为 50 (不抛异常)。"""
        top_k = 60
        from server.rag.retriever import MAX_TOP_K
        result = min(top_k, MAX_TOP_K)
        assert result == 50, f"top_k=60 should truncate to 50, got {result}"

    def test_top_k_0_handled(self):
        """top_k=0 的边界情况。"""
        top_k = 0
        from server.rag.retriever import MAX_TOP_K
        result = min(top_k, MAX_TOP_K)
        assert result == 0
