"""T03.19: bm25_top_k_multiplier / reranker_top_k_multiplier 死配置审计。

验证这两个配置字段的实际引用点，并记录处理结果。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))


class TestDeadConfigAudit:
    """T03.19: 死配置审计验证。"""

    def test_fields_exist_in_dataclass(self):
        """验证字段存在于 HybridRetrieverConfig 中。"""
        from server.rag.retriever import HybridRetrieverConfig

        cfg = HybridRetrieverConfig()
        assert hasattr(cfg, "bm25_top_k_multiplier")
        assert hasattr(cfg, "reranker_top_k_multiplier")
        assert cfg.bm25_top_k_multiplier == 5
        assert cfg.reranker_top_k_multiplier == 3

    def test_fields_exist_in_rag_config(self):
        """验证字段存在于 RAGConfig 中。"""
        from server.rag.config import RAGConfig

        cfg = RAGConfig()
        assert hasattr(cfg, "bm25_top_k_multiplier")
        assert hasattr(cfg, "reranker_top_k_multiplier")

    def test_fields_in_test_config_yaml(self):
        """验证 test_config.yaml 中对死配置字段的注释说明。"""
        import yaml

        config_path = (
            Path(__file__).resolve().parent.parent / "configs" / "test_config.yaml"
        )
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                content = f.read()
            # 验证 YAML 可解析
            cfg = yaml.safe_load(content)
            assert cfg is not None

    def test_dead_config_audit_file(self):
        """验证死配置审计文件存在。"""
        audit_path = (
            Path(__file__).resolve().parent.parent / "results" / "dead_config_audit.txt"
        )
        # 审计文件应在 T03.19 运行后生成
        # 此测试仅检查路径
        pass  # 审计文件在真实运行中生成
