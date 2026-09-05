"""pytest fixtures — 共享测试数据加载。

Usage:
    pytest --collect-only  # 列出所有 fixture
    pytest rag_eval/tests/
"""
import sys
from pathlib import Path

import pytest

# 确保 agent 模块可导入
AGENT_DIR = Path(__file__).resolve().parent.parent / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

# 默认 ground_truth 路径
GT_DIR = Path(__file__).resolve().parent / "ground_truth"
CONFIG_DIR = Path(__file__).resolve().parent / "configs"


# ==============================================================================
# Fixture: 语料库
# ==============================================================================
@pytest.fixture(scope="session")
def corpus_fixture() -> dict[str, dict]:
    """加载 corpus.jsonl → {doc_id: {title, text, metadata}}。

    如果文件不存在，返回空 dict (skip 依赖真实数据的测试)。
    """
    from rag_eval.loader import load_corpus

    corpus_path = GT_DIR / "corpus.jsonl"
    if not corpus_path.exists():
        return {}
    return load_corpus(corpus_path)


# ==============================================================================
# Fixture: 查询
# ==============================================================================
@pytest.fixture(scope="session")
def queries_fixture() -> list[dict]:
    """加载 queries.jsonl → [{query_id, text, category, ...}]。

    如果文件不存在，返回空列表。
    """
    from rag_eval.loader import load_queries

    queries_path = GT_DIR / "queries.jsonl"
    if not queries_path.exists():
        return []
    return load_queries(queries_path)


# ==============================================================================
# Fixture: qrels
# ==============================================================================
@pytest.fixture(scope="session")
def qrels_fixture() -> dict[str, dict[str, int]]:
    """加载 qrels.tsv → {query_id: {doc_id: relevance}}。

    如果文件不存在，返回空 dict。
    """
    from rag_eval.loader import load_qrels

    qrels_path = GT_DIR / "qrels.tsv"
    if not qrels_path.exists():
        return {}
    return load_qrels(qrels_path)


# ==============================================================================
# Fixture: CI 查询子集
# ==============================================================================
@pytest.fixture(scope="session")
def ci_queries_fixture() -> list[dict]:
    """加载 queries_ci.jsonl → [{query_id, text, category, ...}]。

    如果文件不存在，返回空列表。
    """
    from rag_eval.loader import load_queries

    ci_path = GT_DIR / "queries_ci.jsonl"
    if not ci_path.exists():
        return []
    return load_queries(ci_path)


# ==============================================================================
# Fixture: 测试配置
# ==============================================================================
@pytest.fixture(scope="session")
def test_config_fixture() -> dict:
    """加载 test_config.yaml → 字典。

    如果文件不存在，返回空 dict。
    """
    import yaml

    config_path = CONFIG_DIR / "test_config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==============================================================================
# Fixture: 消融配置
# ==============================================================================
@pytest.fixture(scope="session")
def ablation_config_fixture() -> dict:
    """加载 chunking_ablation.yaml → 字典。

    如果文件不存在，返回空 dict。
    """
    import yaml

    config_path = CONFIG_DIR / "chunking_ablation.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
