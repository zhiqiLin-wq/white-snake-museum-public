# RAG 检索评估系统（rag_eval）

对 RAG 检索流水线做**量化评测**的子系统：以人工校验的查询集（queries）和相关性标注（qrels）为真值，对比不同检索策略（稠密 / 稀疏 / 融合 / 全管线）与不同 LLM Provider 的 Recall@k、MRR、NDCG 等指标，支持消融实验。

> 本目录仅收录评测框架与当前数据集；开发过程中的一次性诊断、调参、修补脚本不随仓库发布。

## 目录结构

```text
rag_eval/
├── README.md                  # 本文件
├── RETRIEVAL_RULES.md         # 检索规则与口径约定
├── __init__.py
├── conftest.py                # pytest fixtures
├── loader.py                  # BEIR 格式数据加载器（corpus / queries / qrels）
├── metrics.py                 # 指标计算引擎（Recall@k / MRR / NDCG / Hit / Precision / Coverage）
├── evaluate.py                # 评估主入口（配置驱动，支持消融实验）
├── full_eval.py               # 全管线评估（含标签向量 / boost 路径）
├── fusion_ablation_core.py    # 融合策略消融核心（Dense / Sparse / RRF 等对比）
├── detail_log.py              # 检索组件明细日志与查询预处理
├── reporter.py                # 文本 / JSON 报告生成
├── benchmark.py               # 性能基准
├── configs/                   # 评估配置（Provider × 语料风格 × 是否标签增强）
├── ground_truth/              # 评测数据集（当前版本）
│   ├── corpus.jsonl           # 语料 chunk 快照
│   ├── queries_*.jsonl        # 查询集（academic / daily × deepseek / qwen）
│   ├── qrels_*.tsv            # 查询-chunk 相关性标注（augmented / src_only）
│   └── ANNOTATION_LOG.md      # 标注流程说明
├── scripts/                   # 数据集生成与批量评估
│   ├── generate_query_drafts.py      # LLM 生成查询草稿
│   ├── generate_eval_dataset.py      # 学术/日常双风格评测集
│   ├── generate_multi_doc_queries.py # 跨文献多文档查询（deepseek/qwen/ernie）
│   ├── setup_indices.py              # 评测专用索引构建
│   └── run_all_evals.py              # 批量跑全部 Provider 配置
└── tests/                     # 评测系统自测
    ├── test_self_check.py
    ├── test_ci_gate_mock.py
    ├── test_topk_boundary.py
    └── test_topk_multiplier_config.py
```

## 运行评估

```bash
# 单配置评估（配置文件见 configs/）
python -m rag_eval.evaluate --config rag_eval/configs/test_config_daily_deepseek.yaml --verbose
python -m rag_eval.evaluate --config rag_eval/configs/test_config_academic_deepseek.yaml --verbose

# 消融实验（fusion = 融合策略对比；weight = 权重敏感性）
python -m rag_eval.evaluate --config rag_eval/configs/test_config_daily_deepseek.yaml --ablation fusion
python -m rag_eval.evaluate --config rag_eval/configs/test_config_daily_deepseek.yaml --ablation weight

# 单条查询调试
python -m rag_eval.evaluate --config rag_eval/configs/test_config.yaml --query q_001 --debug

# 批量评估全部 Provider
python -m rag_eval.scripts.run_all_evals

# 自测
pytest rag_eval/tests/ -v
```

评估结果输出到 `rag_eval/results/`（运行产物，不入库）：`latest_report.txt`（文本报告）、`latest_report.json`（机器可读）。

## 配置文件

```yaml
evaluation:
  top_k: 10
configs:
  D: {bm25_enabled: false, reranker_enabled: false}   # 仅稠密向量
  S: {bm25_enabled: true,  reranker_enabled: false}   # 仅稀疏 BM25
  F: {bm25_enabled: true,  reranker_enabled: false, fusion_method: rrf}  # 融合
  R: {bm25_enabled: true,  reranker_enabled: true}    # 全管线（含重排）
thresholds:
  recall_at_10_fail: 0.65
  recall_at_10_warn: 0.75
```

## 指标说明

| 指标 | 说明 |
|------|------|
| Recall@k | 前 k 个结果命中相关 chunk 的比例 |
| MRR | 第一个相关结果排名的倒数均值 |
| NDCG@k | 归一化折损累积增益（相关性越高的结果排得越前得分越高） |
| Hit@k | 前 k 个结果中是否存在相关 chunk |
| Precision@k | 前 k 个结果中相关 chunk 的占比 |
| Coverage | 至少有一个相关 chunk 被检索到的查询比例 |

## 数据集说明

- 查询分**学术型**（academic，研究者视角的比较/溯源问题）与**日常型**（daily，博物馆访客的口语化问题）两种风格；
- qrels 由 LLM 初标 + 人工校验得到，`augmented` 为标签增强后的标注、`src_only` 为仅原文依据的标注（用于消融对照）；
- 数据集生成脚本（`scripts/`）的 LLM API Key 从 `agent/.env` 环境变量读取，不在代码中硬编码。
