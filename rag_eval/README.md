# RAG 检索评估系统 (rag_eval)

基于 [rag-evaluation-test-design.md](../rag-evaluation-test-design.md) v1.0 设计，
严格遵循 [rag-eval-implementation-tasks.md](../rag-eval-implementation-tasks.md) 任务分解。

## 目录结构

```
rag_eval/
├── README.md                          # 本文件
├── __init__.py
├── loader.py                          # BEIR 格式数据加载器 (T01.01)
├── metrics.py                         # 指标计算引擎 (T02.01~T02.09 + T07.03)
├── evaluate.py                        # 评估主脚本 (T03.01~T03.19)
├── reporter.py                        # 报告生成器 (T04.01~T04.08)
├── benchmark.py                       # 性能基准测试 (T06.01~T06.06)
├── generate_dataset.py                # 数据集生成工具 (T01.02~T01.06)
├── conftest.py                        # pytest fixtures (T00.05)
├── _verify_imports.py                 # 导入验证 (T00.04)
├── _verify_chunk_ids.py               # chunk_id 跨索引一致性 (T00.08)
├── _smoke_test.py                     # 端到端 Smoke Test (T00.09)
├── configs/
│   ├── test_config.yaml               # 主评估配置 (T00.02)
│   └── chunking_ablation.yaml         # Chunking 消融配置 (T00.03)
├── ground_truth/                      # 数据集目录
│   ├── .gitkeep
│   ├── corpus.jsonl                   # (待生成) 语料库
│   ├── queries.jsonl                  # (待生成) 查询集
│   ├── queries_ci.jsonl               # (待生成) CI 子集
│   ├── qrels.tsv                      # (待生成) 相关性标注
│   └── ANNOTATION_LOG.md             # 标注流程日志模板 (T07.08)
├── results/                           # 评估结果输出
│   └── history/                       # 历史趋势
├── scripts/                           # 辅助脚本 (Phase 5/7)
│   └── __init__.py
└── tests/                             # 测试
    ├── __init__.py
    ├── test_self_check.py             # 评估系统自检 (T03.17)
    ├── test_ci_gate_mock.py           # CI 门禁 Mock 测试 (T08.02)
    ├── test_topk_boundary.py          # top_k 边界测试 (T03.16)
    └── test_topk_multiplier_config.py # 死配置审计 (T03.19)
```

## 快速开始

### 1. 生成数据集

```bash
# 导出语料库
python -m rag_eval.generate_dataset --export-corpus

#query-from-document


python -m rag_eval.scripts.generate_eval_dataset --provider deepseek --style daily
python -m rag_eval.scripts.generate_eval_dataset --provider deepseek --style academic

python -m rag_eval.scripts.generate_eval_dataset --provider qwen --style daily
python -m rag_eval.scripts.generate_eval_dataset --provider qwen --style academic

# 1. 确认语料库就绪
 .venv/Scripts/python -m rag_eval.generate_dataset --stats


#日常测评 
  python -m rag_eval.evaluate --config rag_eval/configs/test_config_daily_deepseek.yaml --verbose
  python -m rag_eval.evaluate --config rag_eval/configs/test_config_academic_deepseek.yaml --verbose
  
  python -m rag_eval.evaluate --config rag_eval/configs/test_config_daily_qwen.yaml --verbose
  python -m rag_eval.evaluate --config rag_eval/configs/test_config_academic_qwen.yaml --verbose
  
#权重
python -m rag_eval.evaluate --config rag_eval/configs/test_config_daily_deepseek.yaml --ablation weight --verbose

#recall情况
python rag_eval/query_analysis.py --config rag_eval/configs/test_config_daily_deepseek.yaml --output rag_eval/results/query_analysis.log

python rag_eval/query_analysis.py --config rag_eval/configs/test_config_academic_deepseek.yaml --output rag_eval/results/query_analysis.log
```

### 2. 运行评估

```bash
# CI 模式 (30 条查询, 约 10 分钟)
python -m rag_eval.evaluate --config rag_eval/configs/test_config.yaml --ci --verbose

# 全量评估 (200 条查询)
python -m rag_eval.evaluate --config rag_eval/configs/test_config.yaml --verbose

# 单条查询调试
python -m rag_eval.evaluate --config rag_eval/configs/test_config.yaml --query q_001 --debug

# 消融实验
python -m rag_eval.evaluate --config rag_eval/configs/test_config.yaml --ablation fusion
python -m rag_eval.evaluate --config rag_eval/configs/test_config.yaml --ablation weight
```

### 3. 查看报告

```bash
# 文本报告
cat rag_eval/results/latest_report.txt

# JSON 报告
cat rag_eval/results/latest_report.json
```

### 4. 运行测试

```bash
# Smoke test (零外部依赖)
python -m rag_eval._smoke_test

# 全部单元测试
pytest rag_eval/tests/ -v

# 仅 CI Mock 测试
pytest rag_eval/tests/test_ci_gate_mock.py -v
```

## 配置文件格式

### test_config.yaml

```yaml
evaluation:
  top_k: 10
configs:
  D: {bm25_enabled: false, reranker_enabled: false}   # Dense Only
  S: {bm25_enabled: true, reranker_enabled: false}      # Sparse Only
  F: {bm25_enabled: true, reranker_enabled: false,      # Fusion
      fusion_method: rrf}
  R: {bm25_enabled: true, reranker_enabled: true}       # Full Pipeline
thresholds:
  recall_at_10_fail: 0.65
  recall_at_10_warn: 0.75
```

## 指标说明

| 指标 | 说明 | 范围 |
|------|------|------|
| Recall@k | 前 k 结果中命中的相关文档比例 | [0, 1] |
| MRR | 第一个相关文档排名的倒数均值 | [0, 1] |
| NDCG@k | 归一化折损累积增益 | [0, 1] |
| Hit@k | 前 k 结果中是否有相关文档 | {0, 1} |
| Precision@k | 前 k 结果中相关文档占比 | [0, 1] |
| Coverage Rate | 至少有一个相关文档被检索到的查询比例 | [0, 1] |

## CI 集成

```bash
# CI 流水线
python -m rag_eval._verify_imports       # 验证导入
python -m rag_eval._smoke_test           # Smoke test
pytest rag_eval/tests/ -v                # 单元测试
python -m rag_eval.evaluate --config rag_eval/configs/test_config.yaml --ci

# Exit codes:
#   0: PASS (recall_at_10 >= warn_threshold)
#   0 + stderr: WARN (recall_at_10 < warn, >= fail)
#   1: FAIL (recall_at_10 < fail)
```
