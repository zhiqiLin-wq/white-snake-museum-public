# 检索增强 + boost 完整规则（最终版，实现唯一依据）

> 2026-08-19 定稿。实现生产检索逻辑时严格按此规则，勿中途改动。

## 一、数据资产（已生成，可复用）

| 文件 | 内容 |
|---|---|
| `chunk_labels_full.json` | 427 chunk 的标签（原始值），chunk_id -> {genre, label} |
| `merged_vocab_full.json` | 四类词表 persons/locations/plot_units/topics，canonical/raw |
| `corpus.jsonl` | 427 chunk 原文，doc_id == chunk_id |

标签字段：
- 文学文本 chunk：persons/locations/plot_unit（原始值字符串列表）、plot_detail、summary
- 研究文献 chunk：涉及版本、分析主题、引用实体{persons/locations}、引用情节、summary
- query：query_type、persons/locations/plot_unit/分析主题（双名 [{"canonical","raw"}]）、plot_detail、对比维度、summary、genre

## 二、检索阶段（三路召回 → 候选池）

1. **dense 标签增强**：query 原文 embed → 标签增强 chunk 向量（标签词拼原文开头）余弦 top-100
2. **sparse 标签增强**：query「原文 + 标签词」→ 标签增强 BM25 top-100
   - 标签增强 BM25 索引：chunk「原文 + 标签词」（标签词 = canonical 实体 + plot_unit + 分析主题 + plot_detail + summary）
3. **标签语义**：query 标签文本 embed → 标签向量库余弦 top-100
   - 标签向量库：chunk 标签文本 embed

候选池 = 三路 union 去重。

## 三、boost 重排（候选池内 → top-10）

```
fused_base = 0.4 × minmax(dense) + 0.4 × minmax(bm25) + 0.2 × minmax(标签语义)
```

### 文学文本 chunk（按 query_type 分流）

| 层 | 匹配 | 权重/方式 |
|---|---|---|
| plot_unit | canonical 精确交集 | **硬分组**（情节题） |
| plot_detail | char bigram Jaccard 连续 | +0.20 × sim（情节题） |
| persons | canonical 交集 | +0.10 × IDF（情节/人物/地点/comparison） |
| locations | canonical 交集 | +0.09 × IDF（情节/人物/地点/comparison） |
| 称谓 raw | raw 交集 | +0.05/个（comparison + 称谓维度） |

### 研究文献 chunk

| 层 | 匹配 | 权重/方式 |
|---|---|---|
| 分析主题 | canonical 精确交集 | **硬分组**（comparison/研究分析） |
| 引用情节 | canonical 精确交集 | **硬分组**（comparison/研究分析）；情节题则 +0.05/个弱线性 |
| 引用实体 persons | canonical 交集 | +0.10 × IDF（情节/人物/地点/研究分析） |
| 引用实体 locations | canonical 交集 | +0.09 × IDF（情节/人物/地点/研究分析） |
| 涉及版本 | 版本命中 | +0.05（comparison + 版本维度） |

### 通用

| 层 | 匹配 | 权重 |
|---|---|---|
| 版本元信息 | GENRE_META[genre] 命中对比维度 | +0.05（comparison） |
| summary dense 语义 | cos(query.summary, chunk.summary) | +0.10 × sim |
| summary sparse 词面 | Jaccard(query.summary, chunk.summary) | +0.10 × jac |
| genre 因子 | chunk.genre == query.genre | ×1.2 |

### 最终分

```
final = (fused_base + boost) × genre因子
若 hard_match：final += HARD_OFFSET(10.0)
```

取 top-10。

## 四、关键实现细节

1. **IDF 加权**：`IDF = log((N+1)/(df+1))` 归一化到 [0.3, 1.5]；高频实体权重低、低频实体权重高。
2. **query_type 归一化**：`comparison -> comparison题`（LLM 可能漏「题」字）。
3. **过滤「待删除」词条**（canonical == "待删除"）。
4. **分流**：情节题答案在文学文本（研究文献引用情节不硬分组）；comparison/研究分析题答案可能在研究文献（才硬分组）。
5. **summary 匹配对象**：query.summary vs chunk.summary（两者都是白话概括）。
6. **Jaccard**：query.summary 与 chunk.summary 的 jieba 分词后 `|交集|/|并集|`。

## 五、query 标签解析（在线）

生产检索时 query 需打标（query_type/persons/locations/plot_unit/分析主题/plot_detail/对比维度/summary/genre）。复用 rag_eval 的 query 打标 prompt + 词表，LLM 输出 YAML。
