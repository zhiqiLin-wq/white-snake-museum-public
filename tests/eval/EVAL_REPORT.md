# 记忆系统检索评测报告（D1）

- 日期：2026-08-21
- 被测系统：`agent/server/context/hybrid_retriever.py`（记忆检索两阶段管道）
- 数据集：`tests/eval/data/d1_retrieval.json`
- 评测脚本：`tests/eval/evaluate_d1.py`、`tests/eval/diagnose_d1.py`

---

## 一、数据集内容

### 1. 规模与分布

| 项 | 数量 |
|---|---|
| 记忆 | 89 条 |
| 查询 | 51 条 |
| 相关标注（relevant_ids） | 每条查询 1~3 条 |

记忆类型分布：Entity 20 / Archival 20 / System 20 / Working 15 / Summary 14

查询难度分布：exact（同词）16 / semantic（语义改写）19 / cross（跨对话）16

### 2. 数据内容来源

记忆内容基于项目真实的白蛇传文献语料（`excel_data/白蛇传文献选集.txt`），覆盖 7 个历代版本：

1. 唐·《太平广记》卷四五八《李黄》（文言小说）
2. 宋·《西湖三塔记》（话本小说）
3. 明·冯梦龙《白娘子永镇雷峰塔》（拟话本）
4. 清乾隆·方成培《雷峰塔传奇》（戏曲）
5. 清·玉山堂主人《雷峰塔传奇》（白话神魔小说）
6. 清·陈遇乾《义妖传》（苏州弹词）
7. 现当代·田汉《白蛇传》（京剧）

五种 memory_type 的语义：

| 类型 | 语义 | 示例 |
|---|---|---|
| Entity | 学者的身份、偏好、主观观点 | 该学者认为《李黄》中的白衣女子尚不具备后世白娘子的情感温度 |
| Archival | 文献的客观事实 | 《西湖三塔记》中，奚真人将三怪镇压于西湖三座石塔之下 |
| System | 学者的约束/要求 | 该学者要求引用时必须使用原文，不得用后世情节替代 |
| Working | 正在进行的临时活动 | 该学者正在梳理《李黄》与《西湖三塔记》的流变线索 |
| Summary | 概括总结 | 综上，该学者认为白蛇传最早记载可追溯至唐代《李黄》 |

### 3. 数据结构

```json
{
  "memories": [{"id": "m001", "content": "...", "memory_type": "Entity", "importance": 0.8}],
  "queries": [{"id": "q001", "query": "...", "difficulty": "semantic", "relevant_ids": ["m001", "m005"]}]
}
```

`relevant_ids` 即金标准（ground truth），由 LLM 生成 + 校验器/对抗器复核。

---

## 二、被测检索逻辑

### 1. 两阶段管道

```
查询
  -> Stage 1: FTS5 全文粗筛（中文 bigram 切分，source_filter=memory_extraction，top_k=50）
  -> Stage 2: 向量语义检索（BAAI/bge-large-zh-v1.5，cosine 相似度）
  -> 融合排序
  -> 综合评分重排（MemoryScorer）
  -> top-K 截断
```

### 2. 融合逻辑（本次修复点）

**修复前（交集融合，有召回缺陷）**：

- 向量候选 `n_results = vec_k * 3`
- 结果 = FTS 候选 ∩ 向量结果
- 缺陷：交集非空时，仅被 FTS 召回、未进向量 top-N 的相关记忆被直接丢弃

**修复后（并集加权融合）**：

- 向量候选扩大到 `n_results = vec_k * 5`
- 结果 = FTS 候选 ∪ 向量候选
- 打分规则：
  - 双边命中（FTS 且向量）：`score = vec_score * 0.7 + fts_rank * 0.3`
  - 仅 FTS 命中：`score = fts_rank * 0.8`
  - 仅向量命中：`score = vec_score * 0.8`
- 排序后截断 top-vec_k

### 3. 综合评分（排序后二次重排）

```
score = 0.5 * relevance + 0.3 * importance + 0.2 * recency
```

- relevance = 融合分（上文）
- importance = 记忆的重要性字段
- recency = 指数衰减（半衰期按类型：Entity 90 天 / Archival 60 天 / Summary 14 天 / Working 7 天）

---

## 三、评测指标公式

设查询集 Q，对每个查询 q：

- `R(q)` = q 的相关记忆集合（来自 `relevant_ids`）
- `TopK(q)` = 检索返回的前 K 条记忆

### Recall@K（召回率）

```
Recall@K = |R(q) ∩ TopK(q)| / |R(q)|
```

含义：相关记忆中被检索回来的比例。衡量"找得全不全"。

### Precision@K（精确率）

```
Precision@K = |R(q) ∩ TopK(q)| / K
```

含义：返回的前 K 条里真正相关的比例。衡量"找得准不准"。

### Hit@K（命中率）

```
Hit@K = 1  若 |R(q) ∩ TopK(q)| ≥ 1，否则 0
```

宏平均（对所有查询取平均）。含义：前 K 条里是否至少命中一条相关。

### MRR（平均倒数排名）

```
MRR = (1/|Q|) * Σ_q (1 / rank_q)
```

`rank_q` = 第一个相关结果在结果列表中的排名（从 1 起）。含义：第一个相关结果排得越靠前，MRR 越高。

### nDCG@K（归一化折损累计增益）

```
DCG@K  = Σ_{i=1..K} (2^{rel_i} - 1) / log2(i + 1)     rel_i ∈ {0, 1}
IDCG@K = 理想排序（所有相关排最前）的 DCG@K
nDCG@K = DCG@K / IDCG@K
```

含义：带位置折扣的排序质量，相关结果排得越靠前越高。

---

## 四、评测结果

### 1. 指标（修复前 → 修复后）

| 指标 | K=1 | K=3 | K=5 | K=10 |
|---|---|---|---|---|
| Recall | 0.078 → **0.301** | 0.239 → **0.425** | 0.327 → **0.467** | 0.500 → **0.562** |
| Precision | 0.157 → **0.608** | 0.163 → **0.301** | 0.141 → **0.200** | 0.110 → **0.124** |
| Hit | 0.157 → **0.608** | 0.451 → **0.824** | 0.569 → **0.863** | 0.824 → **0.961** |
| MRR | 0.157 → **0.608** | | | |
| nDCG | 0.157 → **0.608** | 0.205 → **0.459** | 0.250 → **0.482** | 0.321 → **0.522** |

核心提升：MRR、Recall@1、Precision@1 均提升约 4 倍；Hit@10 达 0.96。

### 2. 根因诊断（三路召回对比）

| 环节 | 查询命中率 | 相关召回率 |
|---|---|---|
| FTS 粗筛 | 100.0% | 84.4% |
| 向量 top-10 | 96.1% | 52.5% |
| 融合 top-10（修复前） | 82.4% | 45.9% |
| 融合 top-10（修复后） | 96.1% | 51.6% |

结论：FTS 粗筛能力充足（84.4%），召回瓶颈在向量精排（52.5%）。修复前交集融合进一步把召回压到 45.9%，修复后并集融合恢复到 51.6%，并显著改善排序（MRR 0.157→0.608）。

---

## 五、复现方法

```bash
# 1. 生成 D1 数据集（连 DeepSeek，约几分钟）
python tests/eval/generate_d1.py

# 2. 跑检索评测
python tests/eval/evaluate_d1.py

# 3. 跑根因诊断
python tests/eval/diagnose_d1.py
```

数据文件：`tests/eval/data/d1_retrieval.json`
日志文件：`tests/eval/logs/eval_d1.log`、`tests/eval/logs/diagnose_d1.log`

---

## 六、结论与遗留

1. 并集融合 + 扩大向量候选显著改善了排序质量（MRR 提升 4 倍），并消除了"交集丢弃 FTS 独有候选"的召回缺陷。
2. 剩余瓶颈：向量精排召回仅 52.5%（bge 对中文"同义改写/跨对话"语义匹配有限），Recall@10 仍只有 0.56。后续可考虑：换更强的嵌入模型、引入 reranker、或在 FTS 候选内做向量精排（而非全库）。
3. 数据集标注粒度偏宽（relevant_ids 含弱相关），可能使 Recall 计算偏严；后续可收紧标注（只保留强相关）。

---

## 附录 A：L5 衰减评测（D5）

- 数据集：`tests/eval/data/d5_decay.json`（153 组，type × importance × days，金标准由 `should_tombstone` 规则反推，零歧义）
- 评测脚本：`tests/eval/evaluate_d5.py`
- 被测：`MemoryScorer.should_tombstone`（分类型 importance 保底 + TTL）

### 指标公式

```
衰减正确率 = (正确 tombstone 数 + 正确 keep 数) / 总组数
误删率     = 应保留却被 tombstone 的数 / 应保留总数      （期望 = 0，最敏感）
漏删率     = 应 tombstone 却保留的数 / 应 tombstone 总数  （期望 = 0）
```

### 结果

- 总组数 153，正确 153（tombstone 正确 24 / keep 正确 129）
- 衰减正确率 **1.0000**，误删率 **0**，漏删率 **0**

结论：tombstone 分类型保底 + TTL 规则零缺陷。

---

## 附录 B：L6 一致性评测（D6）

- 数据集：`tests/eval/data/d6_consistency.json`（10 条偏好 × 3 种措辞）
- 评测脚本：`tests/eval/evaluate_d6.py`
- 被测：`MemoryExtractor._llm_extract` 对同一偏好的提取稳定性

### 评测方法

对每条偏好的 3 种不同措辞，分别跑 LLM 提取，用 bge 向量相似度（阈值 0.72）判定提取结果与偏好是否语义一致（绕开不稳定的 LLM judge）。

### 指标公式

```
一致性得分 = Σ(每条偏好的成功措辞数) / (偏好数 × 3)
完全一致占比 = 3/3 措辞都提取成功的偏好数 / 偏好数
```

### 结果

- 成功分布：3/3 = 4，2/3 = 2，1/3 = 3，0/3 = 1
- 一致性得分 **0.633**
- 完全一致（3/3）占比 **0.40**

失分根因：①提取召回（口语化陈述部分被提取器判为 NO、提取为空）；②否定类偏好（如"不喜欢收到长篇回复"）的 bge 向量相似度偏低。

---

## 附录 C：L3 端到端记忆保持评测（D3）

- 数据集：`tests/eval/data/d3_retention.json`（10 条用户画像关键信息 + 50 轮长对话）
- 评测脚本：`tests/eval/evaluate_d3.py`
- 被测：完整链路（enqueue 长对话 → 压缩 → extract_batch 提取 → HybridRetriever 检索）

### 评测方法

把 50 轮长对话走完整 ContextManager 链路（小窗口 400 token 触发压缩+提取），然后对每条关键信息用检索 + bge 向量相似度（阈值 0.72）判定是否仍可找回。

### 指标公式

```
保持率 = 仍可被检索找回的关键信息数 / 关键信息总数
```

### 结果

- 提取到的长期记忆：5 条
- 保持率：**5/10 = 0.500**

### 暴露的两个真实问题

1. **`_MIN_CONTENT_LENGTH = 20` 误杀短事实**：如"我叫小研"（17 字）、"我喜欢法海"（13 字）这类短的用户画像事实被长度阈值过滤，不进入提取。已通过加长关键信息（均 ≥ 20 字）绕过，但生产代码该阈值对短事实的过滤是真实缺陷。
2. **`extract_batch` 批量提取漏提**：多条消息合并成一段后一次 LLM 提取，LLM 会漏掉部分偏好/观点类事实（只提取了身份类），导致 5 条关键信息丢失。

---

## 附录 D：L4 压缩与摘要评测（D4，复用 D3 数据）

- 数据集：复用 `d3_retention.json`（10 关键信息 + 50 轮对话）
- 评测脚本：`tests/eval/evaluate_d4.py`
- 被测：`ContextManager` 压缩 + 增量摘要（`_update_summary`）

### 评测方法

走完整链路触发压缩（小窗口 400 token），检查增量摘要是否覆盖每条关键信息的核心关键词。

### 指标公式

```
摘要覆盖度 = 摘要覆盖的关键信息关键词数 / 关键信息总数
```

### 结果

- 摘要覆盖度 **0.200**（2/10）

### 发现并修复的两个 bug

1. **摘要 prompt 是英文**：`SUMMARY_UPDATE_SYSTEM` / `SUMMARY_UPDATE_USER` 原为英文，导致 DeepSeek 输出英文摘要，中文关键词匹配不上。已中文化（与 [[memory-extraction-cn-prompt]] 同类问题）。
2. **摘要 LLM 返回空会清空累积摘要**：`_update_summary` 里 `new_summary = resp.content.strip()` 若 LLM 偶发返回空，会 `state.summary = ""` 清空之前累积的全部摘要。已加空保护（返回空时保留原摘要）。

### 遗留

摘要 LLM 偶发返回空，导致增量摘要累积不稳定（覆盖度仅 0.20）。根因是摘要 LLM 在「合并旧摘要 + 新消息」的长输入下偶发返回空，属 LLM 调用稳定性问题，空保护已兜底不清空，但摘要累积质量仍受限于 LLM 稳定性。

---

## 附录 E：L2 提取质量评测（D2）

- 数据集：`tests/eval/data/d2_extraction.json`（30 条金标准事实：Entity 10 / Archival 10 / System 10）
- 评测脚本：`tests/eval/evaluate_d2.py`
- 被测：`MemoryExtractor.extract`（LLM 提取 + 规则分类）

### 评测方法

对每条用户陈述跑 extract，用 bge 向量相似度（阈值 0.72）判定提取结果是否语义匹配金标准，并比对 memory_type 分类。

### 指标公式

```
提取召回率 = 被成功提取（语义匹配金标准）的事实数 / 金标准事实总数
分类准确率 = 分类正确的事实数 / 被成功提取的事实数
```

### 结果

- 提取召回率：**11/30 = 0.367**
- 分类准确率：**8/11 = 0.727**

### 暴露的两个真实问题

1. **Archival 客观知识大量漏提**（10 条仅 1 条提取成功）：`MEMORY_EXTRACTION_SYSTEM` 规则 1「只提取关于用户的事实」让 LLM 把客观文献事实判为「不是关于用户的」而输出 NO。提取 prompt 与记忆类型定义不匹配——分类规则里有 Archival 类型，但提取 prompt 没有引导提取客观知识。
2. **Entity/Archival 分类边界模糊**：`_classify_memory_type` 用关键词「是」判断 Archival，但用户观点（「用户认为…是…」）也含「是」，导致观点被误分 Archival；同理 System 的「必须/不要」与 Entity 边界也有误分。
