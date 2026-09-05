# 白蛇传 RAG 检索优化完整报告

> 整理时间：2026-08-20
> 目标：用「结构化标签 + 检索增强 + boost 重排」提升白蛇传多版本语料的检索 core recall

---

## 一、项目背景与问题

- 语料：427 个 chunk，7 个文学版本（文言小说/话本/拟话本/戏曲/章回体/弹词/剧本）+ 研究文献，跨朝代，情节高度相似（同一母题「白蛇传」的多版本复述）。
- 核心矛盾：**白话 query ↔ 文言 chunk** 语义对不上、词面不重叠，且「跨版本 rel=1」标注大量存在，导致原始 recall@10 被结构性压低。
- 基线（旧生产，2026-08-17）：core recall@10（rel=2）= 0.537。

---

## 二、环境问题排查与修复（重要，可复用）

| # | 问题 | 根因 | 解决 |
|---|---|---|---|
| 1 | ChromaDB 检索 segfault | chromadb 0.6.3 的 hnswlib 在 Windows 上 add/query 都崩 | 降级 `chromadb==0.5.3` + `chroma-hnswlib==0.7.3`（troubleshooting 文档记录的锁定版本） |
| 2 | onnxruntime 批量 embed 内存泄漏（7GB+ → bad allocation） | onnxruntime 1.19.2 的 CPU 内存 arena 不释放 | 禁用 arena 无效（极慢）；降 1.15.1 又读不了 IR v10 模型 → **最终改用 torch + safetensors 做 embedding** |
| 3 | torch 加载被拒 | transformers 4.57.6 因 CVE-2025-32434 拒绝 torch.load（要求 torch≥2.6） | 用 `use_safetensors=True` 从本地 `model.safetensors` 加载 |
| 4 | ChromaDB add 报 None 值错 | chromadb 0.5.3 拒绝 metadata 的 None 值 | `vector_store.py` 加 `_clean_metadata`（None→""，复杂类型→str） |
| 5 | deepseek-v4-flash 打标返回空 | 它是推理模型，复杂文本 reasoning 失控（2048 token 全被 reasoning 占用） | 打标 `max_tokens=8192`；归并改用非推理模型 `deepseek-chat` |
| 6 | JSON 打标 parse_failed 高达 13% | 大模型严格 JSON 语法不稳（耶鲁团队：JSON 70% vs YAML 94-96%） | **打标输出改用 YAML**，parse_failed 降到 0.5% |

**关键结论**：本项目 embedding 用 **torch + safetensors**（绕开 onnxruntime 所有坑），打标/归并用 **DeepSeek**（flash 打标 + chat 归并），输出 **YAML**。

---

## 三、标签体系

### 3.1 双轨打标（按 chunk 的 genre 分两类 prompt）

**文学文本 chunk**（genre 从 metadata 取，不打标）：
```yaml
persons: [原文人物称呼]        # 原始值字符串列表
locations: [原文地名]          # 原始值字符串列表
plot_unit: [主干情节名]        # 原始值字符串列表
plot_detail: 情节细节一句话     # 字符串
summary: 一句话摘要            # 字符串
```

**研究文献 chunk**：
```yaml
涉及版本: [genre代码]
分析主题: [分析主题]           # 原始值字符串列表
引用实体: {persons: [], locations: []}
引用情节: [情节名]
summary: 一句话摘要
```

### 3.2 合并 + 词表（先开放提取、后归并）

- persons/locations/plot_unit/分析主题 全局汇总后交 LLM 归并：合并后的标准名 → `canonical`，原始叫法 → `raw`；
- 词表四类：persons / locations / plot_units / topics，内容 = canonical ∪ raw；
- 「待删除」词条（泛称/单字噪音）过滤。

### 3.3 query 打标

query 标签结构：`query_type`（情节题/人物题/地点题/comparison题/研究分析题）、`persons`/`locations`/`plot_unit`/`分析主题`（双名 `[{"canonical","raw"}]`，从词表选）、`plot_detail`、`对比维度`（朝代/文本/体裁/作者/称谓）、`summary`、`genre`。

---

## 四、检索方案（召回 + 重排）

### 4.1 五路召回（捞进候选池）

| # | 路 | 做法 | 规模 |
|---|---|---|---|
| 1 | dense（标签增强） | chunk「标签词+原文」embed，query 原文 embed，余弦 | top-100 |
| 2 | sparse（标签增强） | BM25 索引「原文+标签词」，query「原文+标签词」 | top-100 |
| 3 | 标签语义 | chunk 标签文本 embed（去高频实体），query 标签文本 embed，余弦 | top-100 |
| 4 | plot_unit 精确 | query 的 plot_unit canonical 用 BM25 精确检索 | top-50 |
| 5 | 研究文献专用 | comparison/研究分析题时，分析主题+plot_unit 检索研究文献 | top-50 |

候选池 = 五路 union 去重。

### 4.2 boost 重排（候选池 → top-10）

```
fused_base = 0.4×minmax(dense) + 0.4×minmax(bm25) + 0.2×minmax(标签语义)
final = (fused_base + boost) × genre因子；hard_match 加 HARD_OFFSET(10.0)
```

**文学文本 chunk**：
- 情节题：plot_unit 语义匹配（精确 或 embedding≥0.75）→ 硬分组
- 情节题：plot_detail char bigram Jaccard × 0.20
- 人物/地点题：persons/locations 高 IDF(≥0.8) → 硬分组
- 情节/人物/地点/comparison：persons × 0.10×IDF、locations × 0.09×IDF
- comparison+称谓：persons raw × 0.05

**研究文献 chunk**：
- comparison/研究分析：分析主题、引用情节 语义匹配 → 硬分组
- 情节题：引用情节 × 0.05 弱线性
- 引用实体 persons/locations × 0.10/0.09×IDF
- comparison+版本维度：涉及版本 × 0.05

**通用**：
- comparison+版本维度：版本元信息 × 0.05
- summary 双路：dense 语义 cos(query.summary, chunk.summary) × 0.10 + sparse 词面 Jaccard × 0.10
- genre 因子：chunk.genre == query.genre → ×1.2

### 4.3 关键参数（已锁定）

| 参数 | 值 |
|---|---|
| 融合权重 dense/sparse/tag | 0.4 / 0.4 / 0.2 |
| plot_unit 语义阈值 | 0.75 |
| persons/locations base | 0.10 / 0.09 |
| IDF 阈值 | 0.8 |
| HARD_OFFSET | 10.0 |

---

## 五、迭代过程与效果（每个尝试）

### 5.1 boost 规则迭代（抽样 query 上的 core recall）

| 轮次 | 改动 | core recall delta |
|---|---|---|
| 初版 | 原始 boost 初值 | +0.0150 |
| 2 | plot_unit 硬分组 + plot_detail 加大(0.20) | +0.0317 |
| 3 | canonical 偏置统一提权(0.10/0.06) | -0.0083（恶化，回退） |
| 4 | IDF 加权 canonical 偏置 | +0.0317 |
| 5 | 扩大样本 10→60 query | +0.0450 |
| 6 | 版本元信息硬前置 | 无效（回退） |
| 7 | 研究文献硬分组（全触发） | +0.0375（情节题误伤） |
| 8 | 收窄引用情节硬分组（仅 comparison/研究分析） | +0.0486 ✅ |
| 9 | 参数扫描（融合权重/阈值/boost权重） | 确认当前最优 |

**教训**：
- canonical 统一提权失败 → 高频实体（白素贞）区分度低，要 IDF 加权；
- 研究文献引用情节硬分组在情节题误伤 → 情节题答案在文学文本，研究文献不硬分组；
- 版本元信息硬前置无效 → comparison 题核心 62% 是研究文献，版本匹配「药不对症」。

### 5.2 检索增强（unretrievable 核心捞回）

| 尝试 | unretrievable 候选池捞回 |
|---|---|
| 原始 BM25 top-50 | 2.0% |
| sparse 标签增强 top-50 | 11.8% |
| sparse 标签增强 top-100 | 30.3% |
| dense100 + sparse增强100 + 标签语义100 | 39.5% |
| 五路召回（含 plot_unit 精确 + 研究文献专用） | 49.3% |

### 5.3 深入分析与方法探索

1. **分析 core recall 天花板**：sortable 核心 best_rank 中位数 24，62% 的 fused_base 排名 > 30 → 瓶颈在「检索排序」而非「重排」；
2. **搜索相关研究**：SitEmb（situated embeddings 让短 chunk 感知情节上下文）、BGE 三国演义（事件标签/别名归一化/硬负例）、IMKG（知识图谱符号匹配）、ColBERT（多向量匹配）；
3. **方法 B：plot_unit 语义匹配**（embedding 相似度 0.75，解决「断桥借伞 vs 游湖借伞」归并不一致）→ sortable +1.8%；
4. **标签语义 + dense 融合**（dense/sparse/tag 三路融合）→ sortable 0.3856；
5. **标签增强 dense**（标签词拼 chunk 开头，缓解稀释）→ unretr +2.25%、sortable +0.47% ✅ 最有效；
6. **情节层级化**（plot_unit embedding 聚类分主干/子情节）→ 短词 embedding 聚类效果差（阈值难调），放弃；
7. **query 侧标签增强**（query 拼标签词 embed）→ 未做，是下一步候选方向。

---

## 六、最终成绩（全量 1339 query）

| 指标 | academic_deepseek | academic_qwen | daily_deepseek | daily_qwen | 整体 |
|---|---|---|---|---|---|
| recall@5 (rel≥1) | 0.3477 | 0.2344 | 0.3830 | 0.3030 | 0.3262 |
| recall@10 (rel≥1) | 0.4786 | 0.3679 | 0.5190 | 0.4194 | 0.4550 |
| recall@20 (rel≥1) | 0.6134 | 0.4753 | 0.6272 | 0.5210 | 0.5686 |
| **core recall@10 (rel=2)** | 0.6267 | 0.6666 | 0.6636 | 0.6564 | **0.6518** |
| mrr@10 | 0.8304 | 0.7453 | 0.8243 | 0.8504 | 0.8201 |
| hit@1 | 0.7473 | 0.6188 | 0.7258 | 0.7688 | 0.7259 |
| hit@10 | 0.9758 | 0.9865 | 0.9892 | 0.9866 | 0.9843 |
| precision@10 | 0.3508 | 0.3704 | 0.3427 | 0.3352 | 0.3475 |
| ndcg@10 | 0.5372 | 0.4725 | 0.5443 | 0.5167 | 0.5227 |

**核心提升**：core recall@10 从旧基线 0.537 → **0.6518**（+11.5 个百分点，src_only 干净 qrels 口径）。

**四类对比**：academic_qwen 是最短板（recall@10 仅 0.3679、ndcg 仅 0.4725，qwen 学术题更抽象）；daily_deepseek 最好；core recall 四类都接近（0.63~0.67）。

**关键洞察**：core recall（0.65）远高于 recall@10（0.46），说明「找对核心答案」比「找全跨版本相关（rel=1）」容易——recall@10 被跨版本串结构性压低，core recall 才是真实反映「找到答案」的指标。

---

## 七、可复用资产清单

### 7.1 数据资产（`rag_eval/results/`）

| 文件 | 内容 |
|---|---|
| `chunk_labels_full.json` | 427 chunk 的标签（原始值） |
| `merged_vocab_full.json` | 四类词表（persons/locations/plot_units/topics，canonical/raw） |
| `query_labels_full.json` | 1339 query 的标签 |
| `emb_cache/*.npz` | 向量缓存（tag_vectors / summary_vectors / plot_unit_vectors / tag_aug_dense / q_*_vecs） |
| `plot_hierarchy_cluster.txt/.json` | plot_unit embedding 聚类结果（阈值 0.68） |
| `picked_sortable_queries.json` / `picked_unretrievable_queries.json` | 抽样 query |

### 7.2 生产模块（`agent/server/rag/`）

| 文件 | 作用 |
|---|---|
| `tag_store.py` | 标签加载 + 归一化 + IDF + 各类向量库（含磁盘缓存） |
| `boost.py` | boost 规则（硬分组 + IDF 加权 + summary 双路 + genre 因子） |
| `query_tagger.py` | query 在线打标（YAML，从词表选 canonical） |
| `tag_retriever.py` | 五路召回 + 候选池 + boost 重排 |
| `vector_store.py` | 加了 `_clean_metadata`（metadata None 清理） |

### 7.3 文档与脚本

| 文件 | 作用 |
|---|---|
| `rag_eval/RETRIEVAL_RULES.md` | 检索规则（实现唯一依据） |
| `rag_eval/白蛇传RAG检索优化完整报告.md` | 本报告 |
| `rag_eval/select_sortable_queries.py` / `select_unretrievable_queries.py` | 抽样 |
| `rag_eval/label_chunks.py` / `label_all_chunks.py` / `retry_*` | 打标 + 重试 |
| `rag_eval/merge_vocab2.py` / `merge_vocab_full.py` | 词表归并 |
| `rag_eval/label_queries.py` / `label_all_queries.py` | query 打标 |
| `rag_eval/full_eval.py` | 全量多指标评估 |
| `rag_eval/tag_augment_dense_verify.py` | 标签增强 dense 验证 |
| 各 `_diag_*` / `analyze_*` / `*_sweep_*` | 诊断/分析/参数扫描脚本 |

---

## 八、经验教训与后续方向

### 8.1 经验教训

1. **环境坑要先扫 troubleshooting 文档**：chromadb 版本、onnxruntime arena、torch CVE 都是已记录的坑；
2. **推理模型（deepseek-v4-flash）不适合复杂打标/归并**：reasoning 失控，归并用非推理模型；
3. **YAML 输出比 JSON 稳**：结构化抽取优先 YAML；
4. **短词 embedding 聚类不可靠**：plot_unit 是 2~4 字短词，embedding 相似度区分度不够，固定阈值难同时「聚同义 + 分异义」；
5. **高频实体（白素贞）区分度低**：canonical 匹配要 IDF 加权，不能统一提权；
6. **情节题答案在文学文本，comparison/研究分析题答案可能在研究文献**：硬分组要按 query_type 分流，否则误伤；
7. **重排（boost）救不了「检索排序太靠后」的核心**：核心 fused_base 排名 > 30 的，靠线性 boost 压不过，要靠「检索阶段」的标签增强（让核心 dense/sparse 排名提前）。
