# 结构化标签 + boost 召回提升方案（设计文档）

> 目标：验证「结构化标签 + boost」能否提升 core recall（把丢失的核心答案捞回 top-10）。
> 现状：core recall@10 = 0.537（原始 recall@10 = 0.476 被 rel=1 跨版本串灌水）。
> 更新：2026-08-18

---

## 一、核心结论（先记住，别跑偏）

1. **boost 是「排序增强」，不是「检索替换」**：只能救 dense/sparse 已找到但排名不够靠前的核心（fusion_buried 14% + sortable 40.6%），救不了 unretrievable（45.4%，dense/sparse 分数≈0，boost 加 0.2~0.3 压不过 top-10 的 0.5~0.6）。
2. **验证对象必须是 sortable 的 query**，不是 unretrievable（否则结论必然是"boost 没用"，是误导）。
3. 融合（fusion）没提升空间了：4 种融合策略（minmax/median/rrf/consensus_bonus）全不如当前 minmax_zero。
4. cross-encoder 重排放弃：reranker 在「近重复多版本 + 文言文」场景结构性失败。

---

## 二、标签体系（chunk 双轨，按信息形态分两类）

### 2.1 文学文本 chunk（7 个版本，是"故事叙述"）

```json
{
  "genre": "opera_libretto",
  "persons": [{"canonical": "许仙", "raw": "汉文"}],
  "locations": [{"canonical": "杭州", "raw": "临安"}],
  "plot_unit": "盗仙草",
  "plot_detail": "白娘子昆仑盗灵芝救许仙",
  "summary": "白娘子为救许仙去昆仑盗灵芝"
}
```

### 2.2 研究文献 chunk（是"学术分析"，会引用各版本、做对比）

```json
{
  "涉及版本": ["huaben", "imitation_huaben"],
  "分析主题": "景观互构",
  "引用实体": {"persons": ["奚宣赞", "奚真人"], "locations": ["西湖三塔"]},
  "引用情节": ["三塔镇三怪"],
  "summary": "论述三塔镇三怪叙事及三塔毁后镇妖功能转移至雷峰塔的景观移位"
}
```

> 研究文献的"情节"是「引用」不是「叙述」，字段语义与文学文本完全不同，必须隔离。

### 2.3 标签结构铁律

- persons / locations 用 `{"canonical": "标准名", "raw": "原文名"}` 双名结构；
- 词表用「标准名 + 别名列表」=「归并后 + 归并前」；
- `raw` 用于 comparison 题的称谓对比（"汉文 vs 许宣 vs 许仙"）。

---

## 三、query 解析（分类型 + 对比维度具体值）

```json
{
  "query_type": "情节题|人物题|地点题|comparison题|研究分析题|其他",
  "need_examples": true,
  "need_analysis": false,
  "persons": [{"canonical": "许仙", "raw": "汉文"}],
  "locations": [{"canonical": "杭州", "raw": "临安"}],
  "plot_unit": "盗仙草",
  "plot_detail": "白娘子昆仑盗灵芝",
  "分析主题": "景观互构",
  "对比维度": {"朝代": "清", "体裁": "话本", "作者": "冯梦龙", "称谓": ["汉文", "许宣"]},
  "summary": "问题问的是什么事"
}
```

> 对比维度是 `{维度名: 具体值}` 对象，不是维度名列表——版本元信息匹配需要具体值。

---

## 四、受控词表（先开放提取、后归并）

| 词表 | 数量 | 文件 |
|---|---|---|
| plot_unit（情节单元） | 19 | merged_vocab.json |
| 分析主题 | 6 | merged_vocab.json |
| 地点 | 27 | merged_vocab.json |

### 版本映射表 GENRE_META（genre → 朝代/文本名/作者/体裁，映射零 LLM）

| genre | 朝代 | 文本名 | 作者 | 体裁 |
|---|---|---|---|---|
| classical_chinese | 唐 | 李黄 | 佚名 | 文言小说 |
| huaben | 宋 | 西湖三塔记 | 佚名 | 话本 |
| imitation_huaben | 明 | 白娘子永镇雷峰塔 | 冯梦龙 | 拟话本 |
| opera_libretto | 清 | 雷峰塔传奇 | 方成培 | 戏曲 |
| chapter_vernacular | 清 | 雷峰塔传奇 | 玉山堂主人 | 白话神魔小说 |
| tanci | 清 | 义妖传 | 陈遇乾 | 苏州弹词 |
| script | 现当代 | 白蛇传 | 田汉 | 京剧 |
| research_literature | None | 研究文献 | 研究团队 | 学术 |

---

## 五、boost 规则（完整版，按 query_type 分流）

```
fused_base = 0.5 × minmax(dense_score) + 0.5 × minmax(bm25_score)
boost = 0

# ── 文学文本 chunk 专属 ──
if C 是文学文本:
    if Q.plot_unit 且 C.plot_unit == Q.plot_unit:          boost += 0.10
    if overlap(Q.plot_detail, C.plot_detail):              boost += 0.08
    boost += 0.03 × |Q.persons[].canonical ∩ C.persons[].canonical|
    boost += 0.02 × |Q.locations[].canonical ∩ C.locations[].canonical|
    if "称谓" in Q.对比维度:
        boost += 0.05 × |Q.persons[].raw ∩ C.persons[].raw|

# ── 研究文献 chunk 专属 ──
if C 是研究文献:
    if Q.分析主题 且 C.分析主题 == Q.分析主题:             boost += 0.05
    if Q.对比维度含"文本/体裁/作者/朝代" 且 C.涉及版本 命中:  boost += 0.05
    if Q.plot_unit 且 Q.plot_unit in C.引用情节:          boost += 0.05
    boost += 0.03 × |Q.persons[].canonical ∩ C.引用实体.persons|
    boost += 0.02 × |Q.locations[].canonical ∩ C.引用实体.locations|

# ── 版本元信息（comparison 题，文学+研究都适用）──
if Q.query_type == comparison题 且 Q.对比维度含"朝代/文本/体裁/作者":
    if GENRE_META[C.genre] 的对应字段 == Q.对比维度的具体值:  boost += 0.05

# ── 摘要第二路语义（所有 chunk）──
boost += 0.05 × sim(Q, C.summary)

final = (fused_base + boost) × (1.2 if C.genre == Q.genre else 1.0)
```

### query_type 分流（决定哪些层对哪类 chunk 生效）

| query_type | 对文学文本启用 | 对研究文献启用 |
|---|---|---|
| 情节题 | plot_unit/plot_detail/persons/locations | 引用情节/引用实体 |
| 人物/地点题 | persons/locations | 引用实体 |
| comparison 题 | 版本元信息/称谓 raw/persons/locations | 涉及版本/分析主题/引用情节 |
| 研究分析题 | （基本不用） | 分析主题/引用情节/引用实体 |

---

## 六、验证方案

1. **抽 10 条 sortable query**（核心 chunk 的 best_rank 在 D/S top-11~50，不是 unretrievable）；
2. **跑检索**：D 路 top-50 + S 路 top-50，拿每个候选 chunk 的 dense_score / bm25_score；
3. **补打竞争 chunk 标签**：候选集（D∪S top-50）里非核心的 chunk，用双轨 prompt 打标；
4. **归一**：plot_unit/分析主题/地点 经 merged_vocab 映射到标准名；
5. **boost 重排序**：按上面公式算 final，降序取 top-10；
6. **三个硬指标**：
   - 核心捞回率：核心 chunk 从 top-11~50 进入 top-10 的比例；
   - core recall@10 变化：boost 前 vs 后（用 src_only 干净 qrels）；
   - 副作用：挤掉多少原 top-10 里真相关的 chunk。

**竞争 chunk 定义**：候选集（D∪S top-50）里除核心外的所有 chunk，不管 rel=1/rel=0/unlabeled。

---

## 七、关键结论与避坑（反复踩过的）

1. **query_id 跨 4 数据集重复**：脚本加载 query 文本必须按 dataset 分（qtext_by_ds / qtext_by_ds[ds]），否则后加载覆盖前加载，query 文本错配。
2. **不能用 LLM 打的一个标签去判定另一个 LLM 标注**（循环论证）——判定 rel 是否错、chunk 是否真相关，必须人工看原文。
3. **源 chunk 溯源**（description 里的 source_chunk_ids）只做客观定位：rel=2 核心里 A 类（group 成员 79.3%）/ B 类（批量标注补的 20.7%，qwen 最高 36.1%）。已生成 qrels_*_src_only.tsv（B 类 rel=2 降级 rel=1）。
4. **rel=2 污染在标注环节，不在 query 生成环节**：source_relevance 标注自相矛盾 + 批量标注判据宽（"与主题相关就标 1/2"），把同章邻居抬成 rel=2。
5. **归并流程**：所有词表（情节/主题/地点/人物）都是「先开放提取 → 后归并」，归并后不用重新打标，代码 dict 查表把原始值映射成标准名即可（省 token）。

---

## 八、产出文件清单

| 文件 | 作用 |
|---|---|
| relabel_v2.py | 双轨 chunk 打标 + query 分类型解析（含对比维度具体值） |
| merged_vocab.json | plot_unit 19 + 分析主题 6 + 地点 27 受控词表 |
| normalize_and_match.py | 代码归一 + 匹配度分析 |
| source_trace.py / build_source_only_qrels.py | rel=2 核心 A/B 溯源 + 生成干净 qrels |
| qrels_*_src_only.tsv | 排除 B 类的干净 qrels |
