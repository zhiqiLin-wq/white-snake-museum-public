# EvolutionWorkbench v8 状态机文档

## 概述

EvolutionWorkbench 是一个 12 节点、3 个中断点的 LangGraph StateGraph，实现文脉演化分析的人机协同流程。

## 状态转换图

```
START
  │
  v
[E1: decompose_motifs]    追踪目标 -> 3-7 个叙事母题
  │
  v
  🛑 Interrupt #1: confirm_motifs   用户确认/编辑母题
  │
  v
[E2: scan_motif_matrix]   母题 x 朝代 -> 矩阵单元格
  │
  v
[E3: classify_changes]    矩阵 -> 7 种变化类型分类
  │
  v
[E4: infer_phylogeny]     变化模式 -> 谱系网络 (nodes + edges)
  │
  v
[E5: correlate_external]  变化 + 外部历史 -> 关联分析
  │
  v
[E6: formulate_hypothesis] 所有分析 -> 2-4 条候选假设
  │
  v
  🛑 Interrupt #2: select_hypothesis  用户选择假设
  │
  v
[E7: test_claims]          假设 -> 主张验证 -> verdict
  │
  ├── all supported -> [E8: synthesize]
  ├── refuted >= 50% -> [E2: scan_motif_matrix] (re_explore)
  └── insufficient > 0 -> [decide_next_action]
                            │
                            v
                            🛑 Interrupt #3: decide_next_action
                            │
                            ├── investigate -> [reflect_and_re_search] -> [E7]
                            ├── reexplore -> [E2]
                            ├── accept -> [E8]
                            └── change_hypothesis -> [E6 select_hypothesis]
  v
[E8: synthesize]           综合所有分析 -> 4 个 SSE 事件
  │
  v
[present]                  刷新 pending_events -> END
```

## 12 个节点

| 节点 | 阶段 | 输入 | 输出 |
|------|------|------|------|
| decompose_motifs | E1 | tracking_target | motifs[], suggested_additions |
| confirm_motifs | Int#1 | (Command resume) | confirmed motifs |
| scan_motif_matrix | E2 | motifs[] | motif_matrix[] |
| classify_changes | E3 | motif_matrix[] | changes[] |
| infer_phylogeny | E4 | motif_matrix[], changes[] | phylogeny |
| correlate_external | E5 | changes[] | correlations[] |
| formulate_hypothesis | E6 | 所有分析结果 | candidate_hypotheses[] |
| select_hypothesis | Int#2 | (Command resume) | selected_hypothesis_ids[] |
| test_claims | E7 | hypotheses[] | claims[] (with verdicts) |
| decide_next_action | Int#3 | claims[] | decision_options[] |
| reflect_and_re_search | E7b | claims[] | 补充证据 |
| synthesize | E8 | 所有结果 | evolution_narrative |

## 3 个中断点

1. **confirm_motifs** (E1->E2): 用户确认/编辑 AI 分解的母题列表
2. **select_hypothesis** (E6->E7): 用户从候选假设中选择要验证的
3. **decide_next_action** (E7 后): 用户决定下一步方向 (investigate/reexplore/accept/change)

## 条件路由

### _route_after_test (E7 后)
- all supported -> synthesize
- refuted >= 50% -> re_explore (回到 E2)
- insufficient > 0 && iteration < 3 -> decide (进入中断 #3)
- else -> synthesize

### _route_after_decision (中断 #3 后)
- investigate -> reflect_and_re_search -> test_claims
- reexplore -> scan_motif_matrix
- accept -> synthesize
- change_hypothesis -> select_hypothesis

## 开发者上手

1. 启动 Python Agent: `cd agent && python -m server.main`
2. 启动 Node 代理: `cd server && npm run dev`
3. POST /chat 发送演化查询: `{"query": "追踪雷峰塔的叙事演变", "stream": true}`
4. 接收 SSE 流 -> 遇到 interrupt -> POST /chat/resume -> 继续
