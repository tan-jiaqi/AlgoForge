# 真实 LLM 机制验收

每组三次独立进程运行，固定 data/churn.csv、TaskSpec 和 5 折 × 3 种子验证；失败和未触发分支全部保留，注入与自然缺陷分别统计。

| 运行 | 类型 | 状态 | 修复 | recall | 调用数 | 远端调用 | tokens | 耗时(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kg_full_1 | kg | failed | 0 | 0.603 | 3 | 3 | 15313 | 53.6 |
| kg_no_lessons_1 | kg | failed | 0 | 0.603 | 3 | 3 | 13313 | 19.3 |
| kg_catalog_1 | kg | failed | 1 | 0.4949 | 4 | 4 | 10763 | 38.6 |
| kg_full_2 | kg | failed | 0 | 0.603 | 3 | 3 | 15313 | 20.8 |
| kg_no_lessons_2 | kg | failed | 0 | 0.603 | 3 | 3 | 13313 | 19.5 |
| kg_catalog_2 | kg | failed | 1 | 0.5969 | 4 | 4 | 11822 | 33.4 |
| kg_full_3 | kg | failed | 0 | 0.603 | 3 | 3 | 15313 | 25.5 |
| kg_no_lessons_3 | kg | failed | 0 | 0.603 | 3 | 3 | 13313 | 18.9 |
| kg_catalog_3 | kg | failed | 1 | 0.4949 | 4 | 4 | 10763 | 42.1 |
| probe_1 | probe | failed | 1 | 0.6775 | 7 | 7 | 25316 | 106.6 |
| probe_3 | probe | failed | 1 | 0.6789 | 7 | 7 | 25898 | 104.8 |
| repair_natural_1 | repair_natural | failed | 1 | 0.6775 | 7 | 7 | 25316 | 56.1 |
| repair_natural_2 | repair_natural | failed | 1 | 0.6775 | 7 | 7 | 25316 | 80.3 |
| repair_natural_3 | repair_natural | failed | 1 | 0.6775 | 7 | 7 | 25316 | 92.7 |
| search_1 | search | succeeded | 0 | 0.8475 | 6 | 6 | 45149 | 45.4 |
| search_2 | search | succeeded | 0 | 0.8475 | 6 | 6 | 44811 | 45.2 |
| probe_2 | probe | succeeded | 0 | 0.7785 | 3 | 3 | 27947 | 45.4 |
| repair_injected_1 | repair_injected | succeeded | 1 | 0.7785 | 2 | 2 | 4530 | 56.9 |
| repair_injected_2 | repair_injected | succeeded | 1 | 0.7785 | 2 | 2 | 4590 | 44.2 |
| repair_injected_3 | repair_injected | succeeded | 1 | 0.7785 | 2 | 2 | 4727 | 15.4 |
| repair_injected_1_reuse | reuse | succeeded | 0 | 0.778 | 4 | 4 | 32445 | 63.6 |
| repair_injected_2_reuse | reuse | succeeded | 0 | 0.7785 | 2 | 2 | 26080 | 43.9 |
| repair_injected_3_reuse | reuse | succeeded | 1 | 0.7793 | 6 | 6 | 24074 | 92.1 |
| search_3 | search | failed | 1 | 0.6243 | 12 | 12 | 84448 | 115.4 |

完整证据：manifest.json、results.json；每个运行目录含 execution.log、runs/<运行名>/state.json、报告和源码。注入运行另外保留 injection.json；搜索运行保留 intervention.json。

回放重做训练、验证和知识回写，只有 LLM 响应来自录制。

full：完整方案通过 0/3；触发修复 0/3；修复算子通过且注册 0/3。

no_lessons：完整方案通过 0/3；触发修复 0/3；修复算子通过且注册 0/3。

catalog：完整方案通过 0/3；触发修复 3/3；修复算子通过且注册 0/3。

probe：完整方案通过 1/3；触发修复 2/3；修复算子通过且注册 0/3。

repair_natural：完整方案通过 0/3；触发修复 3/3；修复算子通过且注册 0/3。

search：完整方案通过 2/3；触发修复 1/3；修复算子通过且注册 0/3。

repair_injected：完整方案通过 3/3；触发修复 3/3；修复算子通过且注册 3/3。

reuse：完整方案通过 3/3；触发修复 1/3；修复算子通过且注册 0/3。

## KG 配对观察

| 重复 | 去掉 Lesson 后最优候选结构/参数变化 | 仅目录时最优候选结构/参数变化 |
| --- | --- | --- |
| 1 | True | True |
| 2 | True | True |
| 3 | True | True |

比较时忽略方案名称、理由和证据引用。

## 固定搜索子方案池的历史证据干预

| 运行 | 轮次 | 有历史时入选 | 无历史时入选 | 选择变化 |
| --- | --- | --- | --- | --- |
| search_1 | 1 | lgbm_with_feature_engineering, logreg_interpretable_tuned | logreg_interpretable_tuned, rf_recall_optimized | True |
| search_2 | 1 | lgbm_basic_balanced, interpretable_logreg_pipeline | rf_recall_optimized, interpretable_logreg_pipeline | True |
| search_3 | 1 | lgbm_with_tenure_bucket, lgbm_recall_optimized | rf_recall_focused, interpretable_logreg_pipeline | True |
| search_3 | 2 | logistic_interpretable_weighted, rf_feature_ensemble_recall | rf_feature_ensemble_recall, logistic_interpretable_weighted | False |

固定同一组子方案、本次观测、阈值和 beam 宽度，只去掉 KG 中的历史验证记录，比较入选集合。
