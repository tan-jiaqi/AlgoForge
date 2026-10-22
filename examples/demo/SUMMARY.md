# 演示结果摘要

由 `python -m factory.cli demo` 生成。三个任务依次在同一份知识图谱副本上运行。

| 运行 | 状态 | 最优方案 | 主指标 | 生成算子 | 修复轮数 | 注册新算子 | 复用已沉淀算子 | 新增经验 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [01_tickets_transfer](runs/01_tickets_transfer/report.md) | succeeded | char_ngram_tfidf_lr_tuned | f1_macro = 0.820 | 1 | 1 | - | - | 1 |
| [02_churn_synthesize](runs/02_churn_synthesize/report.md) | succeeded | lr_interpretable_pipeline | recall = 0.778 | 4 | 0 | TotalChargesCleaner@2、TenureBucketizer@2 | - | 0 |
| [03_churn_reuse](runs/03_churn_reuse/report.md) | succeeded | interpretable_logistic_regression | recall = 0.778 | 2 | 0 | IdColumnDropper@2 | TenureBucketizer@2、TotalChargesCleaner@2 | 0 |

知识图谱的变化：Lesson 11 → 12；Metric 4 → 5；Operator 14 → 17；Scenario 1 → 2；ValidationRun 5 → 12。

- 知识图谱：`kg/capability_kg.json`（可视化 `kg/capability_kg.html`）
- 沉淀的生成算子源码：`kg/operators/`
