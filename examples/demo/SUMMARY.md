# 演示结果摘要

由 `python -m factory.cli demo` 生成。三个任务依次在同一份知识图谱副本上运行。

| 运行 | 状态 | 最优方案 | 主指标 | 生成算子 | 修复轮数 | 注册新算子 | 复用已沉淀算子 | 新增经验 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [01_tickets_transfer](runs/01_tickets_transfer/report.md) | succeeded | tfidf_logistic_baseline | f1_macro = 0.824 | 1 | 0 | - | - | 0 |
| [02_churn_synthesize](runs/02_churn_synthesize/report.md) | succeeded | logistic_regression_pipeline | recall = 0.778 | 4 | 3 | TotalChargesCleaner@2、TenureBucketizer@2 | - | 1 |
| [03_churn_reuse](runs/03_churn_reuse/report.md) | succeeded | logistic_regression_pipeline | recall = 0.777 | 1 | 0 | IdColumnDropper@2 | TenureBucketizer@2、TotalChargesCleaner@2 | 0 |

知识图谱的变化：Condition 19 → 23；Dependency 4 → 5；Lesson 10 → 11；Metric 3 → 4；Operator 14 → 17；Scenario 1 → 2；ValidationRun 5 → 12。

- 知识图谱：`kg/capability_kg.json`（可视化：概览 `kg/capability_kg.html`，完整 `kg/capability_kg_full.html`）
- 沉淀的生成算子源码：`kg/operators/`
