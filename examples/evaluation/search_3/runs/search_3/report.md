# AlgoForge 验证报告

| 项目 | 内容 |
| --- | --- |
| 需求 | 预测客户流失，召回率不低于0.8；优先尝试可解释模型，未达标时探索不同模型或参数。 |
| 数据 | data/churn.csv |
| 运行 ID | search_3 |
| LLM | gemini-3.5-flash |
| 状态 | ❌ 未通过 |
| 最优方案 | lgbm_imbalanced_pipeline |
| 总耗时 | 115.4 s |

共验证 5 个候选方案，0 个通过；重新规划 2 次，修复 1 轮。最优方案 `lgbm_imbalanced_pipeline` 的 recall = 0.624（要求 ≥ 0.8）。

## 1. 需求理解（TaskSpec）

| 字段 | 值 |
| --- | --- |
| 任务类型 | binary_classification |
| 数据模态 | tabular |
| 目标列 | churn |
| 评价指标 | recall ≥ 0.8（主） |
| 业务约束 | interpretable、recall_first |
| 数据画像标签 | has_missing_values、numeric_features、has_categorical_features、low_cardinality、string_typed_numeric、id_columns、imbalanced、time_tenure、mixed_features |

## 2. 知识检索（设计依据）

匹配场景：telecom_churn。每类步骤从知识图谱检索候选算子，得分 = 0.4 × 文本相关度 + 0.3 × 条件匹配 + 0.3 × 历史验证分。

| 步骤 | 候选算子 | 可直接复用 | 得分 | 文本/条件/历史 | 证据 |
| --- | --- | --- | --- | --- | --- |
| imputer | MissingValueImputer@1 | 是 | 0.492 | 0.62 / 0.09 / 0.72 | 匹配条件：has_missing_values；历史验证 3 次，平均 recall 0.717 |
| imputer | TotalChargesCleaner@1 | 否（历史实现，可参考生成） | 0.482 | 1.00 / 0.27 / 0.00 | 匹配条件：has_missing_values、numeric_features、string_typed_numeric |
| feature | TenureBucketizer@1 | 否（历史实现，可参考生成） | 0.722 | 1.00 / 0.27 / 0.80 | 匹配条件：interpretable、numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | ChargeRatioFeatureGenerator@1 | 否（历史实现，可参考生成） | 0.616 | 0.80 / 0.18 / 0.80 | 匹配条件：numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | IdColumnDropper@1 | 否（历史实现，可参考生成） | 0.419 | 0.98 / 0.09 / 0.00 | 匹配条件：id_columns |
| feature | ServiceCountFeature@1 | 否（历史实现，可参考生成） | 0.415 | 0.97 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| encoder | OneHotCategoricalEncoder@1 | 是 | 0.669 | 1.00 / 0.18 / 0.72 | 匹配条件：has_categorical_features、low_cardinality；历史验证 3 次，平均 recall 0.717 |
| encoder | KFoldTargetEncoder@1 | 否（历史实现，可参考生成） | 0.366 | 0.85 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| scaler | StandardFeatureScaler@1 | 是 | 0.640 | 1.00 / 0.09 / 0.71 | 匹配条件：numeric_features；历史验证 1 次，平均 recall 0.710 |
| model | BalancedLightGBMClassifier@1 | 否（历史实现，可参考生成） | 0.661 | 1.00 / 0.09 / 0.78 | 匹配条件：imbalanced；历史验证 1 次，平均 recall 0.780 |
| model | LogisticRegressionModel@1 | 是 | 0.618 | 0.94 / 0.09 / 0.71 | 匹配条件：interpretable；历史验证 1 次，平均 recall 0.710 |
| model | LightGBMModel@1 | 是 | 0.549 | 0.70 / 0.09 / 0.80 | 匹配条件：imbalanced；历史验证 1 次，平均 recall 0.800 |
| model | RandomForestModel@1 | 是 | 0.541 | 0.79 / 0.09 / 0.66 | 匹配条件：mixed_features；历史验证 1 次，平均 recall 0.660 |

### 参考的历史经验

- **样本不平衡导致默认模型召回率偏低**（M_THRESHOLD）：启用平衡类权重（如 class_weight='balanced' 或 BalancedLightGBMClassifier）以提升召回率
- **逻辑回归标准化与可解释性**：必须先进行特征标准化（如 StandardFeatureScaler），在 0.1~10 之间搜索正则化强度 C，可直接输出系数给业务方。
- **随机森林防过拟合与召回优化**：设置 n_estimators 300 左右，限制 max_depth 在 8~16 防止过拟合；召回优先时需配合调整分类阈值。
- **数值列含空白字符导致类型转换失败**（E_RUNTIME）：先 str.strip()，再 pd.to_numeric(errors='coerce')，缺失值填补为 0
- **LightGBM 调参及不平衡处理**：设置 num_leaves 15~63, learning_rate 0.03~0.1, n_estimators 200~600；不平衡时设置 class_weight='balanced'；树模型无需标准化。

### 知识引用检查

检索并传入上下文不等于采纳。以下引用由模型声明，程序仅核对可见范围与结构化适用对象；不证明语义落实或指标收益。

| 方案 | 步骤 | 模型引用 | 范围检查 | 语义落实 |
| --- | --- | --- | --- | --- |
| lgbm_imbalanced_pipeline | 1 | Operator:TotalChargesCleaner@1 | 通过 | 未验证 |
| lgbm_imbalanced_pipeline | 2 | Operator:MissingValueImputer@1 | 通过 | 未验证 |
| lgbm_imbalanced_pipeline | 3 | Operator:IdColumnDropper@1 | 通过 | 未验证 |
| lgbm_imbalanced_pipeline | 4 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| lgbm_imbalanced_pipeline | 5 | Operator:LightGBMModel@1、ValidationRun:f6e4284c | 通过 | 未验证 |
| （被拒绝的规划） | - | 无 | 未通过：interpretable_logreg_pipeline: 数据含类别列，模型之前缺少 encoder 步骤（如 OneHotCategoricalEncoder）；interpretable_logreg_pipeline: 第 2 步：缺少 Operator:MissingValueImputer@1 证据引用；rf_recall_focused: 数据含类别列，模型之前缺少 encoder 步骤（如 OneHotCategoricalEncoder） | 未验证 |
| lgbm_recall_optimized | 1 | Operator:TotalChargesCleaner@1、Lesson:69d51358 | 通过 | 未验证 |
| lgbm_recall_optimized | 2 | Operator:MissingValueImputer@1 | 通过 | 未验证 |
| lgbm_recall_optimized | 3 | Operator:ChargeRatioFeatureGenerator@1、ValidationRun:f6e4284c | 通过 | 未验证 |
| lgbm_recall_optimized | 4 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| lgbm_recall_optimized | 5 | Operator:LightGBMModel@1、Lesson:9e9e77d6 | 通过 | 未验证 |
| lgbm_with_tenure_bucket | 1 | Operator:TotalChargesCleaner@1 | 通过 | 未验证 |
| lgbm_with_tenure_bucket | 2 | Operator:TenureBucketizer@1、ValidationRun:f6e4284c | 通过 | 未验证 |
| lgbm_with_tenure_bucket | 3 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| lgbm_with_tenure_bucket | 4 | Operator:LightGBMModel@1、Lesson:f2c1398b | 通过 | 未验证 |
| interpretable_logreg_pipeline | 1 | Operator:TotalChargesCleaner@1 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 2 | Operator:MissingValueImputer@1 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 3 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 4 | Operator:StandardFeatureScaler@1 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 5 | Operator:LogisticRegressionModel@1、Lesson:74e2da10 | 通过 | 未验证 |
| rf_recall_focused | 1 | Operator:TotalChargesCleaner@1 | 通过 | 未验证 |
| rf_recall_focused | 2 | Operator:ServiceCountFeature@1 | 通过 | 未验证 |
| rf_recall_focused | 3 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| rf_recall_focused | 4 | Operator:RandomForestModel@1、Lesson:57718714 | 通过 | 未验证 |
| （被拒绝的规划） | - | 无 | 未通过：logistic_interpretable_weighted: 第 2 步：缺少 Operator:OneHotCategoricalEncoder@1 证据引用 | 未验证 |
| lgbm_feature_rich_weighted | 1 | Operator:TotalChargesCleaner@1 | 通过 | 未验证 |
| lgbm_feature_rich_weighted | 2 | Operator:ChargeRatioFeatureGenerator@1 | 通过 | 未验证 |
| lgbm_feature_rich_weighted | 3 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| lgbm_feature_rich_weighted | 4 | Operator:LightGBMModel@1、Lesson:f2c1398b | 通过 | 未验证 |
| rf_feature_ensemble_recall | 1 | Operator:TotalChargesCleaner@1 | 通过 | 未验证 |
| rf_feature_ensemble_recall | 2 | Operator:ServiceCountFeature@1 | 通过 | 未验证 |
| rf_feature_ensemble_recall | 3 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| rf_feature_ensemble_recall | 4 | Operator:RandomForestModel@1、Lesson:57718714 | 通过 | 未验证 |
| lgbm_tenure_bucket_optimized | 1 | Operator:TotalChargesCleaner@1 | 通过 | 未验证 |
| lgbm_tenure_bucket_optimized | 2 | Operator:TenureBucketizer@1 | 通过 | 未验证 |
| lgbm_tenure_bucket_optimized | 3 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| lgbm_tenure_bucket_optimized | 4 | Operator:LightGBMModel@1、ValidationRun:f6e4284c | 通过 | 未验证 |
| logistic_interpretable_weighted | 1 | Operator:TotalChargesCleaner@1 | 通过 | 未验证 |
| logistic_interpretable_weighted | 2 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| logistic_interpretable_weighted | 3 | Operator:StandardFeatureScaler@1 | 通过 | 未验证 |
| logistic_interpretable_weighted | 4 | Operator:LogisticRegressionModel@1、Lesson:74e2da10 | 通过 | 未验证 |

## 3. 候选方案比较

排序规则：先看是否通过全部检查，再比主指标，再比次要指标之和，最后训练耗时短者优先。指标为多随机种子 × 分层 K 折交叉验证的均值 ± 种子间标准差。

| 排名 | 方案 | 结果 | recall | roc_auc | precision | f1 | recall_at_20 | 单次训练 (s) | 峰值内存 (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ★ lgbm_imbalanced_pipeline | 未通过 | 0.624 ± 0.007 | 0.783 ± 0.003 | 0.492 ± 0.002 | 0.550 ± 0.003 | 0.420 ± 0.006 | 0.287 | 265 |
| 2 | lgbm_with_tenure_bucket | 未通过 | 0.621 ± 0.009 | 0.781 ± 0.002 | 0.491 ± 0.003 | 0.548 ± 0.005 | 0.420 ± 0.003 | 0.293 | 260 |
| 3 | lgbm_recall_optimized | 未通过 | 0.597 ± 0.007 | 0.780 ± 0.002 | 0.500 ± 0.004 | 0.544 ± 0.005 | 0.417 ± 0.010 | 0.332 | 269 |
| 4 | rf_feature_ensemble_recall | 未通过 | 0.596 ± 0.002 | 0.801 ± 0.002 | 0.541 ± 0.003 | 0.567 ± 0.001 | 0.452 ± 0.002 | 1.476 | 299 |
| 5 | logistic_interpretable_weighted | 未通过 | - | - | - | - | - | - | 232 |

```
recall（| 为阈值 0.8）
lgbm_imbalanced_pipeline             ■■■■■■■■■■■■····|··· 0.624
lgbm_with_tenure_bucket              ■■■■■■■■■■■■····|··· 0.621
lgbm_recall_optimized                ■■■■■■■■■■■■····|··· 0.597
rf_feature_ensemble_recall           ■■■■■■■■■■■■····|··· 0.596
logistic_interpretable_weighted      ················|··· -
```

### 各方案的结构与诊断

- `lgbm_imbalanced_pipeline`：生成:TotalChargesCleaner → MissingValueImputer → 生成:IdColumnDropper → OneHotCategoricalEncoder → LightGBMModel。诊断：`M_THRESHOLD` recall = 0.6243 低于阈值 0.8
- `lgbm_with_tenure_bucket`：生成:TotalChargesCleaner → 生成:TenureBucketizer → OneHotCategoricalEncoder → LightGBMModel。诊断：`M_THRESHOLD` recall = 0.6205 低于阈值 0.8
- `lgbm_recall_optimized`：生成:TotalChargesCleaner → MissingValueImputer → 生成:ChargeRatioFeatureGenerator → OneHotCategoricalEncoder → LightGBMModel。诊断：`M_THRESHOLD` recall = 0.5969 低于阈值 0.8
- `rf_feature_ensemble_recall`：生成:TotalChargesCleaner → 生成:ServiceCountFeature → OneHotCategoricalEncoder → RandomForestModel。诊断：`M_THRESHOLD` recall = 0.5957 低于阈值 0.8
- `logistic_interpretable_weighted`：生成:TotalChargesCleaner → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel。诊断：`E_RUNTIME` ValueError: Input X contains NaN.
LogisticRegression does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to pr (at _assert_all_finite_element_wise:186)

### 方案搜索（Beam Search）

指标不达标时，以已验证结果排名前 k 的方案为父方案，由 Planner 扩展 k × m 个子方案；与已验证方案结构相同的直接去重，其余按先验（知识图谱中该场景的历史验证记录 + 本次运行的观测，平滑后按步骤加权，模型步骤权重更高）估计 recall，只保留前 k 个进入验证。

第 1 轮：父方案 `lgbm_imbalanced_pipeline`，beam 宽度 2。

| 子方案 | 结构 | 先验 recall | 实际 recall | 结果 |
| --- | --- | --- | --- | --- |
| ✔ lgbm_recall_optimized | 生成:TotalChargesCleaner → MissingValueImputer → 生成:ChargeRatioFeatureGenerator → OneHotCategoricalEncoder → LightGBMModel | 0.738 | 0.597 | 先验第 2，进入验证 |
| ✔ lgbm_with_tenure_bucket | 生成:TotalChargesCleaner → 生成:TenureBucketizer → OneHotCategoricalEncoder → LightGBMModel | 0.742 | 0.621 | 先验第 1，进入验证 |
| interpretable_logreg_pipeline | 生成:TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel | 0.737 | - | 先验第 3，超出 beam 宽度 2，剪枝 |
| rf_recall_focused | 生成:TotalChargesCleaner → 生成:ServiceCountFeature → OneHotCategoricalEncoder → RandomForestModel | 0.736 | - | 先验第 4，超出 beam 宽度 2，剪枝 |

| 子方案 | 步骤 | 先验证据来源 | 数值 |
| --- | --- | --- | --- |
| lgbm_recall_optimized | *TotalChargesCleaner | 本次运行观测 | 0.6243 |
| lgbm_recall_optimized | MissingValueImputer@1 | ValidationRun:e11ba947 | 0.71 |
| lgbm_recall_optimized | MissingValueImputer@1 | ValidationRun:e0d43fa3 | 0.66 |
| lgbm_recall_optimized | MissingValueImputer@1 | ValidationRun:a7f163bc | 0.78 |
| lgbm_recall_optimized | MissingValueImputer@1 | 本次运行观测 | 0.6243 |
| lgbm_recall_optimized | *ChargeRatioFeatureGenerator | ValidationRun:f6e4284c | 0.8 |
| lgbm_recall_optimized | OneHotCategoricalEncoder@1 | ValidationRun:e11ba947 | 0.71 |
| lgbm_recall_optimized | OneHotCategoricalEncoder@1 | ValidationRun:e0d43fa3 | 0.66 |
| lgbm_recall_optimized | OneHotCategoricalEncoder@1 | ValidationRun:a7f163bc | 0.78 |
| lgbm_recall_optimized | OneHotCategoricalEncoder@1 | 本次运行观测 | 0.6243 |
| lgbm_recall_optimized | LightGBMModel@1 | ValidationRun:f6e4284c | 0.8 |
| lgbm_recall_optimized | LightGBMModel@1 | 本次运行观测 | 0.6243 |
| lgbm_with_tenure_bucket | *TotalChargesCleaner | 本次运行观测 | 0.6243 |
| lgbm_with_tenure_bucket | *TenureBucketizer | ValidationRun:f6e4284c | 0.8 |
| lgbm_with_tenure_bucket | OneHotCategoricalEncoder@1 | ValidationRun:e11ba947 | 0.71 |
| lgbm_with_tenure_bucket | OneHotCategoricalEncoder@1 | ValidationRun:e0d43fa3 | 0.66 |
| lgbm_with_tenure_bucket | OneHotCategoricalEncoder@1 | ValidationRun:a7f163bc | 0.78 |
| lgbm_with_tenure_bucket | OneHotCategoricalEncoder@1 | 本次运行观测 | 0.6243 |
| lgbm_with_tenure_bucket | LightGBMModel@1 | ValidationRun:f6e4284c | 0.8 |
| lgbm_with_tenure_bucket | LightGBMModel@1 | 本次运行观测 | 0.6243 |
| interpretable_logreg_pipeline | *TotalChargesCleaner | 本次运行观测 | 0.6243 |
| interpretable_logreg_pipeline | MissingValueImputer@1 | ValidationRun:e11ba947 | 0.71 |
| interpretable_logreg_pipeline | MissingValueImputer@1 | ValidationRun:e0d43fa3 | 0.66 |
| interpretable_logreg_pipeline | MissingValueImputer@1 | ValidationRun:a7f163bc | 0.78 |
| interpretable_logreg_pipeline | MissingValueImputer@1 | 本次运行观测 | 0.6243 |
| interpretable_logreg_pipeline | OneHotCategoricalEncoder@1 | ValidationRun:e11ba947 | 0.71 |
| interpretable_logreg_pipeline | OneHotCategoricalEncoder@1 | ValidationRun:e0d43fa3 | 0.66 |
| interpretable_logreg_pipeline | OneHotCategoricalEncoder@1 | ValidationRun:a7f163bc | 0.78 |
| interpretable_logreg_pipeline | OneHotCategoricalEncoder@1 | 本次运行观测 | 0.6243 |
| interpretable_logreg_pipeline | StandardFeatureScaler@1 | ValidationRun:e11ba947 | 0.71 |
| interpretable_logreg_pipeline | LogisticRegressionModel@1 | ValidationRun:e11ba947 | 0.71 |
| rf_recall_focused | *TotalChargesCleaner | 本次运行观测 | 0.6243 |
| rf_recall_focused | OneHotCategoricalEncoder@1 | ValidationRun:e11ba947 | 0.71 |
| rf_recall_focused | OneHotCategoricalEncoder@1 | ValidationRun:e0d43fa3 | 0.66 |
| rf_recall_focused | OneHotCategoricalEncoder@1 | ValidationRun:a7f163bc | 0.78 |
| rf_recall_focused | OneHotCategoricalEncoder@1 | 本次运行观测 | 0.6243 |
| rf_recall_focused | RandomForestModel@1 | ValidationRun:e0d43fa3 | 0.66 |

第 2 轮：父方案 `lgbm_imbalanced_pipeline`, `lgbm_with_tenure_bucket`，beam 宽度 2。

| 子方案 | 结构 | 先验 recall | 实际 recall | 结果 |
| --- | --- | --- | --- | --- |
| lgbm_feature_rich_weighted | 生成:TotalChargesCleaner → 生成:ChargeRatioFeatureGenerator → OneHotCategoricalEncoder → LightGBMModel | 0.690 | - | 先验第 4，超出 beam 宽度 2，剪枝 |
| ✔ rf_feature_ensemble_recall | 生成:TotalChargesCleaner → 生成:ServiceCountFeature → OneHotCategoricalEncoder → RandomForestModel | 0.723 | 0.596 | 先验第 2，进入验证 |
| lgbm_tenure_bucket_optimized | 生成:TotalChargesCleaner → 生成:TenureBucketizer → OneHotCategoricalEncoder → LightGBMModel | 0.692 | - | 先验第 3，超出 beam 宽度 2，剪枝 |
| ✔ logistic_interpretable_weighted | 生成:TotalChargesCleaner → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel | 0.728 | - | 先验第 1，进入验证 |

| 子方案 | 步骤 | 先验证据来源 | 数值 |
| --- | --- | --- | --- |
| lgbm_feature_rich_weighted | *TotalChargesCleaner | 本次运行观测 | 0.6243 |
| lgbm_feature_rich_weighted | *TotalChargesCleaner | 本次运行观测 | 0.6205 |
| lgbm_feature_rich_weighted | *TotalChargesCleaner | 本次运行观测 | 0.5969 |
| lgbm_feature_rich_weighted | *ChargeRatioFeatureGenerator | ValidationRun:f6e4284c | 0.8 |
| lgbm_feature_rich_weighted | *ChargeRatioFeatureGenerator | 本次运行观测 | 0.5969 |
| lgbm_feature_rich_weighted | OneHotCategoricalEncoder@1 | ValidationRun:e11ba947 | 0.71 |
| lgbm_feature_rich_weighted | OneHotCategoricalEncoder@1 | ValidationRun:e0d43fa3 | 0.66 |
| lgbm_feature_rich_weighted | OneHotCategoricalEncoder@1 | ValidationRun:a7f163bc | 0.78 |
| lgbm_feature_rich_weighted | OneHotCategoricalEncoder@1 | 本次运行观测 | 0.6243 |
| lgbm_feature_rich_weighted | OneHotCategoricalEncoder@1 | 本次运行观测 | 0.6205 |
| lgbm_feature_rich_weighted | OneHotCategoricalEncoder@1 | 本次运行观测 | 0.5969 |
| lgbm_feature_rich_weighted | LightGBMModel@1 | ValidationRun:f6e4284c | 0.8 |
| lgbm_feature_rich_weighted | LightGBMModel@1 | 本次运行观测 | 0.6243 |
| lgbm_feature_rich_weighted | LightGBMModel@1 | 本次运行观测 | 0.6205 |
| lgbm_feature_rich_weighted | LightGBMModel@1 | 本次运行观测 | 0.5969 |
| rf_feature_ensemble_recall | *TotalChargesCleaner | 本次运行观测 | 0.6243 |
| rf_feature_ensemble_recall | *TotalChargesCleaner | 本次运行观测 | 0.6205 |
| rf_feature_ensemble_recall | *TotalChargesCleaner | 本次运行观测 | 0.5969 |
| rf_feature_ensemble_recall | OneHotCategoricalEncoder@1 | ValidationRun:e11ba947 | 0.71 |
| rf_feature_ensemble_recall | OneHotCategoricalEncoder@1 | ValidationRun:e0d43fa3 | 0.66 |
| rf_feature_ensemble_recall | OneHotCategoricalEncoder@1 | ValidationRun:a7f163bc | 0.78 |
| rf_feature_ensemble_recall | OneHotCategoricalEncoder@1 | 本次运行观测 | 0.6243 |
| rf_feature_ensemble_recall | OneHotCategoricalEncoder@1 | 本次运行观测 | 0.6205 |
| rf_feature_ensemble_recall | OneHotCategoricalEncoder@1 | 本次运行观测 | 0.5969 |
| rf_feature_ensemble_recall | RandomForestModel@1 | ValidationRun:e0d43fa3 | 0.66 |
| lgbm_tenure_bucket_optimized | *TotalChargesCleaner | 本次运行观测 | 0.6243 |
| lgbm_tenure_bucket_optimized | *TotalChargesCleaner | 本次运行观测 | 0.6205 |
| lgbm_tenure_bucket_optimized | *TotalChargesCleaner | 本次运行观测 | 0.5969 |
| lgbm_tenure_bucket_optimized | *TenureBucketizer | ValidationRun:f6e4284c | 0.8 |
| lgbm_tenure_bucket_optimized | *TenureBucketizer | 本次运行观测 | 0.6205 |
| lgbm_tenure_bucket_optimized | OneHotCategoricalEncoder@1 | ValidationRun:e11ba947 | 0.71 |
| lgbm_tenure_bucket_optimized | OneHotCategoricalEncoder@1 | ValidationRun:e0d43fa3 | 0.66 |
| lgbm_tenure_bucket_optimized | OneHotCategoricalEncoder@1 | ValidationRun:a7f163bc | 0.78 |
| lgbm_tenure_bucket_optimized | OneHotCategoricalEncoder@1 | 本次运行观测 | 0.6243 |
| lgbm_tenure_bucket_optimized | OneHotCategoricalEncoder@1 | 本次运行观测 | 0.6205 |
| lgbm_tenure_bucket_optimized | OneHotCategoricalEncoder@1 | 本次运行观测 | 0.5969 |
| lgbm_tenure_bucket_optimized | LightGBMModel@1 | ValidationRun:f6e4284c | 0.8 |
| lgbm_tenure_bucket_optimized | LightGBMModel@1 | 本次运行观测 | 0.6243 |
| lgbm_tenure_bucket_optimized | LightGBMModel@1 | 本次运行观测 | 0.6205 |
| lgbm_tenure_bucket_optimized | LightGBMModel@1 | 本次运行观测 | 0.5969 |
| logistic_interpretable_weighted | *TotalChargesCleaner | 本次运行观测 | 0.6243 |
| logistic_interpretable_weighted | *TotalChargesCleaner | 本次运行观测 | 0.6205 |
| logistic_interpretable_weighted | *TotalChargesCleaner | 本次运行观测 | 0.5969 |
| logistic_interpretable_weighted | OneHotCategoricalEncoder@1 | ValidationRun:e11ba947 | 0.71 |
| logistic_interpretable_weighted | OneHotCategoricalEncoder@1 | ValidationRun:e0d43fa3 | 0.66 |
| logistic_interpretable_weighted | OneHotCategoricalEncoder@1 | ValidationRun:a7f163bc | 0.78 |
| logistic_interpretable_weighted | OneHotCategoricalEncoder@1 | 本次运行观测 | 0.6243 |
| logistic_interpretable_weighted | OneHotCategoricalEncoder@1 | 本次运行观测 | 0.6205 |
| logistic_interpretable_weighted | OneHotCategoricalEncoder@1 | 本次运行观测 | 0.5969 |
| logistic_interpretable_weighted | StandardFeatureScaler@1 | ValidationRun:e11ba947 | 0.71 |
| logistic_interpretable_weighted | LogisticRegressionModel@1 | ValidationRun:e11ba947 | 0.71 |

## 4. 最优方案：lgbm_imbalanced_pipeline

针对不平衡数据流失预测任务，采用轻量级清洗及经过历史验证的 LightGBM 增强方案，并设置 class_weight='balanced' 以满足 recall_first 需求。

| # | 步骤 | 算子 | 参数 | 选择理由 |
| --- | --- | --- | --- | --- |
| 1 | imputer | 生成 TotalChargesCleaner | - | 根据Lesson:69d51358，需要对脏数据进行清洗以支持后续数值计算。 |
| 2 | imputer | MissingValueImputer@1 | - | 填补数据中的残余缺失值以确保训练数据的完整性。 |
| 3 | feature | 生成 IdColumnDropper | - | 删除无关的标识列以降低噪声影响。 |
| 4 | encoder | OneHotCategoricalEncoder@1 | max_categories=20 | 将类别特征进行 one-hot 编码，符合数据画像中低基数的特征。 |
| 5 | model | LightGBMModel@1 | n_estimators=400, learning_rate=0.05, num_leaves=31, class_weight=balanced, random_state=42 | 根据 Lesson:f2c1398b，使用具有 balanced 权重的 LightGBM 能够有效处理不平衡数据并满足召回率要求。 |

## 5. 能力生成与修复

| 生成算子 | 类型 | 用途 | 参考的历史实现 | 修复轮数 | 注册到能力库 |
| --- | --- | --- | --- | --- | --- |
| TotalChargesCleaner@2 | imputer | 将total_charges字段中的空白字符串转换为数值，并处理异常值以符合后续模型计算需求 | TotalChargesCleaner@1 | 0 | 否 |
| IdColumnDropper@2 | feature | 删除高基数 ID 列，防止其被误解析为类别特征导致过拟合 | IdColumnDropper@1 | 0 | 否 |
| TenureBucketizer@2 | feature | 在网时长分箱，提升模型对不同生命周期用户的识别 | TenureBucketizer@1 | 0 | 否 |
| ChargeRatioFeatureGenerator@2 | feature | 生成用户消费变动趋势特征，捕捉流失风险 | ChargeRatioFeatureGenerator@1 | 1 | 否 |
| ServiceCountFeature@2 | feature | 统计订购服务总数，捕捉粘性特征 | ServiceCountFeature@1 | 0 | 否 |

### 修复 ChargeRatioFeatureGenerator@2 第 1 轮：✅ 已修复

- 诊断：`E_RUNTIME`（sandbox）加载生成算子失败：TypeError: Cannot create a consistent method resolution
order (MRO) for bases BaseEstimator, TransformerMixin, Operator (at <module>:7)
- 提示：-
- 参考经验：2 条

```diff
--- 修复前
+++ 修复后
@@ -2,3 +2,3 @@
 import pandas as pd
-from sklearn.base import TransformerMixin, BaseEstimator
+from sklearn.base import TransformerMixin
 from factory.operators.base import Operator, OperatorMeta
@@ -6,3 +6,3 @@
 
-class ChargeRatioFeatureGenerator(BaseEstimator, TransformerMixin, Operator):
+class ChargeRatioFeatureGenerator(TransformerMixin, Operator):
     """
@@ -34,16 +34,19 @@
         
-        # 预处理：处理潜在的脏数据（空格及非数值）
+        # 预处理：确保数值列类型正确，处理空白字符串及非数值
+        # 先转为字符串去除空格，再转为数值，缺失值填充为0
         total_series = pd.to_numeric(X[self.total].astype(str).str.strip(), errors='coerce').fillna(0.0)
-        tenure_series = pd.to_numeric(X[self.tenure], errors='coerce').replace(0, np.nan)
-        monthly_series = pd.to_numeric(X[self.monthly], errors='coerce').replace(0, np.nan)
+        tenure_series = pd.to_numeric(X[self.tenure], errors='coerce').fillna(0.0)
+        monthly_series = pd.to_numeric(X[self.monthly], errors='coerce').fillna(0.0)
         
         # 计算逻辑：平均消费 = 总消费 / 在网时长
-        avg_charge = total_series / tenure_series
+        # 为避免除以0，使用 np.where 处理 tenure 为 0 的情况
+        avg_charge = np.where(tenure_series > 0, total_series / tenure_series, 0.0)
         
         # 计算比值：charge_ratio = avg_charge / current_monthly_charge
-        # 若在网时长为0则该月暂无消费对比意义，设为1.0
-        ratio = (avg_charge / monthly_series)
+        # 为避免除以0，当月费为0时，结果置为 1.0 (表示无变动)
+        ratio = np.where(monthly_series > 0, avg_charge / monthly_series, 1.0)
         
-        # 处理无穷值与缺失值
-        X["charge_ratio"] = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
+        # 处理可能产生的无穷值与缺失值
+        res = pd.Series(ratio, index=X.index)
+        X["charge_ratio"] = res.replace([np.inf, -np.inf], np.nan).fillna(1.0)
         
```

## 6. 验证机制

| 级别 | 检查内容 | 本次配置 |
| --- | --- | --- |
| 1 静态检查 | 语法、import 白名单、禁止危险调用、接口与 meta 一致性 | 仅针对生成的算子 |
| 2 沙箱执行 | 独立子进程，限制超时与内存，清除含 KEY/TOKEN 的环境变量 | 超时 180 s，内存 8192 MB |
| 3 契约检查 | 不修改输入、行数一致、不引入 NaN/Inf、概率合法、能处理未见类别与缺失值 | 冒烟样本 300 行 |
| 4 指标检查 | 阈值、种子间稳定性、疑似泄漏、耗时与内存 | 5 折 × 种子 [42, 7, 2024]；std 上限 0.03 |

最优方案资源消耗：单次训练 0.287 s，预测 55.6 ms / 千行，峰值内存 265 MB。

修复前的 1 次验证结果保留在 state.json 的 history 中。

## 7. 知识沉淀

- 回写验证记录 ValidationRun：5 条
- 注册到能力库的新算子：无
- 新提炼的经验：Lesson:ab7c0120
- 本次提供的经验：7 条（累计检索命中，不代表采纳）

## 附录 A. 生成的算法代码（pipeline.py）

```python
# 警告：该方案未通过验证，仅为得分最高的候选，供人工参考。
"""预测客户流失，召回率不低于0.8；优先尝试可解释模型，未达标时探索不同模型或参数 —— 自动生成的算法流水线

需求：预测客户流失，召回率不低于0.8；优先尝试可解释模型，未达标时探索不同模型或参数。
方案：lgbm_imbalanced_pipeline
设计依据：针对不平衡数据流失预测任务，采用轻量级清洗及经过历史验证的 LightGBM 增强方案，并设置 class_weight='balanced' 以满足 recall_first 需求。
主指标：recall >= 0.8

由 AlgoForge 自动生成。用法：
    python pipeline.py --data <train.csv>                      # 训练并保存 model.joblib
    python pipeline.py --predict <new.csv> --model model.joblib  # 预测
"""

import argparse

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from factory.operators.models import LightGBMModel
from factory.operators.tabular import MissingValueImputer, OneHotCategoricalEncoder


import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class TotalChargesCleaner(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="TotalChargesCleaner",
        kind=OperatorKind.IMPUTER,
        version=2,
        origin="synthesized",
        description="将total_charges字段中的空白字符串转换为数值，并处理异常值以符合后续模型计算需求",
        conditions=["数据包含 total_charges 列且该列含有空白字符或被识别为字符串"],
        dependencies=["pandas", "numpy"],
        params={"col": "total_charges"},
        parent="TotalChargesCleaner@1"
    )

    def __init__(self, col="total_charges"):
        self.col = col

    def fit(self, X, y=None):
        """记录列名并返回 self"""
        return self

    def transform(self, X):
        """
        修正 total_charges：
        1. 复制输入数据以防原地修改
        2. 将字符串两端的空白符清除
        3. 利用 pd.to_numeric 将非数值转为 NaN
        4. 使用 0 填充缺失值（代表新入网用户）
        """
        X = X.copy()
        if self.col in X.columns:
            # 执行类型转换：先转字符串并去除空格，coerce 非法值为 NaN，最后填补为 0
            X[self.col] = pd.to_numeric(
                X[self.col].astype(str).str.strip(), 
                errors="coerce"
            ).fillna(0.0)

            # 确保最终列为浮点型，并处理任何可能产生的 inf 值
            X[self.col] = X[self.col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return X

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class IdColumnDropper(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="IdColumnDropper",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="自动识别并剔除高基数ID类列，防止模型过拟合及维度灾难",
        conditions=["数据集包含唯一标识列"],
        dependencies=["pandas", "numpy"],
        params={"id_cols": ("customer_id",)},
        parent="IdColumnDropper@1"
    )

    def __init__(self, id_cols=("customer_id",)):
        self.id_cols = id_cols

    def fit(self, X, y=None):
        """
        验证输入数据结构并记录状态。
        """
        self.cols_to_drop_ = [c for c in self.id_cols if c in X.columns]
        return self

    def transform(self, X):
        """
        执行删除操作。
        """
        X = X.copy()

        # 剔除预定义的ID列
        cols_actually_dropped = [c for c in self.cols_to_drop_ if c in X.columns]
        X = X.drop(columns=cols_actually_dropped)

        # 逻辑兜底：检查是否存在其他疑似高基数ID的列（超过行数90%的唯一值）
        # 避免在线上推理阶段因为动态列名导致的问题
        potential_ids = [
            col for col in X.select_dtypes(include=['object']).columns 
            if X[col].nunique() / len(X) > 0.9
        ]
        X = X.drop(columns=potential_ids)

        return X


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_imputer", TotalChargesCleaner()),  # 根据Lesson:69d51358，需要对脏数据进行清洗以支持后续数值计算。
        ("1_imputer", MissingValueImputer()),  # 填补数据中的残余缺失值以确保训练数据的完整性。
        ("2_feature", IdColumnDropper()),  # 删除无关的标识列以降低噪声影响。
        ("3_encoder", OneHotCategoricalEncoder(max_categories=20)),  # 将类别特征进行 one-hot 编码，符合数据画像中低基数的特征。
        ("4_model", LightGBMModel(n_estimators=400, learning_rate=0.05, num_leaves=31, class_weight='balanced', random_state=42)),  # 根据 Lesson:f2c1398b，使用具有 balanced 权重的 LightGBM 能够有效处理不平衡数据并满足召回率要求。
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="训练数据 CSV")
    parser.add_argument("--predict", help="待预测数据 CSV")
    parser.add_argument("--model", default="model.joblib")
    args = parser.parse_args()
    if args.data:
        df = pd.read_csv(args.data)
        pipe = build_pipeline().fit(df.drop(columns=[TARGET]), df[TARGET])
        joblib.dump(pipe, args.model)
        print(f"模型已保存到 {args.model}")
    if args.predict:
        pipe = joblib.load(args.model)
        df = pd.read_csv(args.predict)
        X = df.drop(columns=[TARGET], errors="ignore")
        out = pd.DataFrame(pipe.predict_proba(X), columns=[f"proba_{c}" for c in pipe.classes_])
        out["prediction"] = pipe.predict(X)
        print(out.head().to_string())


if __name__ == "__main__":
    main()

```

## 附录 B. 运行记录

各阶段耗时：PLAN 4.6s，BUILD 17.1s，VERIFY 54.9s，ROUTE 0.0s，REPLAN 29.8s，REPAIR 7.0s，CURATE 2.0s

```
[PLAN] 场景 telecom_churn；画像标签 ['has_missing_values', 'numeric_features', 'has_categorical_features', 'low_cardinality', 'string_typed_numeric', 'id_columns', 'imbalanced', 'time_tenure', 'mixed_features']
[PLAN] TaskSpec：binary_classification，目标 churn，主指标 recall >= 0.8
[PLAN] 候选方案：['lgbm_imbalanced_pipeline(*TotalChargesCleaner → MissingValueImputer → *IdColumnDropper → OneHotCategoricalEncoder → LightGBMModel)']
[PLAN] 完成，用时 4.6s → BUILD
[BUILD] 生成算子 TotalChargesCleaner@2（参考 TotalChargesCleaner@1）：将total_charges字段中的空白字符串转换为数值，并处理异常值以符合后续模型计算需求
[BUILD] 生成算子 IdColumnDropper@2（参考 IdColumnDropper@1）：删除高基数 ID 列，防止其被误解析为类别特征导致过拟合
[BUILD] 完成，用时 6.9s → VERIFY
[VERIFY] lgbm_imbalanced_pipeline：未通过，recall=0.6243；M_THRESHOLD recall = 0.6243 低于阈值 0.8
[VERIFY] 完成，用时 6.9s → ROUTE
[ROUTE] 完成，用时 0.0s → REPLAN
[REPLAN] 第 1 轮 Beam 搜索：父方案 ['lgbm_imbalanced_pipeline']，扩展 4 个，入选 ['lgbm_with_tenure_bucket(*TotalChargesCleaner → *TenureBucketizer → OneHotCategoricalEncoder → LightGBMModel)', 'lgbm_recall_optimized(*TotalChargesCleaner → MissingValueImputer → *ChargeRatioFeatureGenerator → OneHotCategoricalEncoder → LightGBMModel)']，剪枝/去重 ['interpretable_logreg_pipeline(0.7374)', 'rf_recall_focused(0.7362)']
[REPLAN] 完成，用时 14.4s → BUILD
[BUILD] 生成算子 TenureBucketizer@2（参考 TenureBucketizer@1）：在网时长分箱，提升模型对不同生命周期用户的识别
[BUILD] 生成算子 ChargeRatioFeatureGenerator@2（参考 ChargeRatioFeatureGenerator@1）：生成用户消费变动趋势特征，捕捉流失风险
[BUILD] 完成，用时 6.3s → VERIFY
[VERIFY] lgbm_with_tenure_bucket：未通过，recall=0.6205；M_THRESHOLD recall = 0.6205 低于阈值 0.8
[VERIFY] lgbm_recall_optimized：未通过，recall=nan；E_RUNTIME 加载生成算子失败：TypeError: Cannot create a consistent method resolution
order (MRO) for bases BaseEstimator, TransformerMixin, Operator (at <module>:7)
[VERIFY] 完成，用时 8.5s → ROUTE
[ROUTE] 完成，用时 0.0s → REPAIR
[REPAIR] 修复 ChargeRatioFeatureGenerator@2 第 1 轮：E_RUNTIME 加载生成算子失败：TypeError: Cannot create a consistent method resolution
order (MRO) for，参考经验：['数值列含空白字符导致类型转换失败', '预测阶段出现未见类别导致编码报错']
[REPAIR] 完成，用时 7.0s → VERIFY
[VERIFY] lgbm_recall_optimized：未通过，recall=0.5969；M_THRESHOLD recall = 0.5969 低于阈值 0.8
[VERIFY] 完成，用时 7.8s → ROUTE
[ROUTE] 完成，用时 0.0s → REPLAN
[REPLAN] 第 2 轮 Beam 搜索：父方案 ['lgbm_imbalanced_pipeline', 'lgbm_with_tenure_bucket']，扩展 4 个，入选 ['logistic_interpretable_weighted(*TotalChargesCleaner → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel)', 'rf_feature_ensemble_recall(*TotalChargesCleaner → *ServiceCountFeature → OneHotCategoricalEncoder → RandomForestModel)']，剪枝/去重 ['lgbm_feature_rich_weighted(0.6904)', 'lgbm_tenure_bucket_optimized(0.6917)']
[REPLAN] 完成，用时 15.4s → BUILD
[BUILD] 生成算子 ServiceCountFeature@2（参考 ServiceCountFeature@1）：统计订购服务总数，捕捉粘性特征
[BUILD] 完成，用时 3.9s → VERIFY
[VERIFY] logistic_interpretable_weighted：未通过，recall=nan；E_RUNTIME ValueError: Input X contains NaN.
LogisticRegression does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to pr (at _assert_all_finite_element_wise:186)
[VERIFY] rf_feature_ensemble_recall：未通过，recall=0.5957；M_THRESHOLD recall = 0.5957 低于阈值 0.8
[VERIFY] 完成，用时 31.6s → ROUTE
[ROUTE] 没有方案通过验证
[ROUTE] 完成，用时 0.0s → CURATE
[CURATE] 写入 KG：5 条 ValidationRun；注册新算子 []；新增经验 1 条
[CURATE] 完成，用时 2.1s → FAILED
```

### LLM 调用来源

| 阶段 | 请求哈希 | 来源 | 本次耗时(s) | 录制耗时(s) | tokens |
| --- | --- | --- | --- | --- | --- |
| plan | 9e8b1f363da5adca | remote | 4.457 | - | 10497 |
| synthesize | 958dc06c79ca26de | remote | 2.309 | - | 2556 |
| synthesize | 6b1b41adf0120fc0 | remote | 4.534 | - | 2432 |
| plan | 94c5a021617d7b6b | remote | 6.600 | - | 12601 |
| plan | 754eeea3bb0ca8f1 | remote | 7.807 | - | 15008 |
| synthesize | 3cce52cac7568db9 | remote | 3.031 | - | 2714 |
| synthesize | 66c718b6ea879136 | remote | 3.304 | - | 2673 |
| repair | 6dd7a2efe9f51715 | remote | 7.019 | - | 2576 |
| plan | 418654db3359021c | remote | 8.856 | - | 13309 |
| plan | ad7d8a48c09a29d9 | remote | 6.533 | - | 15636 |
| synthesize | 67d2cb7d8b4567f2 | remote | 3.861 | - | 2527 |
| lesson | 7731f1c33d28d0f4 | remote | 2.048 | - | 1919 |
