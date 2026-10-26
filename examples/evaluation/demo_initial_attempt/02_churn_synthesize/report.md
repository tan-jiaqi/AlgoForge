# AlgoForge 验证报告

| 项目 | 内容 |
| --- | --- |
| 需求 | 预测30天内可能流失的客户，需可解释，召回优先 |
| 数据 | data/churn.csv |
| 运行 ID | 02_churn_synthesize |
| LLM | gemini-3.5-flash |
| 状态 | ❌ 未通过 |
| 最优方案 | lgbm_balanced_pipeline |
| 总耗时 | 82.8 s |

共验证 6 个候选方案，0 个通过；重新规划 1 次，修复 3 轮。最优方案 `lgbm_balanced_pipeline` 的 recall = 0.624（要求 ≥ 0.75）。

## 1. 需求理解（TaskSpec）

| 字段 | 值 |
| --- | --- |
| 任务类型 | binary_classification |
| 数据模态 | tabular |
| 目标列 | churn |
| 评价指标 | recall ≥ 0.75（主）；roc_auc ≥ 0.8 |
| 业务约束 | interpretable、recall_first |
| 数据画像标签 | has_missing_values、numeric_features、has_categorical_features、low_cardinality、string_typed_numeric、id_columns、imbalanced、time_tenure、mixed_features |

## 2. 知识检索（设计依据）

匹配场景：telecom_churn。每类步骤从知识图谱检索候选算子，得分 = 0.4 × 文本相关度 + 0.3 × 条件匹配 + 0.3 × 历史验证分。

| 步骤 | 候选算子 | 可直接复用 | 得分 | 文本/条件/历史 | 证据 |
| --- | --- | --- | --- | --- | --- |
| imputer | MissingValueImputer@1 | 是 | 0.492 | 0.62 / 0.09 / 0.72 | 匹配条件：has_missing_values；历史验证 3 次，平均 recall 0.717 |
| imputer | TotalChargesCleaner@1 | 否（历史实现，可参考生成） | 0.482 | 1.00 / 0.27 / 0.00 | 匹配条件：has_missing_values、numeric_features、string_typed_numeric |
| feature | TenureBucketizer@1 | 否（历史实现，可参考生成） | 0.707 | 0.96 / 0.27 / 0.80 | 匹配条件：interpretable、numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | ChargeRatioFeatureGenerator@1 | 否（历史实现，可参考生成） | 0.606 | 0.78 / 0.18 / 0.80 | 匹配条件：numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | ServiceCountFeature@1 | 否（历史实现，可参考生成） | 0.427 | 1.00 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| encoder | OneHotCategoricalEncoder@1 | 是 | 0.669 | 1.00 / 0.18 / 0.72 | 匹配条件：has_categorical_features、low_cardinality；历史验证 3 次，平均 recall 0.717 |
| encoder | KFoldTargetEncoder@1 | 否（历史实现，可参考生成） | 0.402 | 0.94 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| scaler | StandardFeatureScaler@1 | 是 | 0.640 | 1.00 / 0.09 / 0.71 | 匹配条件：numeric_features；历史验证 1 次，平均 recall 0.710 |
| model | LogisticRegressionModel@1 | 是 | 0.640 | 1.00 / 0.09 / 0.71 | 匹配条件：interpretable；历史验证 1 次，平均 recall 0.710 |
| model | BalancedLightGBMClassifier@1 | 否（历史实现，可参考生成） | 0.593 | 0.83 / 0.09 / 0.78 | 匹配条件：imbalanced；历史验证 1 次，平均 recall 0.780 |
| model | RandomForestModel@1 | 是 | 0.546 | 0.80 / 0.09 / 0.66 | 匹配条件：mixed_features；历史验证 1 次，平均 recall 0.660 |

### 参考的历史经验

- **逻辑回归标准化与可解释性**：必须先进行特征标准化（如 StandardFeatureScaler），在 0.1~10 之间搜索正则化强度 C，可直接输出系数给业务方。
- **随机森林防过拟合与召回优化**：设置 n_estimators 300 左右，限制 max_depth 在 8~16 防止过拟合；召回优先时需配合调整分类阈值。
- **数值列含空白字符导致类型转换失败**（E_RUNTIME）：先 str.strip()，再 pd.to_numeric(errors='coerce')，缺失值填补为 0
- **样本不平衡导致默认模型召回率偏低**（M_THRESHOLD）：启用平衡类权重（如 class_weight='balanced' 或 BalancedLightGBMClassifier）以提升召回率
- **移除对 pandas 特定 API 的依赖以增强输入类型兼容性**（E_RUNTIME）：在数据预处理阶段，优先进行类型检查，通过转换为 NumPy 数组或使用通用的数值清洗逻辑，避免直接调用特定数据结构（如 DataFrame）独有的 API，从而确保对密集矩阵、稀疏矩阵及 DataFrame 的广泛兼容性。

### 知识引用检查

检索并传入上下文不等于采纳。以下引用由模型声明，程序仅核对可见范围与结构化适用对象；不证明语义落实或指标收益。

| 方案 | 步骤 | 模型引用 | 范围检查 | 语义落实 |
| --- | --- | --- | --- | --- |
| lgbm_balanced_pipeline | 1 | Operator:TotalChargesCleaner@1、Lesson:69d51358 | 通过 | 未验证 |
| lgbm_balanced_pipeline | 2 | Operator:MissingValueImputer@1 | 通过 | 未验证 |
| lgbm_balanced_pipeline | 3 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| lgbm_balanced_pipeline | 4 | Operator:BalancedLightGBMClassifier@1、Lesson:9e9e77d6 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 1 | Operator:TotalChargesCleaner@1 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 2 | Operator:TenureBucketizer@1 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 3 | Operator:StandardFeatureScaler@1、Lesson:74e2da10 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 4 | Operator:LogisticRegressionModel@1、Lesson:9e9e77d6 | 通过 | 未验证 |
| rf_nonlinear_pipeline | 1 | Operator:MissingValueImputer@1 | 通过 | 未验证 |
| rf_nonlinear_pipeline | 2 | Operator:ChargeRatioFeatureGenerator@1 | 通过 | 未验证 |
| rf_nonlinear_pipeline | 3 | Operator:RandomForestModel@1、Lesson:57718714 | 通过 | 未验证 |
| lgbm_tuned_recall_boost | 1 | Operator:TotalChargesCleaner@1 | 通过 | 未验证 |
| lgbm_tuned_recall_boost | 2 | Operator:MissingValueImputer@1 | 通过 | 未验证 |
| lgbm_tuned_recall_boost | 3 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| lgbm_tuned_recall_boost | 4 | Operator:BalancedLightGBMClassifier@1 | 通过 | 未验证 |
| logreg_improved_robustness | 1 | Operator:TotalChargesCleaner@1 | 通过 | 未验证 |
| logreg_improved_robustness | 2 | Operator:TenureBucketizer@1 | 通过 | 未验证 |
| logreg_improved_robustness | 3 | Operator:StandardFeatureScaler@1 | 通过 | 未验证 |
| logreg_improved_robustness | 4 | Operator:LogisticRegressionModel@1 | 通过 | 未验证 |
| rf_nonlinear_cleaned | 1 | Operator:TotalChargesCleaner@1 | 通过 | 未验证 |
| rf_nonlinear_cleaned | 2 | Operator:MissingValueImputer@1 | 通过 | 未验证 |
| rf_nonlinear_cleaned | 3 | Operator:ChargeRatioFeatureGenerator@1 | 通过 | 未验证 |
| rf_nonlinear_cleaned | 4 | Operator:RandomForestModel@1 | 通过 | 未验证 |
| lgbm_basic_fixed | 1 | Operator:TotalChargesCleaner@1 | 通过 | 未验证 |
| lgbm_basic_fixed | 2 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| lgbm_basic_fixed | 3 | Operator:BalancedLightGBMClassifier@1 | 通过 | 未验证 |
| logreg_target_encoding | 1 | Operator:TotalChargesCleaner@1 | 通过 | 未验证 |
| logreg_target_encoding | 2 | Operator:KFoldTargetEncoder@1 | 通过 | 未验证 |
| logreg_target_encoding | 3 | Operator:StandardFeatureScaler@1 | 通过 | 未验证 |
| logreg_target_encoding | 4 | Operator:LogisticRegressionModel@1 | 通过 | 未验证 |
| rf_tenure_enhanced | 1 | Operator:TotalChargesCleaner@1 | 通过 | 未验证 |
| rf_tenure_enhanced | 2 | Operator:TenureBucketizer@1 | 通过 | 未验证 |
| rf_tenure_enhanced | 3 | Operator:RandomForestModel@1 | 通过 | 未验证 |

## 3. 候选方案比较

排序规则：先看是否通过全部检查，再比主指标，再比次要指标之和，最后训练耗时短者优先。指标为多随机种子 × 分层 K 折交叉验证的均值 ± 种子间标准差。

| 排名 | 方案 | 结果 | recall | roc_auc | precision | f1 | recall_at_20 | 单次训练 (s) | 峰值内存 (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ★ lgbm_balanced_pipeline | 未通过 | 0.624 ± 0.007 | 0.783 ± 0.003 | 0.492 ± 0.002 | 0.550 ± 0.003 | 0.420 ± 0.006 | 0.311 | 262 |
| 2 | interpretable_logreg_pipeline | 未通过 | - | - | - | - | - | - | 233 |
| 3 | rf_nonlinear_pipeline | 未通过 | - | - | - | - | - | - | 234 |
| 4 | logreg_improved_robustness | 未通过 | - | - | - | - | - | - | 233 |
| 5 | rf_tenure_enhanced | 未通过 | - | - | - | - | - | - | 232 |
| 6 | logreg_target_encoding | 未通过 | - | - | - | - | - | - | 232 |

```
recall（| 为阈值 0.75）
lgbm_balanced_pipeline               ■■■■■■■■■■■■···|···· 0.624
interpretable_logreg_pipeline        ···············|···· -
rf_nonlinear_pipeline                ···············|···· -
logreg_improved_robustness           ···············|···· -
rf_tenure_enhanced                   ···············|···· -
logreg_target_encoding               ···············|···· -
```

### 各方案的结构与诊断

- `lgbm_balanced_pipeline`：生成:TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → 生成:BalancedLightGBMClassifier。诊断：`M_THRESHOLD` recall = 0.6243 低于阈值 0.75；`M_THRESHOLD` roc_auc = 0.7835 低于阈值 0.8
- `interpretable_logreg_pipeline`：生成:TotalChargesCleaner → 生成:TenureBucketizer → StandardFeatureScaler → LogisticRegressionModel。诊断：`E_RUNTIME` ValueError: could not convert string to float: 'C102752' (at __array__:2027)
- `rf_nonlinear_pipeline`：MissingValueImputer → 生成:ChargeRatioFeatureGenerator → RandomForestModel。诊断：`E_RUNTIME` ValueError: could not convert string to float: 'C102752' (at __array__:2027)
- `logreg_improved_robustness`：生成:TotalChargesCleaner → 生成:TenureBucketizer → StandardFeatureScaler → LogisticRegressionModel。诊断：`E_RUNTIME` ValueError: could not convert string to float: 'C102752' (at __array__:2027)
- `rf_tenure_enhanced`：生成:TotalChargesCleaner → 生成:TenureBucketizer → RandomForestModel。诊断：`E_RUNTIME` ValueError: could not convert string to float: 'C102752' (at __array__:2027)
- `logreg_target_encoding`：生成:TotalChargesCleaner → 生成:KFoldTargetEncoder → StandardFeatureScaler → LogisticRegressionModel。诊断：`E_RUNTIME` ValueError: Input X contains NaN.
LogisticRegression does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to pr (at _assert_all_finite_element_wise:186)

### 方案搜索（Beam Search）

指标不达标时，以已验证结果排名前 k 的方案为父方案，由 Planner 扩展 k × m 个子方案；与已验证方案结构相同的直接去重，其余按先验（知识图谱中该场景的历史验证记录 + 本次运行的观测，平滑后按步骤加权，模型步骤权重更高）估计 recall，只保留前 k 个进入验证。

第 1 轮：父方案 `lgbm_balanced_pipeline`, `interpretable_logreg_pipeline`, `rf_nonlinear_pipeline`，beam 宽度 3。

| 子方案 | 结构 | 先验 recall | 实际 recall | 结果 |
| --- | --- | --- | --- | --- |
| lgbm_tuned_recall_boost | 生成:TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → 生成:BalancedLightGBMClassifier | - | - | 与已有方案重复，跳过 |
| ✔ logreg_improved_robustness | 生成:TotalChargesCleaner → 生成:TenureBucketizer → StandardFeatureScaler → LogisticRegressionModel | 0.730 | - | 先验第 1，进入验证 |
| rf_nonlinear_cleaned | 生成:TotalChargesCleaner → MissingValueImputer → 生成:ChargeRatioFeatureGenerator → RandomForestModel | 0.714 | - | 先验第 4，超出 beam 宽度 3，剪枝 |
| lgbm_basic_fixed | 生成:TotalChargesCleaner → OneHotCategoricalEncoder → 生成:BalancedLightGBMClassifier | 0.709 | - | 先验第 5，超出 beam 宽度 3，剪枝 |
| ✔ logreg_target_encoding | 生成:TotalChargesCleaner → 生成:KFoldTargetEncoder → StandardFeatureScaler → LogisticRegressionModel | 0.726 | - | 先验第 2，进入验证 |
| ✔ rf_tenure_enhanced | 生成:TotalChargesCleaner → 生成:TenureBucketizer → RandomForestModel | 0.715 | - | 先验第 3，进入验证 |

| 子方案 | 步骤 | 先验证据来源 | 数值 |
| --- | --- | --- | --- |
| logreg_improved_robustness | *TotalChargesCleaner | 本次运行观测 | 0.6243 |
| logreg_improved_robustness | *TenureBucketizer | ValidationRun:f6e4284c | 0.8 |
| logreg_improved_robustness | StandardFeatureScaler@1 | ValidationRun:e11ba947 | 0.71 |
| logreg_improved_robustness | LogisticRegressionModel@1 | ValidationRun:e11ba947 | 0.71 |
| rf_nonlinear_cleaned | *TotalChargesCleaner | 本次运行观测 | 0.6243 |
| rf_nonlinear_cleaned | MissingValueImputer@1 | ValidationRun:e11ba947 | 0.71 |
| rf_nonlinear_cleaned | MissingValueImputer@1 | ValidationRun:e0d43fa3 | 0.66 |
| rf_nonlinear_cleaned | MissingValueImputer@1 | ValidationRun:a7f163bc | 0.78 |
| rf_nonlinear_cleaned | MissingValueImputer@1 | 本次运行观测 | 0.6243 |
| rf_nonlinear_cleaned | *ChargeRatioFeatureGenerator | ValidationRun:f6e4284c | 0.8 |
| rf_nonlinear_cleaned | RandomForestModel@1 | ValidationRun:e0d43fa3 | 0.66 |
| lgbm_basic_fixed | *TotalChargesCleaner | 本次运行观测 | 0.6243 |
| lgbm_basic_fixed | OneHotCategoricalEncoder@1 | ValidationRun:e11ba947 | 0.71 |
| lgbm_basic_fixed | OneHotCategoricalEncoder@1 | ValidationRun:e0d43fa3 | 0.66 |
| lgbm_basic_fixed | OneHotCategoricalEncoder@1 | ValidationRun:a7f163bc | 0.78 |
| lgbm_basic_fixed | OneHotCategoricalEncoder@1 | 本次运行观测 | 0.6243 |
| lgbm_basic_fixed | *BalancedLightGBMClassifier | ValidationRun:a7f163bc | 0.78 |
| lgbm_basic_fixed | *BalancedLightGBMClassifier | 本次运行观测 | 0.6243 |
| logreg_target_encoding | *TotalChargesCleaner | 本次运行观测 | 0.6243 |
| logreg_target_encoding | StandardFeatureScaler@1 | ValidationRun:e11ba947 | 0.71 |
| logreg_target_encoding | LogisticRegressionModel@1 | ValidationRun:e11ba947 | 0.71 |
| rf_tenure_enhanced | *TotalChargesCleaner | 本次运行观测 | 0.6243 |
| rf_tenure_enhanced | *TenureBucketizer | ValidationRun:f6e4284c | 0.8 |
| rf_tenure_enhanced | RandomForestModel@1 | ValidationRun:e0d43fa3 | 0.66 |

## 4. 最优方案：lgbm_balanced_pipeline

利用 LightGBM 的处理不平衡能力及针对性清洗，实现高性能召回。

| # | 步骤 | 算子 | 参数 | 选择理由 |
| --- | --- | --- | --- | --- |
| 1 | imputer | 生成 TotalChargesCleaner | - | 解决数据中 string_typed_numeric 问题，防止类型转换错误。 |
| 2 | imputer | MissingValueImputer@1 | - | 对剩余的缺失值进行通用填充。 |
| 3 | encoder | OneHotCategoricalEncoder@1 | max_categories=20 | 类别特征编码，适配 LightGBM。 |
| 4 | model | 生成 BalancedLightGBMClassifier | - | 利用集成学习模型处理不平衡数据，优先保证高召回率。 |

## 5. 能力生成与修复

| 生成算子 | 类型 | 用途 | 参考的历史实现 | 修复轮数 | 注册到能力库 |
| --- | --- | --- | --- | --- | --- |
| TotalChargesCleaner@2 | imputer | 将字符串类型的数值列转换为浮点型并处理缺失值 | TotalChargesCleaner@1 | 0 | 否 |
| BalancedLightGBMClassifier@2 | model | 不平衡数据下的流失预测模型 | BalancedLightGBMClassifier@1 | 0 | 否 |
| TenureBucketizer@2 | feature | 时间长度分箱增强线性模型解释性 | TenureBucketizer@1 | 0 | 否 |
| ChargeRatioFeatureGenerator@2 | feature | 构造历史消费比值特征 | ChargeRatioFeatureGenerator@1 | 0 | 否 |
| KFoldTargetEncoder@2 | encoder | 目标编码 | KFoldTargetEncoder@1 | 3 | 否 |

### 修复 KFoldTargetEncoder@2 第 1 轮：❌ 未修复

- 诊断：`E_RUNTIME`（sandbox）TypeError: cannot concatenate object of type '<class 'numpy.ndarray'>'; only Series and DataFrame objs are valid (at _clean_keys_and_objs:836)
- 提示：-
- 参考经验：3 条

```diff
--- 修复前
+++ 修复后
@@ -3,5 +3,5 @@
 from sklearn.base import TransformerMixin
-from sklearn.model_selection import KFold
 from factory.operators.base import Operator, OperatorMeta
 from factory.schemas import OperatorKind
+from sklearn.model_selection import KFold
 
@@ -27,7 +27,9 @@
     def fit(self, X, y=None):
-        X = X.copy()
+        X = pd.DataFrame(X)
         if y is None:
             y = X[self.target]
-        
-        self.cols = self.cols if self.cols else X.select_dtypes(include=['object', 'category']).columns.tolist()
+        else:
+            y = pd.Series(y, index=X.index)
+
+        self.cols_ = self.cols if self.cols else X.select_dtypes(include=['object', 'category']).columns.tolist()
         self.mapping_ = {}
@@ -35,9 +37,8 @@
 
-        for col in self.cols:
-            # 存储每个折的编码映射以及最终的全量平滑值
-            col_mappings = []
-            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
-            
-            # 计算全量平滑值，用于 transform 阶段的填充（包含未见类别）
-            stats = pd.concat([X[col], y], axis=1).groupby(col)[self.target].agg(["mean", "count"])
+        # 准备用于存储训练集 OOF (Out-of-fold) 结果的容器
+        self.train_encoded_ = X.copy()
+
+        for col in self.cols_:
+            # 计算全局平滑映射，用于 transform 阶段（测试集或未见类别）
+            stats = pd.concat([X[col], y], axis=1).groupby(col)[y.name].agg(["mean", "count"])
             smooth = (stats["mean"] * stats["count"] + self.prior_ * self.smoothing) / (stats["count"] + self.smoothing)
@@ -45,4 +46,6 @@
 
-            # 存储折内映射用于训练集转换
+            # 进行 K 折交叉验证，为训练集生成 OOF 编码
             fold_encodings = pd.Series(index=X.index, dtype=float)
+            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
+            
             for fit_idx, val_idx in kf.split(X):
@@ -51,3 +54,3 @@
                 
-                stats_fold = pd.concat([train_fold[col], target_fold], axis=1).groupby(col)[self.target].agg(["mean", "count"])
+                stats_fold = pd.concat([train_fold[col], target_fold], axis=1).groupby(col)[y.name].agg(["mean", "count"])
                 smooth_fold = (stats_fold["mean"] * stats_fold["count"] + self.prior_ * self.smoothing) / (stats_fold["count"] + self.smoothing)
@@ -56,3 +59,3 @@
             
-            self.mapping_[f"{col}_fold_enc"] = fold_encodings
+            self.train_encoded_[col] = fold_encodings.fillna(self.prior_)
 
@@ -61,12 +64,16 @@
     def transform(self, X):
-        X = X.copy()
-        for col in self.cols:
-            # 如果在训练集中，优先使用折内编码，否则使用全量平均值映射
-            if f"{col}_fold_enc" in self.mapping_:
-                X[col] = self.mapping_[f"{col}_fold_enc"]
+        X = pd.DataFrame(X).copy()
+        
+        # 判断是否为训练集的 transform，如果是则直接返回训练好的 OOF 结果（通过比较列名和内容判断）
+        # 这里为了稳健性，通过是否属于 fit 过程产生的数据来判断
+        if hasattr(self, 'train_encoded_') and X.equals(X.iloc[:len(self.train_encoded_)].head(len(X)) if len(X) <= len(self.train_encoded_) else X.head(0)):
+             # 简化判断：如果输入与 fit 时一致，返回缓存的 OOF
+             return self.train_encoded_[self.cols_].copy()
+
+        for col in self.cols_:
+            # 对新数据，必须使用全局平滑映射
+            mapping = self.mapping_.get(col)
+            X[col] = X[col].map(mapping).fillna(self.prior_)
             
-            # 使用全量平滑后的映射处理测试集或未见类别
-            X[col] = X[col].map(self.mapping_[col]).fillna(self.prior_)
-            
-            # 处理可能的无穷值或缺失值
+            # 处理缺失值、无穷值
             X[col] = X[col].replace([np.inf, -np.inf], self.prior_).fillna(self.prior_)
```

### 修复 KFoldTargetEncoder@2 第 2 轮：❌ 未修复

- 诊断：`E_RUNTIME`（sandbox）KeyError: 'Column not found: None' (at __getitem__:228)
- 提示：-
- 参考经验：3 条

```diff
--- 修复前
+++ 修复后
@@ -29,3 +29,7 @@
         if y is None:
-            y = X[self.target]
+            # 尝试从 X 中获取 target，若不存在则可能抛出 KeyError，此处应做保护
+            if self.target in X.columns:
+                y = X[self.target]
+            else:
+                raise ValueError(f"Target column '{self.target}' not found in input data.")
         else:
@@ -33,3 +37,3 @@
 
-        self.cols_ = self.cols if self.cols else X.select_dtypes(include=['object', 'category']).columns.tolist()
+        self.cols_ = self.cols if self.cols is not None else X.select_dtypes(include=['object', 'category']).columns.tolist()
         self.mapping_ = {}
@@ -37,3 +41,3 @@
 
-        # 准备用于存储训练集 OOF (Out-of-fold) 结果的容器
+        # 记录训练集结果
         self.train_encoded_ = X.copy()
@@ -41,3 +45,3 @@
         for col in self.cols_:
-            # 计算全局平滑映射，用于 transform 阶段（测试集或未见类别）
+            # 计算全局平滑映射，用于预测阶段
             stats = pd.concat([X[col], y], axis=1).groupby(col)[y.name].agg(["mean", "count"])
@@ -48,3 +52,3 @@
             fold_encodings = pd.Series(index=X.index, dtype=float)
-            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
+            kf = KFold(n_splits=min(self.n_splits, len(X)), shuffle=True, random_state=42)
             
@@ -66,15 +70,14 @@
         
-        # 判断是否为训练集的 transform，如果是则直接返回训练好的 OOF 结果（通过比较列名和内容判断）
-        # 这里为了稳健性，通过是否属于 fit 过程产生的数据来判断
-        if hasattr(self, 'train_encoded_') and X.equals(X.iloc[:len(self.train_encoded_)].head(len(X)) if len(X) <= len(self.train_encoded_) else X.head(0)):
-             # 简化判断：如果输入与 fit 时一致，返回缓存的 OOF
-             return self.train_encoded_[self.cols_].copy()
-
+        # 针对每个列进行编码
         for col in self.cols_:
-            # 对新数据，必须使用全局平滑映射
+            if col not in X.columns:
+                continue
+            
+            # 使用训练阶段得到的全局映射
             mapping = self.mapping_.get(col)
+            # 使用 map 处理，未见类别会变为 NaN，fill 后变为 prior_
             X[col] = X[col].map(mapping).fillna(self.prior_)
             
-            # 处理缺失值、无穷值
-            X[col] = X[col].replace([np.inf, -np.inf], self.prior_).fillna(self.prior_)
+            # 处理数值异常
+            X[col] = pd.to_numeric(X[col], errors='coerce').replace([np.inf, -np.inf], self.prior_).fillna(self.prior_)
             
```

### 修复 KFoldTargetEncoder@2 第 3 轮：-

- 诊断：`E_RUNTIME`（sandbox）KeyError: 'Column not found: None' (at __getitem__:228)
- 提示：-
- 参考经验：3 条

```diff
--- 修复前
+++ 修复后
@@ -29,3 +29,2 @@
         if y is None:
-            # 尝试从 X 中获取 target，若不存在则可能抛出 KeyError，此处应做保护
             if self.target in X.columns:
@@ -36,2 +35,6 @@
             y = pd.Series(y, index=X.index)
+        
+        # 确保 y 有名称以支持后续 groupby 操作
+        y_name = y.name if y.name is not None else 'target'
+        y = y.rename(y_name)
 
@@ -46,3 +49,4 @@
             # 计算全局平滑映射，用于预测阶段
-            stats = pd.concat([X[col], y], axis=1).groupby(col)[y.name].agg(["mean", "count"])
+            df_col = pd.concat([X[col], y], axis=1)
+            stats = df_col.groupby(col)[y_name].agg(["mean", "count"])
             smooth = (stats["mean"] * stats["count"] + self.prior_ * self.smoothing) / (stats["count"] + self.smoothing)
@@ -52,3 +56,4 @@
             fold_encodings = pd.Series(index=X.index, dtype=float)
-            kf = KFold(n_splits=min(self.n_splits, len(X)), shuffle=True, random_state=42)
+            n_splits = min(self.n_splits, len(X))
+            kf = KFold(n_splits=max(n_splits, 2), shuffle=True, random_state=42)
             
@@ -58,3 +63,3 @@
                 
-                stats_fold = pd.concat([train_fold[col], target_fold], axis=1).groupby(col)[y.name].agg(["mean", "count"])
+                stats_fold = pd.concat([train_fold[col], target_fold], axis=1).groupby(col)[y_name].agg(["mean", "count"])
                 smooth_fold = (stats_fold["mean"] * stats_fold["count"] + self.prior_ * self.smoothing) / (stats_fold["count"] + self.smoothing)
@@ -70,3 +75,2 @@
         
-        # 针对每个列进行编码
         for col in self.cols_:
@@ -75,8 +79,7 @@
             
-            # 使用训练阶段得到的全局映射
+            # 使用训练阶段得到的全局映射，未见类别通过 fillna(self.prior_) 处理
             mapping = self.mapping_.get(col)
-            # 使用 map 处理，未见类别会变为 NaN，fill 后变为 prior_
             X[col] = X[col].map(mapping).fillna(self.prior_)
             
-            # 处理数值异常
+            # 处理数值异常与类型转换
             X[col] = pd.to_numeric(X[col], errors='coerce').replace([np.inf, -np.inf], self.prior_).fillna(self.prior_)
```

## 6. 验证机制

| 级别 | 检查内容 | 本次配置 |
| --- | --- | --- |
| 1 静态检查 | 语法、import 白名单、禁止危险调用、接口与 meta 一致性 | 仅针对生成的算子 |
| 2 沙箱执行 | 独立子进程，限制超时与内存，清除含 KEY/TOKEN 的环境变量 | 超时 180 s，内存 8192 MB |
| 3 契约检查 | 不修改输入、行数一致、不引入 NaN/Inf、概率合法、能处理未见类别与缺失值 | 冒烟样本 300 行 |
| 4 指标检查 | 阈值、种子间稳定性、疑似泄漏、耗时与内存 | 5 折 × 种子 [42, 7, 2024]；std 上限 0.03 |

最优方案资源消耗：单次训练 0.311 s，预测 63.5 ms / 千行，峰值内存 262 MB。

修复前的 3 次验证结果保留在 state.json 的 history 中。

## 7. 知识沉淀

- 回写验证记录 ValidationRun：6 条
- 注册到能力库的新算子：无
- 新提炼的经验：无
- 本次提供的经验：10 条（累计检索命中，不代表采纳）

## 附录 A. 生成的算法代码（pipeline.py）

```python
# 警告：该方案未通过验证，仅为得分最高的候选，供人工参考。
"""预测30天内可能流失的客户，需可解释，召回优先 —— 自动生成的算法流水线

需求：预测30天内可能流失的客户，需可解释，召回优先
方案：lgbm_balanced_pipeline
设计依据：利用 LightGBM 的处理不平衡能力及针对性清洗，实现高性能召回。
主指标：recall >= 0.75

由 AlgoForge 自动生成。用法：
    python pipeline.py --data <train.csv>                      # 训练并保存 model.joblib
    python pipeline.py --predict <new.csv> --model model.joblib  # 预测
"""

import argparse

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from factory.operators.tabular import MissingValueImputer, OneHotCategoricalEncoder


import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class TotalChargesCleaner(TransformerMixin, Operator):
    """
    将 total_charges 列从字符串转换为浮点型，处理包含空白字符的异常值，并填充缺失值。
    """

    meta = OperatorMeta(
        name="TotalChargesCleaner",
        kind=OperatorKind.IMPUTER,
        version=2,
        origin="synthesized",
        description="将字符串类型的total_charges列转换为浮点型并处理缺失值",
        conditions=["数据包含total_charges列", "列内存在空白字符串或非数值字符"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"col": "total_charges"},
        parent="TotalChargesCleaner@1"
    )

    def __init__(self, col="total_charges"):
        self.col = col

    def fit(self, X, y=None):
        # 记录已适配状态，该算子为无状态转换器
        self.is_fitted_ = True
        return self

    def transform(self, X):
        # 1. 确保输入为 DataFrame 副本以避免副作用
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.copy()

        # 2. 检查列是否存在
        if self.col not in X.columns:
            return X

        # 3. 处理逻辑：
        # - strip() 清理空白
        # - to_numeric(errors='coerce') 将无法转换的转换为 NaN
        # - fillna(0.0) 填充由于新入网产生的空值或异常值
        X[self.col] = pd.to_numeric(
            X[self.col].astype(str).str.strip(), 
            errors="coerce"
        ).fillna(0.0)

        # 4. 确保输出列为数值类型，处理潜在的无穷值
        X[self.col] = X[self.col].replace([np.inf, -np.inf], 0.0)

        return X

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import ClassifierMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class BalancedLightGBMClassifier(ClassifierMixin, Operator):
    meta = OperatorMeta(
        name="BalancedLightGBMClassifier",
        kind=OperatorKind.MODEL,
        version=2,
        origin="synthesized",
        description="针对流失预测场景下不平衡数据设计的 LightGBM 分类器，优先保障召回率。",
        conditions=["适用于正负样本不平衡的二分类场景", "无需对特征进行标准化"],
        dependencies=["lightgbm", "numpy", "pandas", "sklearn"],
        params={
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "class_weight": "balanced",
            "random_state": 42
        },
        parent="BalancedLightGBMClassifier@1"
    )

    def __init__(self, n_estimators=400, learning_rate=0.05, num_leaves=31, class_weight="balanced", random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y):
        # 确保输入为 DataFrame 以处理潜在的 ID 列过滤
        X_df = pd.DataFrame(X)
        # 简单剔除字符型特征（模型通常需要数值输入，LightGBM可直接处理category，此处简化为数值化预处理）
        X_numeric = X_df.select_dtypes(include=[np.number])

        self.model_ = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            class_weight=self.class_weight,
            random_state=self.random_state,
            verbose=-1
        )

        self.model_.fit(X_numeric, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        X_df = pd.DataFrame(X)
        X_numeric = X_df.select_dtypes(include=[np.number])
        # 处理缺失值，填充为均值或0，防止报错
        X_numeric = X_numeric.fillna(0)
        return self.model_.predict_proba(X_numeric)

    def predict(self, X):
        X_df = pd.DataFrame(X)
        X_numeric = X_df.select_dtypes(include=[np.number])
        X_numeric = X_numeric.fillna(0)
        return self.model_.predict(X_numeric)


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_imputer", TotalChargesCleaner()),  # 解决数据中 string_typed_numeric 问题，防止类型转换错误。
        ("1_imputer", MissingValueImputer()),  # 对剩余的缺失值进行通用填充。
        ("2_encoder", OneHotCategoricalEncoder(max_categories=20)),  # 类别特征编码，适配 LightGBM。
        ("3_model", BalancedLightGBMClassifier()),  # 利用集成学习模型处理不平衡数据，优先保证高召回率。
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

各阶段耗时：PLAN 6.7s，BUILD 17.7s，VERIFY 21.4s，ROUTE 0.0s，REPLAN 16.1s，REPAIR 20.9s，CURATE 0.0s

```
[PLAN] 场景 telecom_churn；画像标签 ['has_missing_values', 'numeric_features', 'has_categorical_features', 'low_cardinality', 'string_typed_numeric', 'id_columns', 'imbalanced', 'time_tenure', 'mixed_features']
[PLAN] TaskSpec：binary_classification，目标 churn，主指标 recall >= 0.75
[PLAN] 候选方案：['lgbm_balanced_pipeline(*TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → *BalancedLightGBMClassifier)', 'interpretable_logreg_pipeline(*TotalChargesCleaner → *TenureBucketizer → StandardFeatureScaler → LogisticRegressionModel)', 'rf_nonlinear_pipeline(MissingValueImputer → *ChargeRatioFeatureGenerator → RandomForestModel)']
[PLAN] 完成，用时 6.7s → BUILD
[BUILD] 生成算子 TotalChargesCleaner@2（参考 TotalChargesCleaner@1）：将字符串类型的数值列转换为浮点型并处理缺失值
[BUILD] 生成算子 BalancedLightGBMClassifier@2（参考 BalancedLightGBMClassifier@1）：不平衡数据下的流失预测模型
[BUILD] 生成算子 TenureBucketizer@2（参考 TenureBucketizer@1）：时间长度分箱增强线性模型解释性
[BUILD] 生成算子 ChargeRatioFeatureGenerator@2（参考 ChargeRatioFeatureGenerator@1）：构造历史消费比值特征
[BUILD] 完成，用时 14.2s → VERIFY
[VERIFY] lgbm_balanced_pipeline：未通过，recall=0.6243；M_THRESHOLD recall = 0.6243 低于阈值 0.75；M_THRESHOLD roc_auc = 0.7835 低于阈值 0.8
[VERIFY] interpretable_logreg_pipeline：未通过，recall=nan；E_RUNTIME ValueError: could not convert string to float: 'C102752' (at __array__:2027)
[VERIFY] rf_nonlinear_pipeline：未通过，recall=nan；E_RUNTIME ValueError: could not convert string to float: 'C102752' (at __array__:2027)
[VERIFY] 完成，用时 11.4s → ROUTE
[ROUTE] 完成，用时 0.0s → REPLAN
[REPLAN] 第 1 轮 Beam 搜索：父方案 ['lgbm_balanced_pipeline', 'interpretable_logreg_pipeline', 'rf_nonlinear_pipeline']，扩展 6 个，入选 ['logreg_improved_robustness(*TotalChargesCleaner → *TenureBucketizer → StandardFeatureScaler → LogisticRegressionModel)', 'logreg_target_encoding(*TotalChargesCleaner → *KFoldTargetEncoder → StandardFeatureScaler → LogisticRegressionModel)', 'rf_tenure_enhanced(*TotalChargesCleaner → *TenureBucketizer → RandomForestModel)']，剪枝/去重 ['lgbm_tuned_recall_boost(None)', 'rf_nonlinear_cleaned(0.7137)', 'lgbm_basic_fixed(0.7093)']
[REPLAN] 完成，用时 16.1s → BUILD
[BUILD] 生成算子 KFoldTargetEncoder@2（参考 KFoldTargetEncoder@1）：目标编码
[BUILD] 完成，用时 3.5s → VERIFY
[VERIFY] logreg_improved_robustness：未通过，recall=nan；E_RUNTIME ValueError: could not convert string to float: 'C102752' (at __array__:2027)
[VERIFY] logreg_target_encoding：未通过，recall=nan；E_RUNTIME TypeError: cannot concatenate object of type '<class 'numpy.ndarray'>'; only Series and DataFrame objs are valid (at _clean_keys_and_objs:836)
[VERIFY] rf_tenure_enhanced：未通过，recall=nan；E_RUNTIME ValueError: could not convert string to float: 'C102752' (at __array__:2027)
[VERIFY] 完成，用时 4.7s → ROUTE
[ROUTE] 完成，用时 0.0s → REPAIR
[REPAIR] 修复 KFoldTargetEncoder@2 第 1 轮：E_RUNTIME TypeError: cannot concatenate object of type '<class 'numpy.ndarray'>'; only Ser，参考经验：['移除对 pandas 特定 API 的依赖以增强输入类型兼容性', '数值列含空白字符导致类型转换失败', '预测阶段出现未见类别导致编码报错']
[REPAIR] 完成，用时 6.8s → VERIFY
[VERIFY] logreg_target_encoding：未通过，recall=nan；E_RUNTIME KeyError: 'Column not found: None' (at __getitem__:228)
[VERIFY] 完成，用时 1.6s → ROUTE
[ROUTE] 完成，用时 0.0s → REPAIR
[REPAIR] 修复 KFoldTargetEncoder@2 第 2 轮：E_RUNTIME KeyError: 'Column not found: None' (at __getitem__:228)，参考经验：['预测阶段出现未见类别导致编码报错', '数值列含空白字符导致类型转换失败', '移除对 pandas 特定 API 的依赖以增强输入类型兼容性']
[REPAIR] 完成，用时 9.1s → VERIFY
[VERIFY] logreg_target_encoding：未通过，recall=nan；E_RUNTIME KeyError: 'Column not found: None' (at __getitem__:228)
[VERIFY] 完成，用时 1.7s → ROUTE
[ROUTE] 完成，用时 0.0s → REPAIR
[REPAIR] 修复 KFoldTargetEncoder@2 第 3 轮：E_RUNTIME KeyError: 'Column not found: None' (at __getitem__:228)，参考经验：['预测阶段出现未见类别导致编码报错', '数值列含空白字符导致类型转换失败', '移除对 pandas 特定 API 的依赖以增强输入类型兼容性']
[REPAIR] 完成，用时 5.0s → VERIFY
[VERIFY] logreg_target_encoding：未通过，recall=nan；E_RUNTIME ValueError: Input X contains NaN.
LogisticRegression does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to pr (at _assert_all_finite_element_wise:186)
[VERIFY] 完成，用时 1.9s → ROUTE
[ROUTE] 没有方案通过验证
[ROUTE] 完成，用时 0.0s → CURATE
[CURATE] 写入 KG：6 条 ValidationRun；注册新算子 []；新增经验 0 条
[CURATE] 完成，用时 0.0s → FAILED
```

### LLM 调用来源

| 阶段 | 请求哈希 | 来源 | 本次耗时(s) | 录制耗时(s) | tokens |
| --- | --- | --- | --- | --- | --- |
| plan_spec | 235c6bf7467d6da8 | replay | 0.000 | - | 未提供 |
| plan | 7c1e0490d82db23f | remote | 6.576 | - | 10599 |
| synthesize | 6aae7923fca71f2f | remote | 3.616 | - | 2606 |
| synthesize | c2e743c9825f98b7 | remote | 4.157 | - | 2881 |
| synthesize | b48c1b21f6804558 | remote | 2.942 | - | 2660 |
| synthesize | fc284cb9c6506d5b | remote | 3.477 | - | 2602 |
| plan | 98c3efca59796115 | remote | 8.986 | - | 13338 |
| plan | 7a389f8f209fa7b5 | remote | 7.058 | - | 16647 |
| synthesize | 27f062090137bff5 | remote | 3.519 | - | 3175 |
| repair | c86a07fbefaf8ce3 | remote | 6.825 | - | 4722 |
| repair | 8e03973db02a517a | remote | 9.082 | - | 3603 |
| repair | 43ad2f953796a9dc | remote | 4.948 | - | 4184 |
