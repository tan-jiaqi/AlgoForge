# AlgoForge 验证报告

| 项目 | 内容 |
| --- | --- |
| 需求 | 预测30天内可能流失的客户，需可解释，召回优先。比较三个方案，至少一个方案使用清洗、通用缺失值填补、类别编码、标准化和平衡类权重的逻辑回归；可增加在网时长分桶，引用检索到的历史实现。 |
| 数据 | data/churn.csv |
| 运行 ID | 02_churn_synthesize |
| LLM | gemini-3.5-flash |
| 状态 | ✅ 成功 |
| 最优方案 | logistic_regression_pipeline |
| 总耗时 | 23.2 s |

共验证 3 个候选方案，1 个通过；重新规划 0 次，修复 3 轮。最优方案 `logistic_regression_pipeline` 的 recall = 0.778（要求 ≥ 0.75）。

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
| imputer | MissingValueImputer@1 | 是 | 0.642 | 1.00 / 0.09 / 0.72 | 匹配条件：has_missing_values；历史验证 3 次，平均 recall 0.717 |
| imputer | TotalChargesCleaner@1 | 否（历史实现，可参考生成） | 0.450 | 0.92 / 0.27 / 0.00 | 匹配条件：has_missing_values、numeric_features、string_typed_numeric |
| feature | TenureBucketizer@1 | 否（历史实现，可参考生成） | 0.722 | 1.00 / 0.27 / 0.80 | 匹配条件：interpretable、numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | ChargeRatioFeatureGenerator@1 | 否（历史实现，可参考生成） | 0.658 | 0.91 / 0.18 / 0.80 | 匹配条件：numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | ServiceCountFeature@1 | 否（历史实现，可参考生成） | 0.388 | 0.90 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| encoder | OneHotCategoricalEncoder@1 | 是 | 0.607 | 0.84 / 0.18 / 0.72 | 匹配条件：has_categorical_features、low_cardinality；历史验证 3 次，平均 recall 0.717 |
| encoder | KFoldTargetEncoder@1 | 否（历史实现，可参考生成） | 0.427 | 1.00 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| scaler | StandardFeatureScaler@1 | 是 | 0.640 | 1.00 / 0.09 / 0.71 | 匹配条件：numeric_features；历史验证 1 次，平均 recall 0.710 |
| model | BalancedLightGBMClassifier@1 | 否（历史实现，可参考生成） | 0.661 | 1.00 / 0.09 / 0.78 | 匹配条件：imbalanced；历史验证 1 次，平均 recall 0.780 |
| model | LogisticRegressionModel@1 | 是 | 0.613 | 0.93 / 0.09 / 0.71 | 匹配条件：interpretable；历史验证 1 次，平均 recall 0.710 |
| model | LightGBMModel@1 | 是 | 0.539 | 0.68 / 0.09 / 0.80 | 匹配条件：imbalanced；历史验证 1 次，平均 recall 0.800 |

### 参考的历史经验

- **逻辑回归标准化与可解释性**：必须先进行特征标准化（如 StandardFeatureScaler），在 0.1~10 之间搜索正则化强度 C，可直接输出系数给业务方。
- **样本不平衡导致默认模型召回率偏低**（M_THRESHOLD）：启用平衡类权重（如 class_weight='balanced' 或 BalancedLightGBMClassifier）以提升召回率
- **数值列含空白字符导致类型转换失败**（E_RUNTIME）：先 str.strip()，再 pd.to_numeric(errors='coerce')，缺失值填补为 0
- **随机森林防过拟合与召回优化**：设置 n_estimators 300 左右，限制 max_depth 在 8~16 防止过拟合；召回优先时需配合调整分类阈值。
- **预测阶段出现未见类别导致编码报错**（E_RUNTIME）：编码器配置 handle_unknown='ignore' 或 infrequent_if_exist 忽略未知类别

### 知识引用检查

每步引用的知识由 Planner 声明，程序检查引用是否在本轮检索结果中、是否适用于该步骤。

| 方案 | 步骤 | 模型引用 | 范围检查 |
| --- | --- | --- | --- |
| （被拒绝的规划） | - | 无 | 未通过：target_encoded_lgbm_pipeline: 第 3 步：未知算子 BalancedLightGBMClassifier@1；target_encoded_lgbm_pipeline: 第 3 步：BalancedLightGBMClassifier@1 不在本轮检索候选中 |
| logistic_regression_pipeline | 1 | Operator:TotalChargesCleaner@1 | 通过 |
| logistic_regression_pipeline | 2 | Operator:MissingValueImputer@1 | 通过 |
| logistic_regression_pipeline | 3 | Operator:TenureBucketizer@1 | 通过 |
| logistic_regression_pipeline | 4 | Operator:OneHotCategoricalEncoder@1 | 通过 |
| logistic_regression_pipeline | 5 | Operator:StandardFeatureScaler@1 | 通过 |
| logistic_regression_pipeline | 6 | Operator:LogisticRegressionModel@1 | 通过 |
| lgbm_feature_eng_pipeline | 1 | Operator:MissingValueImputer@1 | 通过 |
| lgbm_feature_eng_pipeline | 2 | Operator:ChargeRatioFeatureGenerator@1 | 通过 |
| lgbm_feature_eng_pipeline | 3 | Operator:OneHotCategoricalEncoder@1 | 通过 |
| lgbm_feature_eng_pipeline | 4 | Operator:LightGBMModel@1 | 通过 |
| target_encoded_lgbm_pipeline | 1 | Operator:MissingValueImputer@1 | 通过 |
| target_encoded_lgbm_pipeline | 2 | Operator:KFoldTargetEncoder@1 | 通过 |
| target_encoded_lgbm_pipeline | 3 | Operator:LightGBMModel@1 | 通过 |

## 3. 候选方案比较

排序规则：先看是否通过全部检查，再比主指标，再比次要指标之和，最后训练耗时短者优先。指标为多随机种子 × 分层 K 折交叉验证的均值 ± 种子间标准差。

| 排名 | 方案 | 结果 | recall | roc_auc | precision | f1 | recall_at_20 | 单次训练 (s) | 峰值内存 (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ★ logistic_regression_pipeline | 通过 | 0.778 ± 0.001 | 0.814 ± 0.001 | 0.482 ± 0.001 | 0.595 ± 0.001 | 0.455 ± 0.001 | 0.097 | 261 |
| 2 | target_encoded_lgbm_pipeline | 未通过 | 0.652 ± 0.001 | 0.787 ± 0.002 | 0.495 ± 0.007 | 0.562 ± 0.003 | 0.420 ± 0.008 | 0.372 | 250 |
| 3 | lgbm_feature_eng_pipeline | 未通过 | 0.645 ± 0.004 | 0.787 ± 0.001 | 0.497 ± 0.002 | 0.561 ± 0.002 | 0.422 ± 0.001 | 0.270 | 260 |

```
recall（| 为阈值 0.75）
logistic_regression_pipeline         ■■■■■■■■■■■■■■■|···· 0.778
target_encoded_lgbm_pipeline         ■■■■■■■■■■■■■··|···· 0.652
lgbm_feature_eng_pipeline            ■■■■■■■■■■■■■··|···· 0.645
```

### 各方案的结构与诊断

- `logistic_regression_pipeline`：生成:TotalChargesCleaner → MissingValueImputer → 生成:TenureBucketizer → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel。诊断：无
- `target_encoded_lgbm_pipeline`：MissingValueImputer → 生成:KFoldTargetEncoder → LightGBMModel。诊断：`M_THRESHOLD` recall = 0.6516 低于阈值 0.75；`M_THRESHOLD` roc_auc = 0.7871 低于阈值 0.8
- `lgbm_feature_eng_pipeline`：MissingValueImputer → 生成:ChargeRatioFeatureGenerator → OneHotCategoricalEncoder → LightGBMModel。诊断：`M_THRESHOLD` recall = 0.6446 低于阈值 0.75；`M_THRESHOLD` roc_auc = 0.7866 低于阈值 0.8

## 4. 最优方案：logistic_regression_pipeline

这是满足任务要求的基准方案，使用逻辑回归，包含必要的预处理和样本权重配置。

| # | 步骤 | 算子 | 参数 | 选择理由 |
| --- | --- | --- | --- | --- |
| 1 | imputer | 生成 TotalChargesCleaner | - | 解决数值列被读取为字符串且包含空格导致的问题，引用 Lesson:69d51358。 |
| 2 | imputer | MissingValueImputer@1 | - | 处理其他列中的缺失值。 |
| 3 | feature | 生成 TenureBucketizer | - | 对在网时长进行分桶处理，提高可解释性并提升线性模型表现。 |
| 4 | encoder | OneHotCategoricalEncoder@1 | - | 类别特征 One-Hot 编码。 |
| 5 | scaler | StandardFeatureScaler@1 | - | 逻辑回归必须进行标准化，引用 Lesson:74e2da10。 |
| 6 | model | LogisticRegressionModel@1 | class_weight=balanced | 逻辑回归模型，通过 balanced 权重应对样本不平衡，提升召回率。 |

## 5. 能力生成与修复

| 生成算子 | 类型 | 用途 | 参考的历史实现 | 修复轮数 | 注册到能力库 |
| --- | --- | --- | --- | --- | --- |
| TotalChargesCleaner@2 | imputer | 清理包含空白符的total_charges列并转换为浮点型 | TotalChargesCleaner@1 | 0 | 是 |
| TenureBucketizer@2 | feature | 在网时长分桶以增强线性模型解释性 | TenureBucketizer@1 | 0 | 是 |
| ChargeRatioFeatureGenerator@2 | feature | 生成消费比率特征 | ChargeRatioFeatureGenerator@1 | 3 | 否 |
| KFoldTargetEncoder@2 | encoder | 高基数类别目标编码 | KFoldTargetEncoder@1 | 0 | 否 |

### 修复 ChargeRatioFeatureGenerator@2 第 1 轮：❌ 未修复

- 诊断：`E_RUNTIME`（sandbox）TypeError: operation 'truediv' not supported for dtype 'str' with dtype 'float64' (at _evaluate_op_method:1099)
- 提示：-
- 参考经验：2 条

```diff
--- 修复前
+++ 修复后
@@ -2,5 +2,5 @@
 import pandas as pd
-from sklearn.base import TransformerMixin, ClassifierMixin
+from sklearn.base import TransformerMixin
 from factory.operators.base import Operator, OperatorMeta
-from factory.schemas import OperatorKind
+from factory.schemas import OperatorKind, Modality
 
@@ -38,24 +38,28 @@
         X = X.copy()
+
+        def clean_numeric(series):
+            # 确保列被转换为数值型，非数值字符转为 NaN，然后填充 0
+            if series.dtype == 'object':
+                series = pd.to_numeric(series.astype(str).str.strip(), errors='coerce')
+            return series.fillna(0.0)
+
+        # 获取并清洗数据，确保所有列均为 float 类型
+        monthly_series = clean_numeric(X[self.monthly])
+        total_series = clean_numeric(X[self.total])
+        tenure_series = clean_numeric(X[self.tenure])
+
+        # 计算 avg = total / tenure。处理 tenure=0 的情况（置为 NaN）
+        # 使用 replace 避免除以 0
+        tenure_safe = tenure_series.replace(0, np.nan)
+        avg_series = total_series / tenure_safe
+
+        # 计算 ratio = avg / monthly。处理 monthly=0 的情况
+        monthly_safe = monthly_series.replace(0, np.nan)
+        ratio = avg_series / monthly_safe
+
+        # 处理无穷值和缺失值，保证算子鲁棒性
+        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
         
-        # 处理字符串类型的数值列
-        def clean_numeric(col):
-            if X[col].dtype == 'object':
-                return pd.to_numeric(X[col].astype(str).str.strip(), errors='coerce').fillna(0.0)
-            return X[col].fillna(0.0)
-
-        total_series = clean_numeric(self.total)
-        tenure_series = clean_numeric(self.tenure)
-        monthly_series = clean_numeric(self.monthly)
-
-        # 计算 avg = total / tenure (处理 tenure=0)
-        # 将tenure为0或无效值的情况置为nan，后续作为比值1.0处理
-        avg_series = total_series / tenure_series.replace(0, np.nan)
-        
-        # 计算 ratio = avg / monthly
-        # 处理无穷值和无效值
-        ratio = (avg_series / monthly_series.replace(0, np.nan))
-        X["charge_ratio"] = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
-        
-        # 兜底填充：如果产生缺失值（由于除以0或其他计算问题）
-        X["charge_ratio"] = X["charge_ratio"].fillna(1.0)
+        # 将结果赋值回数据框
+        X["charge_ratio"] = ratio
         
```

### 修复 ChargeRatioFeatureGenerator@2 第 2 轮：❌ 未修复

- 诊断：`E_RUNTIME`（sandbox）TypeError: operation 'truediv' not supported for dtype 'str' with dtype 'float64' (at _evaluate_op_method:1099)
- 提示：-
- 参考经验：2 条

```diff
--- 修复前
+++ 修复后
@@ -40,6 +40,7 @@
         def clean_numeric(series):
-            # 确保列被转换为数值型，非数值字符转为 NaN，然后填充 0
+            # 确保列被转换为数值型，包含空格或特殊字符会被强制转为 NaN
             if series.dtype == 'object':
-                series = pd.to_numeric(series.astype(str).str.strip(), errors='coerce')
-            return series.fillna(0.0)
+                series = series.astype(str).str.strip()
+                series = pd.to_numeric(series, errors='coerce')
+            return series.fillna(0.0).astype(float)
 
@@ -54,3 +55,3 @@
         avg_series = total_series / tenure_safe
-
+        
         # 计算 ratio = avg / monthly。处理 monthly=0 的情况
@@ -59,3 +60,4 @@
 
-        # 处理无穷值和缺失值，保证算子鲁棒性
+        # 处理计算产生的无穷值和缺失值，保证算子鲁棒性
+        # 如果无法计算（例如 tenure 和 monthly 均为 0），默认填充 1.0
         ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
```

### 修复 ChargeRatioFeatureGenerator@2 第 3 轮：✅ 已修复

- 诊断：`E_RUNTIME`（sandbox）ValueError: could not convert string to float: ' ' (at to_numpy:1803)
- 提示：-
- 参考经验：2 条

```diff
--- 修复前
+++ 修复后
@@ -40,9 +40,13 @@
         def clean_numeric(series):
-            # 确保列被转换为数值型，包含空格或特殊字符会被强制转为 NaN
+            # 严格处理：先转字符串，去除空格，再转数值，强制coerce无法转换的为NaN
             if series.dtype == 'object':
                 series = series.astype(str).str.strip()
+                # 排除可能存在的空字符串，将其视为 NaN
+                series = series.replace(r'^\s*$', np.nan, regex=True)
                 series = pd.to_numeric(series, errors='coerce')
-            return series.fillna(0.0).astype(float)
+            else:
+                series = pd.to_numeric(series, errors='coerce')
+            return series.fillna(0.0)
 
-        # 获取并清洗数据，确保所有列均为 float 类型
+        # 获取并清洗数据
         monthly_series = clean_numeric(X[self.monthly])
@@ -51,14 +55,13 @@
 
-        # 计算 avg = total / tenure。处理 tenure=0 的情况（置为 NaN）
-        # 使用 replace 避免除以 0
-        tenure_safe = tenure_series.replace(0, np.nan)
-        avg_series = total_series / tenure_safe
-        
-        # 计算 ratio = avg / monthly。处理 monthly=0 的情况
-        monthly_safe = monthly_series.replace(0, np.nan)
-        ratio = avg_series / monthly_safe
+        # 计算 avg = total / tenure。
+        # tenure 为 0 时结果为 inf，后续统一处理
+        with np.errstate(divide='ignore', invalid='ignore'):
+            avg_series = total_series / tenure_series
+            # 计算 ratio = avg / monthly
+            ratio = avg_series / monthly_series
 
-        # 处理计算产生的无穷值和缺失值，保证算子鲁棒性
-        # 如果无法计算（例如 tenure 和 monthly 均为 0），默认填充 1.0
-        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
+        # 处理 inf, -inf 和 NaN，保证鲁棒性
+        # 如果无法计算（例如 tenure 或 monthly 均为 0），默认填充 1.0 代表比率持平
+        ratio = ratio.replace([np.inf, -np.inf], np.nan)
+        ratio = ratio.fillna(1.0)
         
```

## 6. 验证机制

| 级别 | 检查内容 | 本次配置 |
| --- | --- | --- |
| 1 静态检查 | 语法、import 白名单、禁止危险调用、接口与 meta 一致性 | 仅针对生成的算子 |
| 2 沙箱执行 | 独立子进程，限制超时与内存，清除含 KEY/TOKEN 的环境变量 | 超时 180 s，内存 8192 MB |
| 3 契约检查 | 不修改输入、行数一致、不引入 NaN/Inf、概率合法、能处理未见类别与缺失值 | 冒烟样本 300 行 |
| 4 指标检查 | 阈值、种子间稳定性、疑似泄漏、耗时与内存 | 5 折 × 种子 [42, 7, 2024]；std 上限 0.03 |

最优方案资源消耗：单次训练 0.097 s，预测 47.2 ms / 千行，峰值内存 261 MB。

修复前的 3 次验证结果保留在 state.json 的 history 中。

## 7. 知识沉淀

- 回写验证记录 ValidationRun：3 条
- 注册到能力库的新算子：TotalChargesCleaner@2、TenureBucketizer@2
- 新提炼的经验：Lesson:a3a47067
- 本次检索使用的经验：7 条（命中次数已累计）

## 附录 A. 生成的算法代码（pipeline.py）

```python
"""预测30天内可能流失的客户，需可解释，召回优先。比较三个方案，至少一个方案使用清 —— 自动生成的算法流水线

需求：预测30天内可能流失的客户，需可解释，召回优先。比较三个方案，至少一个方案使用清洗、通用缺失值填补、类别编码、标准化和平衡类权重的逻辑回归；可增加在网时长分桶，引用检索到的历史实现。
方案：logistic_regression_pipeline
设计依据：这是满足任务要求的基准方案，使用逻辑回归，包含必要的预处理和样本权重配置。
主指标：recall >= 0.75

由 AlgoForge 自动生成。用法：
    python pipeline.py --data <train.csv>                      # 训练并保存 model.joblib
    python pipeline.py --predict <new.csv> --model model.joblib  # 预测
"""

import argparse

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from factory.operators.models import LogisticRegressionModel
from factory.operators.tabular import MissingValueImputer, OneHotCategoricalEncoder, StandardFeatureScaler


import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind, Modality

class TotalChargesCleaner(TransformerMixin, Operator):
    """
    清理包含空白符或脏字符的total_charges列并转换为浮点型，缺失值默认为0。
    """
    meta = OperatorMeta(
        name="TotalChargesCleaner",
        kind=OperatorKind.IMPUTER,
        version=2,
        origin="synthesized",
        description="清理包含空白符的total_charges列并转换为浮点型",
        conditions=["数据存在字符串格式的数值列", "存在包含空白字符的缺失值"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"col": "total_charges"},
        parent="TotalChargesCleaner@1"
    )

    def __init__(self, col="total_charges"):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.col in X.columns:
            # 根据Lesson:69d51358经验：先strip，再转数值（强制转换空值），最后填充缺失值
            X[self.col] = pd.to_numeric(
                X[self.col].astype(str).str.strip(), 
                errors="coerce"
            ).fillna(0.0)

            # 确保转换后没有无穷值，保持特征稳定性
            X[self.col] = X[self.col].replace([np.inf, -np.inf], 0.0)

        return X

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class TenureBucketizer(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="TenureBucketizer",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="将数值型在网时长特征划分为业务导向的区间，增强线性模型的解释性。",
        conditions=["输入必须包含数值型tenure列"],
        dependencies=["numpy", "pandas", "sklearn"],
        params={"col": "tenure"},
        parent="TenureBucketizer@1"
    )

    def __init__(self, col="tenure"):
        self.col = col

    def fit(self, X, y=None):
        """记录算子状态，本算子为无状态转换，仅返回self。"""
        self.bins_ = [-1, 6, 12, 24, 48, np.inf]
        self.labels_ = ["0-6", "6-12", "12-24", "24-48", "48+"]
        return self

    def transform(self, X):
        """
        将tenure列分桶，处理缺失值及非数值情况。
        遵循数据清洗规范：先处理潜在脏数据，确保输出类型一致。
        """
        X = X.copy()

        # 处理脏数据：先确保转为数值，无法转换的(含空值)处理为-1（即处于0-6区间内）
        raw_tenure = pd.to_numeric(X[self.col].astype(str).str.strip(), errors='coerce')
        raw_tenure = raw_tenure.fillna(-1)

        # 分桶操作
        bucket_col = f"{self.col}_bucket"
        X[bucket_col] = pd.cut(
            raw_tenure, 
            bins=self.bins_, 
            labels=self.labels_
        ).astype(str)

        # 确保分桶后的缺失值被处理（pd.cut可能在NaN输入下产生nan字符串）
        X[bucket_col] = X[bucket_col].replace("nan", "0-6")

        return X


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_imputer", TotalChargesCleaner()),  # 解决数值列被读取为字符串且包含空格导致的问题，引用 Lesson:69d51358。
        ("1_imputer", MissingValueImputer()),  # 处理其他列中的缺失值。
        ("2_feature", TenureBucketizer()),  # 对在网时长进行分桶处理，提高可解释性并提升线性模型表现。
        ("3_encoder", OneHotCategoricalEncoder(max_categories=20)),  # 类别特征 One-Hot 编码。
        ("4_scaler", StandardFeatureScaler()),  # 逻辑回归必须进行标准化，引用 Lesson:74e2da10。
        ("5_model", LogisticRegressionModel(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)),  # 逻辑回归模型，通过 balanced 权重应对样本不平衡，提升召回率。
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

各阶段耗时：PLAN 0.2s，BUILD 0.0s，VERIFY 22.9s，ROUTE 0.0s，REPAIR 0.0s，CURATE 0.0s

```
[PLAN] 场景 telecom_churn；画像标签 ['has_missing_values', 'numeric_features', 'has_categorical_features', 'low_cardinality', 'string_typed_numeric', 'id_columns', 'imbalanced', 'time_tenure', 'mixed_features']
[PLAN] TaskSpec：binary_classification，目标 churn，主指标 recall >= 0.75
[PLAN] 候选方案：['logistic_regression_pipeline(*TotalChargesCleaner → MissingValueImputer → *TenureBucketizer → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel)', 'lgbm_feature_eng_pipeline(MissingValueImputer → *ChargeRatioFeatureGenerator → OneHotCategoricalEncoder → LightGBMModel)', 'target_encoded_lgbm_pipeline(MissingValueImputer → *KFoldTargetEncoder → LightGBMModel)']
[PLAN] 完成，用时 0.2s → BUILD
[BUILD] 生成算子 TotalChargesCleaner@2（参考 TotalChargesCleaner@1）：清理包含空白符的total_charges列并转换为浮点型
[BUILD] 生成算子 TenureBucketizer@2（参考 TenureBucketizer@1）：在网时长分桶以增强线性模型解释性
[BUILD] 生成算子 ChargeRatioFeatureGenerator@2（参考 ChargeRatioFeatureGenerator@1）：生成消费比率特征
[BUILD] 生成算子 KFoldTargetEncoder@2（参考 KFoldTargetEncoder@1）：高基数类别目标编码
[BUILD] 完成，用时 0.0s → VERIFY
[VERIFY] logistic_regression_pipeline：通过，recall=0.7778
[VERIFY] lgbm_feature_eng_pipeline：未通过，recall=nan；E_RUNTIME TypeError: operation 'truediv' not supported for dtype 'str' with dtype 'float64' (at _evaluate_op_method:1099)
[VERIFY] target_encoded_lgbm_pipeline：未通过，recall=0.6516；M_THRESHOLD recall = 0.6516 低于阈值 0.75；M_THRESHOLD roc_auc = 0.7871 低于阈值 0.8
[VERIFY] 完成，用时 13.3s → ROUTE
[ROUTE] 完成，用时 0.0s → REPAIR
[REPAIR] 修复 ChargeRatioFeatureGenerator@2 第 1 轮：E_RUNTIME TypeError: operation 'truediv' not supported for dtype 'str' with dtype 'float64，参考经验：['数值列含空白字符导致类型转换失败', '预测阶段出现未见类别导致编码报错']
[REPAIR] 完成，用时 0.0s → VERIFY
[VERIFY] lgbm_feature_eng_pipeline：未通过，recall=nan；E_RUNTIME TypeError: operation 'truediv' not supported for dtype 'str' with dtype 'float64' (at _evaluate_op_method:1099)
[VERIFY] 完成，用时 1.4s → ROUTE
[ROUTE] 完成，用时 0.0s → REPAIR
[REPAIR] 修复 ChargeRatioFeatureGenerator@2 第 2 轮：E_RUNTIME TypeError: operation 'truediv' not supported for dtype 'str' with dtype 'float64，参考经验：['数值列含空白字符导致类型转换失败', '预测阶段出现未见类别导致编码报错']
[REPAIR] 完成，用时 0.0s → VERIFY
[VERIFY] lgbm_feature_eng_pipeline：未通过，recall=nan；E_RUNTIME ValueError: could not convert string to float: ' ' (at to_numpy:1803)
[VERIFY] 完成，用时 1.5s → ROUTE
[ROUTE] 完成，用时 0.0s → REPAIR
[REPAIR] 修复 ChargeRatioFeatureGenerator@2 第 3 轮：E_RUNTIME ValueError: could not convert string to float: ' ' (at to_numpy:1803)，参考经验：['数值列含空白字符导致类型转换失败', '预测阶段出现未见类别导致编码报错']
[REPAIR] 完成，用时 0.0s → VERIFY
[VERIFY] lgbm_feature_eng_pipeline：未通过，recall=0.6446；M_THRESHOLD recall = 0.6446 低于阈值 0.75；M_THRESHOLD roc_auc = 0.7866 低于阈值 0.8
[VERIFY] 完成，用时 6.7s → ROUTE
[ROUTE] 完成，用时 0.0s → CURATE
[CURATE] 写入 KG：3 条 ValidationRun；注册新算子 ['TotalChargesCleaner@2', 'TenureBucketizer@2']；新增经验 1 条
[CURATE] 完成，用时 0.0s → DONE
```

### LLM 调用来源

| 阶段 | 请求哈希 | 来源 | 本次耗时(s) | 录制耗时(s) | tokens |
| --- | --- | --- | --- | --- | --- |
| plan_spec | 912f4bae786b0895 | replay | 0.000 | 2.223 | 2047 |
| plan | 75f8c42598221c16 | replay | 0.000 | 4.408 | 10584 |
| plan | 5278bb5dea3a9768 | replay | 0.000 | 6.964 | 12374 |
| synthesize | cf3848ed55768030 | replay | 0.000 | 2.337 | 2519 |
| synthesize | 9f697fa8e20dc4ce | replay | 0.000 | 3.632 | 2717 |
| synthesize | 23409b4b115cd5ba | replay | 0.000 | 2.978 | 2766 |
| synthesize | f523e2062d16bb57 | replay | 0.000 | 2.896 | 3176 |
| repair | c0090a2c28f26c6e | replay | 0.000 | 5.585 | 2666 |
| repair | e686ae5364ad0a14 | replay | 0.001 | 3.263 | 2734 |
| repair | bafbd585da4216df | replay | 0.001 | 2.703 | 2777 |
| lesson | ca8c449f1d6cb4d4 | replay | 0.000 | 2.282 | 1919 |
