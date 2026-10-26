# AlgoForge 验证报告

| 项目 | 内容 |
| --- | --- |
| 需求 | 预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75 |
| 数据 | data/churn.csv |
| 运行 ID | kg_catalog_2 |
| LLM | gemini-3.5-flash |
| 状态 | ❌ 未通过 |
| 最优方案 | random_forest_churn_prediction |
| 总耗时 | 33.4 s |

共验证 1 个候选方案，0 个通过；重新规划 0 次，修复 1 轮。最优方案 `random_forest_churn_prediction` 的 recall = 0.597（要求 ≥ 0.75）。

## 1. 需求理解（TaskSpec）

| 字段 | 值 |
| --- | --- |
| 任务类型 | binary_classification |
| 数据模态 | tabular |
| 目标列 | churn |
| 评价指标 | recall ≥ 0.75（主） |
| 业务约束 | interpretable、recall_first |
| 数据画像标签 | has_missing_values、numeric_features、has_categorical_features、low_cardinality、string_typed_numeric、id_columns、imbalanced、time_tenure、mixed_features |

## 2. 知识检索（设计依据）

匹配场景：telecom_churn。每类步骤从知识图谱检索候选算子，得分 = 0.4 × 文本相关度 + 0.3 × 条件匹配 + 0.3 × 历史验证分。

| 步骤 | 候选算子 | 可直接复用 | 得分 | 文本/条件/历史 | 证据 |
| --- | --- | --- | --- | --- | --- |
| model | LightGBMModel@1 | 是 | 0.000 | 0.00 / 0.00 / 0.00 | - |
| model | LogisticRegressionModel@1 | 是 | 0.000 | 0.00 / 0.00 / 0.00 | - |
| model | RandomForestModel@1 | 是 | 0.000 | 0.00 / 0.00 / 0.00 | - |
| imputer | MissingValueImputer@1 | 是 | 0.000 | 0.00 / 0.00 / 0.00 | - |
| encoder | OneHotCategoricalEncoder@1 | 是 | 0.000 | 0.00 / 0.00 / 0.00 | - |
| scaler | StandardFeatureScaler@1 | 是 | 0.000 | 0.00 / 0.00 / 0.00 | - |

### 知识引用检查

检索并传入上下文不等于采纳。以下引用由模型声明，程序仅核对可见范围与结构化适用对象；不证明语义落实或指标收益。

| 方案 | 步骤 | 模型引用 | 范围检查 | 语义落实 |
| --- | --- | --- | --- | --- |
| random_forest_churn_prediction | 1 | 无 | 通过 | 未验证 |
| random_forest_churn_prediction | 2 | Operator:MissingValueImputer@1 | 通过 | 未验证 |
| random_forest_churn_prediction | 3 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| random_forest_churn_prediction | 4 | Operator:RandomForestModel@1 | 通过 | 未验证 |

## 3. 候选方案比较

排序规则：先看是否通过全部检查，再比主指标，再比次要指标之和，最后训练耗时短者优先。指标为多随机种子 × 分层 K 折交叉验证的均值 ± 种子间标准差。

| 排名 | 方案 | 结果 | recall | roc_auc | precision | f1 | recall_at_20 | 单次训练 (s) | 峰值内存 (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ★ random_forest_churn_prediction | 未通过 | 0.597 ± 0.002 | 0.787 ± 0.001 | 0.517 ± 0.003 | 0.554 ± 0.001 | 0.431 ± 0.006 | 0.641 | 305 |

```
recall（| 为阈值 0.75）
random_forest_churn_prediction       ■■■■■■■■■■■■···|···· 0.597
```

### 各方案的结构与诊断

- `random_forest_churn_prediction`：生成:DataPreprocessor → MissingValueImputer → OneHotCategoricalEncoder → RandomForestModel。诊断：`M_THRESHOLD` recall = 0.5969 低于阈值 0.75

## 4. 最优方案：random_forest_churn_prediction

通过预处理清理脏数据，使用随机森林模型平衡召回率与可解释性需求，并针对不平衡数据使用平衡权重策略。

| # | 步骤 | 算子 | 参数 | 选择理由 |
| --- | --- | --- | --- | --- |
| 1 | feature | 生成 DataPreprocessor | - | 需要处理客户ID列带来的干扰以及 total_charges 列的字符串格式问题，这是数据预处理的第一步。 |
| 2 | imputer | MissingValueImputer@1 | - | 针对 payment_method 和 monthly_charges 等字段的缺失值进行中位数/众数填补。 |
| 3 | encoder | OneHotCategoricalEncoder@1 | - | 将类别型特征转换为数值型以供模型读取，利用其对低基数特征的处理能力。 |
| 4 | model | RandomForestModel@1 | class_weight=balanced, n_estimators=300 | 随机森林在处理表格数据时稳健性好，且可通过特征重要性满足可解释性要求；class_weight='balanced' 针对不平衡数据进行优化，以提升召回率。 |

## 5. 能力生成与修复

| 生成算子 | 类型 | 用途 | 参考的历史实现 | 修复轮数 | 注册到能力库 |
| --- | --- | --- | --- | --- | --- |
| DataPreprocessor@1 | feature | 删除无关的客户标识列，并将字符串形式的数值列转换为浮点数以供模型处理。 | - | 1 | 否 |

### 修复 DataPreprocessor@1 第 1 轮：✅ 已修复

- 诊断：`E_RUNTIME`（sandbox）TypeError: Cannot interpret '<StringDtype(na_value=nan)>' as a data type (at issubdtype:534)
- 提示：-
- 参考经验：0 条

```diff
--- 修复前
+++ 修复后
@@ -26,5 +26,5 @@
         for col in self.numeric_cols:
-            # 强制转换为数值，非数值变为NaN
-            series = pd.to_numeric(X[col], errors='coerce')
-            self.medians_[col] = series.median()
+            if col in X.columns:
+                series = pd.to_numeric(X[col], errors='coerce')
+                self.medians_[col] = series.median() if not series.isna().all() else 0.0
         return self
@@ -37,18 +37,17 @@
         
-        # 2. 转换字符串数值列
+        # 2. 转换指定数值列
         for col in self.numeric_cols:
             if col in X.columns:
-                # 转换类型
                 X[col] = pd.to_numeric(X[col], errors='coerce')
-                # 处理产生的缺失值和无穷值
                 X[col] = X[col].replace([np.inf, -np.inf], np.nan)
-                X[col] = X[col].fillna(self.medians_.get(col, 0))
+                X[col] = X[col].fillna(self.medians_.get(col, 0.0))
         
-        # 3. 对整体DataFrame存在的其他缺失值进行处理（兜底）
-        # 数值列填中位数，非数值列填"Missing"
+        # 3. 对整体DataFrame其他列处理（规避复杂dtype导致issubdtype报错）
         for col in X.columns:
             if X[col].isnull().any():
-                if np.issubdtype(X[col].dtype, np.number):
-                    X[col] = X[col].fillna(X[col].median())
+                # 使用类型检查而非复杂的issubdtype，优先处理数值列
+                if pd.api.types.is_numeric_dtype(X[col]):
+                    X[col] = X[col].fillna(X[col].median() if not X[col].dropna().empty else 0.0)
                 else:
+                    # 显式转换为对象类型以处理复杂类型缺失
                     X[col] = X[col].fillna("Missing").astype(str)
```

## 6. 验证机制

| 级别 | 检查内容 | 本次配置 |
| --- | --- | --- |
| 1 静态检查 | 语法、import 白名单、禁止危险调用、接口与 meta 一致性 | 仅针对生成的算子 |
| 2 沙箱执行 | 独立子进程，限制超时与内存，清除含 KEY/TOKEN 的环境变量 | 超时 180 s，内存 8192 MB |
| 3 契约检查 | 不修改输入、行数一致、不引入 NaN/Inf、概率合法、能处理未见类别与缺失值 | 冒烟样本 300 行 |
| 4 指标检查 | 阈值、种子间稳定性、疑似泄漏、耗时与内存 | 5 折 × 种子 [42, 7, 2024]；std 上限 0.03 |

最优方案资源消耗：单次训练 0.641 s，预测 169.5 ms / 千行，峰值内存 305 MB。

修复前的 1 次验证结果保留在 state.json 的 history 中。

## 7. 知识沉淀

- 回写验证记录 ValidationRun：1 条
- 注册到能力库的新算子：无
- 新提炼的经验：避免在 Pandas 扩展类型上使用 np.issubdtype 以防类型检查报错（E_RUNTIME）
- 本次提供的经验：0 条（累计检索命中，不代表采纳）

## 附录 A. 生成的算法代码（pipeline.py）

```python
# 警告：该方案未通过验证，仅为得分最高的候选，供人工参考。
"""预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75 —— 自动生成的算法流水线

需求：预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75
方案：random_forest_churn_prediction
设计依据：通过预处理清理脏数据，使用随机森林模型平衡召回率与可解释性需求，并针对不平衡数据使用平衡权重策略。
主指标：recall >= 0.75

由 AlgoForge 自动生成。用法：
    python pipeline.py --data <train.csv>                      # 训练并保存 model.joblib
    python pipeline.py --predict <new.csv> --model model.joblib  # 预测
"""

import argparse

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from factory.operators.models import RandomForestModel
from factory.operators.tabular import MissingValueImputer, OneHotCategoricalEncoder


import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class DataPreprocessor(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="DataPreprocessor",
        kind=OperatorKind.FEATURE,
        version=1,
        origin="synthesized",
        description="删除标识列并处理字符串格式的数值特征，同时填补缺失值。",
        conditions=["输入包含customer_id及字符串数值列"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"id_columns": ["customer_id"], "numeric_cols": ["total_charges"]}
    )

    def __init__(self, id_columns=["customer_id"], numeric_cols=["total_charges"]):
        self.id_columns = id_columns
        self.numeric_cols = numeric_cols

    def fit(self, X, y=None):
        X = X.copy()
        self.medians_ = {}
        for col in self.numeric_cols:
            if col in X.columns:
                series = pd.to_numeric(X[col], errors='coerce')
                self.medians_[col] = series.median() if not series.isna().all() else 0.0
        return self

    def transform(self, X):
        X = X.copy()

        # 1. 删除标识列
        X = X.drop(columns=[c for c in self.id_columns if c in X.columns], errors='ignore')

        # 2. 转换指定数值列
        for col in self.numeric_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].replace([np.inf, -np.inf], np.nan)
                X[col] = X[col].fillna(self.medians_.get(col, 0.0))

        # 3. 对整体DataFrame其他列处理（规避复杂dtype导致issubdtype报错）
        for col in X.columns:
            if X[col].isnull().any():
                # 使用类型检查而非复杂的issubdtype，优先处理数值列
                if pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = X[col].fillna(X[col].median() if not X[col].dropna().empty else 0.0)
                else:
                    # 显式转换为对象类型以处理复杂类型缺失
                    X[col] = X[col].fillna("Missing").astype(str)

        return X


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_feature", DataPreprocessor()),  # 需要处理客户ID列带来的干扰以及 total_charges 列的字符串格式问题，这是数据预处理的第一步。
        ("1_imputer", MissingValueImputer()),  # 针对 payment_method 和 monthly_charges 等字段的缺失值进行中位数/众数填补。
        ("2_encoder", OneHotCategoricalEncoder(max_categories=20)),  # 将类别型特征转换为数值型以供模型读取，利用其对低基数特征的处理能力。
        ("3_model", RandomForestModel(n_estimators=300, max_depth=None, class_weight='balanced', random_state=42)),  # 随机森林在处理表格数据时稳健性好，且可通过特征重要性满足可解释性要求；class_weight='balanced' 针对不平衡数据进行优化，以提升召回率。
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

各阶段耗时：PLAN 5.4s，BUILD 2.8s，VERIFY 16.9s，ROUTE 0.0s，REPAIR 2.8s，CURATE 5.5s

```
[PLAN] 场景 telecom_churn；画像标签 ['has_missing_values', 'numeric_features', 'has_categorical_features', 'low_cardinality', 'string_typed_numeric', 'id_columns', 'imbalanced', 'time_tenure', 'mixed_features']
[PLAN] TaskSpec：binary_classification，目标 churn，主指标 recall >= 0.75
[PLAN] 候选方案：['random_forest_churn_prediction(*DataPreprocessor → MissingValueImputer → OneHotCategoricalEncoder → RandomForestModel)']
[PLAN] 完成，用时 5.4s → BUILD
[BUILD] 生成算子 DataPreprocessor@1：删除无关的客户标识列，并将字符串形式的数值列转换为浮点数以供模型处理。
[BUILD] 完成，用时 2.8s → VERIFY
[VERIFY] random_forest_churn_prediction：未通过，recall=nan；E_RUNTIME TypeError: Cannot interpret '<StringDtype(na_value=nan)>' as a data type (at issubdtype:534)
[VERIFY] 完成，用时 1.6s → ROUTE
[ROUTE] 完成，用时 0.0s → REPAIR
[REPAIR] 修复 DataPreprocessor@1 第 1 轮：E_RUNTIME TypeError: Cannot interpret '<StringDtype(na_value=nan)>' as a data type (at iss
[REPAIR] 完成，用时 2.8s → VERIFY
[VERIFY] random_forest_churn_prediction：未通过，recall=0.5969；M_THRESHOLD recall = 0.5969 低于阈值 0.75
[VERIFY] 完成，用时 15.3s → ROUTE
[ROUTE] 没有方案通过验证
[ROUTE] 完成，用时 0.0s → CURATE
[CURATE] 写入 KG：1 条 ValidationRun；注册新算子 []；新增经验 1 条
[CURATE] 完成，用时 5.5s → FAILED
```

### LLM 调用来源

| 阶段 | 请求哈希 | 来源 | 本次耗时(s) | 录制耗时(s) | tokens |
| --- | --- | --- | --- | --- | --- |
| plan | 608e1480ba0a9f85 | remote | 5.204 | - | 5025 |
| synthesize | 7848c537ade02648 | remote | 2.822 | - | 2140 |
| repair | 2d6f2ef140c0d353 | remote | 2.769 | - | 2339 |
| lesson | dd4b9b036247aff3 | remote | 5.455 | - | 2318 |
