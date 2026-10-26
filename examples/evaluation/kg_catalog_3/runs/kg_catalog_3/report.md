# AlgoForge 验证报告

| 项目 | 内容 |
| --- | --- |
| 需求 | 预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75 |
| 数据 | data/churn.csv |
| 运行 ID | kg_catalog_3 |
| LLM | gemini-3.5-flash |
| 状态 | ❌ 未通过 |
| 最优方案 | rf_churn_pipeline |
| 总耗时 | 42.1 s |

共验证 1 个候选方案，0 个通过；重新规划 0 次，修复 1 轮。最优方案 `rf_churn_pipeline` 的 recall = 0.495（要求 ≥ 0.75）。

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

每步引用的知识由 Planner 声明，程序检查引用是否在本轮检索结果中、是否适用于该步骤。

| 方案 | 步骤 | 模型引用 | 范围检查 |
| --- | --- | --- | --- |
| rf_churn_pipeline | 1 | 无 | 通过 |
| rf_churn_pipeline | 2 | Operator:MissingValueImputer@1 | 通过 |
| rf_churn_pipeline | 3 | Operator:OneHotCategoricalEncoder@1 | 通过 |
| rf_churn_pipeline | 4 | Operator:RandomForestModel@1 | 通过 |

## 3. 候选方案比较

排序规则：先看是否通过全部检查，再比主指标，再比次要指标之和，最后训练耗时短者优先。指标为多随机种子 × 分层 K 折交叉验证的均值 ± 种子间标准差。

| 排名 | 方案 | 结果 | recall | roc_auc | precision | f1 | recall_at_20 | 单次训练 (s) | 峰值内存 (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ★ rf_churn_pipeline | 未通过 | 0.495 ± 0.003 | 0.694 ± 0.003 | 0.404 ± 0.004 | 0.445 ± 0.001 | 0.326 ± 0.008 | 1.105 | 341 |

```
recall（| 为阈值 0.75）
rf_churn_pipeline                    ■■■■■■■■■■·····|···· 0.495
```

### 各方案的结构与诊断

- `rf_churn_pipeline`：生成:StringNumericCleaner → MissingValueImputer → OneHotCategoricalEncoder → RandomForestModel。诊断：`M_THRESHOLD` recall = 0.4949 低于阈值 0.75

## 4. 最优方案：rf_churn_pipeline

该方案通过专用清洗算子修复字符串数值，使用随机森林处理正负样本不平衡，并利用其特征重要性满足可解释性约束，提高流失预测召回率。

| # | 步骤 | 算子 | 参数 | 选择理由 |
| --- | --- | --- | --- | --- |
| 1 | feature | 生成 StringNumericCleaner | - | 处理 total_charges 列中存在的 string_typed_numeric 问题，保证后续模型输入正确。 |
| 2 | imputer | MissingValueImputer@1 | - | 对 monthly_charges 和 payment_method 等存在的缺失值进行中位数/众数填补。 |
| 3 | encoder | OneHotCategoricalEncoder@1 | max_categories=10 | 将类别型特征转换为数值表示，以适应模型输入需求。 |
| 4 | model | RandomForestModel@1 | class_weight=balanced, n_estimators=500 | 随机森林在处理不平衡数据方面表现稳健，且通过特征重要性提供可解释性，满足召回率要求。 |

## 5. 能力生成与修复

| 生成算子 | 类型 | 用途 | 参考的历史实现 | 修复轮数 | 注册到能力库 |
| --- | --- | --- | --- | --- | --- |
| StringNumericCleaner@1 | feature | 将字符串格式的数值列转换为浮点数，并处理非法字符 | - | 1 | 否 |

### 修复 StringNumericCleaner@1 第 1 轮：✅ 已修复

- 诊断：`E_NAN`（contract）输出引入了新的 NaN/Inf
- 提示：本算子产生的缺失值或无穷值需要在本算子内处理（如 fillna）
- 参考经验：0 条

```diff
--- 修复前
+++ 修复后
@@ -13,3 +13,3 @@
         origin="synthesized",
-        description="将字符串格式的数值列清洗并转换为浮点数，自动处理非法字符和空值",
+        description="将字符串格式的数值列清洗并转换为浮点数，并确保输出无缺失值",
         conditions=["数据包含数值特征被错误读取为字符串"],
@@ -28,2 +28,10 @@
             self.numeric_cols_ = self.columns
+        
+        # 计算每列中位数以便 transform 阶段填充
+        self.medians_ = {}
+        for col in self.numeric_cols_:
+            if col in X.columns:
+                # 尝试将原列转为数值获取中位数，若全为非数值则设为 0
+                temp = pd.to_numeric(X[col], errors='coerce')
+                self.medians_[col] = temp.median() if not temp.isnull().all() else 0.0
         return self
@@ -36,21 +44,19 @@
                 continue
-                
-            # 将列转换为字符串并清除前后空格
+            
+            # 清洗字符串并转数值
             col_data = X[col].astype(str).str.strip()
-            
-            # 替换为空字符串或仅包含空格的字段为 NaN
             col_data = col_data.replace(r'^\s*$', np.nan, regex=True)
             
-            # 使用 pd.to_numeric 将数据转换为 float，将无法解析的非法字符设为 NaN
-            # errors='coerce' 自动处理非数字字符串为缺失值
-            X[col] = pd.to_numeric(col_data, errors='coerce')
+            # 转换为 float，非法值变 NaN
+            converted = pd.to_numeric(col_data, errors='coerce')
             
-            # 补全转换过程中产生的缺失值（采用中位数填充，避免异常值干扰）
-            if X[col].isnull().sum() > 0:
-                median_val = X[col].median()
-                X[col] = X[col].fillna(median_val)
-                
-            # 处理无穷值（若转换产生 inf）
-            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
-            X[col] = X[col].fillna(X[col].median())
+            # 处理 inf
+            converted = converted.replace([np.inf, -np.inf], np.nan)
+            
+            # 填充缺失值
+            fill_val = self.medians_.get(col, 0.0)
+            if np.isnan(fill_val):
+                fill_val = 0.0
+            
+            X[col] = converted.fillna(fill_val)
             
```

## 6. 验证机制

| 级别 | 检查内容 | 本次配置 |
| --- | --- | --- |
| 1 静态检查 | 语法、import 白名单、禁止危险调用、接口与 meta 一致性 | 仅针对生成的算子 |
| 2 沙箱执行 | 独立子进程，限制超时与内存，清除含 KEY/TOKEN 的环境变量 | 超时 180 s，内存 8192 MB |
| 3 契约检查 | 不修改输入、行数一致、不引入 NaN/Inf、概率合法、能处理未见类别与缺失值 | 冒烟样本 300 行 |
| 4 指标检查 | 阈值、种子间稳定性、疑似泄漏、耗时与内存 | 5 折 × 种子 [42, 7, 2024]；std 上限 0.03 |

最优方案资源消耗：单次训练 1.105 s，预测 293.3 ms / 千行，峰值内存 341 MB。

修复前的 1 次验证结果保留在 state.json 的 history 中。

## 7. 知识沉淀

- 回写验证记录 ValidationRun：1 条
- 注册到能力库的新算子：无
- 新提炼的经验：Lesson:298ee062
- 本次检索使用的经验：0 条（命中次数已累计）

## 附录 A. 生成的算法代码（pipeline.py）

```python
# 警告：该方案未通过验证，仅为得分最高的候选，供人工参考。
"""预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75 —— 自动生成的算法流水线

需求：预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75
方案：rf_churn_pipeline
设计依据：该方案通过专用清洗算子修复字符串数值，使用随机森林处理正负样本不平衡，并利用其特征重要性满足可解释性约束，提高流失预测召回率。
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


import pandas as pd
import numpy as np
import re
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class StringNumericCleaner(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="StringNumericCleaner",
        kind=OperatorKind.FEATURE,
        version=1,
        origin="synthesized",
        description="将字符串格式的数值列清洗并转换为浮点数，并确保输出无缺失值",
        conditions=["数据包含数值特征被错误读取为字符串"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"columns": None}
    )

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        # 记录需要处理的列，若未指定则自动识别 object 类型列
        if self.columns is None:
            self.numeric_cols_ = X.select_dtypes(include=['object']).columns.tolist()
        else:
            self.numeric_cols_ = self.columns

        # 计算每列中位数以便 transform 阶段填充
        self.medians_ = {}
        for col in self.numeric_cols_:
            if col in X.columns:
                # 尝试将原列转为数值获取中位数，若全为非数值则设为 0
                temp = pd.to_numeric(X[col], errors='coerce')
                self.medians_[col] = temp.median() if not temp.isnull().all() else 0.0
        return self

    def transform(self, X):
        X = X.copy()

        for col in self.numeric_cols_:
            if col not in X.columns:
                continue

            # 清洗字符串并转数值
            col_data = X[col].astype(str).str.strip()
            col_data = col_data.replace(r'^\s*$', np.nan, regex=True)

            # 转换为 float，非法值变 NaN
            converted = pd.to_numeric(col_data, errors='coerce')

            # 处理 inf
            converted = converted.replace([np.inf, -np.inf], np.nan)

            # 填充缺失值
            fill_val = self.medians_.get(col, 0.0)
            if np.isnan(fill_val):
                fill_val = 0.0

            X[col] = converted.fillna(fill_val)

        return X


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_feature", StringNumericCleaner()),  # 处理 total_charges 列中存在的 string_typed_numeric 问题，保证后续模型输入正确。
        ("1_imputer", MissingValueImputer()),  # 对 monthly_charges 和 payment_method 等存在的缺失值进行中位数/众数填补。
        ("2_encoder", OneHotCategoricalEncoder(max_categories=10)),  # 将类别型特征转换为数值表示，以适应模型输入需求。
        ("3_model", RandomForestModel(n_estimators=500, max_depth=None, class_weight='balanced', random_state=42)),  # 随机森林在处理不平衡数据方面表现稳健，且通过特征重要性提供可解释性，满足召回率要求。
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

各阶段耗时：PLAN 5.9s，BUILD 3.4s，VERIFY 25.9s，ROUTE 0.0s，REPAIR 3.1s，CURATE 3.8s

```
[PLAN] 场景 telecom_churn；画像标签 ['has_missing_values', 'numeric_features', 'has_categorical_features', 'low_cardinality', 'string_typed_numeric', 'id_columns', 'imbalanced', 'time_tenure', 'mixed_features']
[PLAN] TaskSpec：binary_classification，目标 churn，主指标 recall >= 0.75
[PLAN] 候选方案：['rf_churn_pipeline(*StringNumericCleaner → MissingValueImputer → OneHotCategoricalEncoder → RandomForestModel)']
[PLAN] 完成，用时 5.9s → BUILD
[BUILD] 生成算子 StringNumericCleaner@1：将字符串格式的数值列转换为浮点数，并处理非法字符
[BUILD] 完成，用时 3.4s → VERIFY
[VERIFY] rf_churn_pipeline：未通过，recall=nan；E_NAN 输出引入了新的 NaN/Inf
[VERIFY] 完成，用时 1.7s → ROUTE
[ROUTE] 完成，用时 0.0s → REPAIR
[REPAIR] 修复 StringNumericCleaner@1 第 1 轮：E_NAN 输出引入了新的 NaN/Inf
[REPAIR] 完成，用时 3.1s → VERIFY
[VERIFY] rf_churn_pipeline：未通过，recall=0.4949；M_THRESHOLD recall = 0.4949 低于阈值 0.75
[VERIFY] 完成，用时 24.3s → ROUTE
[ROUTE] 没有方案通过验证
[ROUTE] 完成，用时 0.0s → CURATE
[CURATE] 写入 KG：1 条 ValidationRun；注册新算子 []；新增经验 1 条
[CURATE] 完成，用时 3.8s → FAILED
```

### LLM 调用来源

| 阶段 | 请求哈希 | 来源 | 本次耗时(s) | 录制耗时(s) | tokens |
| --- | --- | --- | --- | --- | --- |
| plan | 608e1480ba0a9f85 | remote | 5.732 | - | 4440 |
| synthesize | 5f2faf3c9dde41b9 | remote | 3.356 | - | 2093 |
| repair | 141819c57be5921b | remote | 3.129 | - | 2243 |
| lesson | bcdc29b05636e758 | remote | 3.795 | - | 1987 |
