# AlgoForge 验证报告

| 项目 | 内容 |
| --- | --- |
| 需求 | 预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75；正确处理累计费用的空白字符串，并增加在网时长分桶特征。 |
| 数据 | data/churn.csv |
| 运行 ID | repair_injected_3 |
| LLM | gemini-3.5-flash |
| 状态 | ✅ 成功 |
| 最优方案 | interpretable_logreg_pipeline |
| 总耗时 | 15.4 s |

共验证 1 个候选方案，1 个通过；重新规划 0 次，修复 1 轮。最优方案 `interpretable_logreg_pipeline` 的 recall = 0.778（要求 ≥ 0.75）。

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
| imputer | TotalChargesCleaner@1 | 否（历史实现，可参考生成） | 0.482 | 1.00 / 0.27 / 0.00 | 匹配条件：has_missing_values、numeric_features、string_typed_numeric |
| imputer | MissingValueImputer@1 | 是 | 0.481 | 0.60 / 0.09 / 0.72 | 匹配条件：has_missing_values；历史验证 3 次，平均 recall 0.717 |
| feature | TenureBucketizer@1 | 否（历史实现，可参考生成） | 0.722 | 1.00 / 0.27 / 0.80 | 匹配条件：interpretable、numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | ChargeRatioFeatureGenerator@1 | 否（历史实现，可参考生成） | 0.622 | 0.82 / 0.18 / 0.80 | 匹配条件：numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | ServiceCountFeature@1 | 否（历史实现，可参考生成） | 0.375 | 0.87 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| feature | IdColumnDropper@1 | 否（历史实现，可参考生成） | 0.344 | 0.79 / 0.09 / 0.00 | 匹配条件：id_columns |
| encoder | OneHotCategoricalEncoder@1 | 是 | 0.574 | 0.76 / 0.18 / 0.72 | 匹配条件：has_categorical_features、low_cardinality；历史验证 3 次，平均 recall 0.717 |
| encoder | KFoldTargetEncoder@1 | 否（历史实现，可参考生成） | 0.427 | 1.00 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| scaler | StandardFeatureScaler@1 | 是 | 0.640 | 1.00 / 0.09 / 0.71 | 匹配条件：numeric_features；历史验证 1 次，平均 recall 0.710 |
| model | LogisticRegressionModel@1 | 是 | 0.640 | 1.00 / 0.09 / 0.71 | 匹配条件：interpretable；历史验证 1 次，平均 recall 0.710 |
| model | BalancedLightGBMClassifier@1 | 否（历史实现，可参考生成） | 0.625 | 0.91 / 0.09 / 0.78 | 匹配条件：imbalanced；历史验证 1 次，平均 recall 0.780 |
| model | RandomForestModel@1 | 是 | 0.566 | 0.85 / 0.09 / 0.66 | 匹配条件：mixed_features；历史验证 1 次，平均 recall 0.660 |
| model | LightGBMModel@1 | 是 | 0.541 | 0.68 / 0.09 / 0.80 | 匹配条件：imbalanced；历史验证 1 次，平均 recall 0.800 |

### 参考的历史经验

- **数值列含空白字符导致类型转换失败**（E_RUNTIME）：先 str.strip()，再 pd.to_numeric(errors='coerce')，缺失值填补为 0
- **样本不平衡导致默认模型召回率偏低**（M_THRESHOLD）：启用平衡类权重（如 class_weight='balanced' 或 BalancedLightGBMClassifier）以提升召回率
- **逻辑回归标准化与可解释性**：必须先进行特征标准化（如 StandardFeatureScaler），在 0.1~10 之间搜索正则化强度 C，可直接输出系数给业务方。
- **随机森林防过拟合与召回优化**：设置 n_estimators 300 左右，限制 max_depth 在 8~16 防止过拟合；召回优先时需配合调整分类阈值。
- **未剔除客户 ID 导致 One-Hot 编码后维度暴涨与内存溢出**（E_TIMEOUT）：建模前剔除 ID 等标识列，并在编码器中设置最大类别数

## 3. 候选方案比较

排序规则：先看是否通过全部检查，再比主指标，再比次要指标之和，最后训练耗时短者优先。指标为多随机种子 × 分层 K 折交叉验证的均值 ± 种子间标准差。

| 排名 | 方案 | 结果 | recall | roc_auc | precision | f1 | recall_at_20 | 单次训练 (s) | 峰值内存 (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ★ interpretable_logreg_pipeline | 通过 | 0.778 ± 0.001 | 0.813 ± 0.001 | 0.481 ± 0.001 | 0.595 ± 0.001 | 0.454 ± 0.002 | 0.105 | 259 |

```
recall（| 为阈值 0.75）
interpretable_logreg_pipeline        ■■■■■■■■■■■■■■■|···· 0.778
```

### 各方案的结构与诊断

- `interpretable_logreg_pipeline`：生成:TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel。诊断：无

## 4. 最优方案：interpretable_logreg_pipeline

该方案侧重于模型的可解释性，符合业务合规约束。

| # | 步骤 | 算子 | 参数 | 选择理由 |
| --- | --- | --- | --- | --- |
| 1 | imputer | 生成 TotalChargesCleaner | - | 清理数值列，参考 Lesson:69d51358。 |
| 2 | imputer | MissingValueImputer@1 | - | 补充缺失数据。 |
| 3 | encoder | OneHotCategoricalEncoder@1 | max_categories=20 | 线性模型需要编码化类别特征。 |
| 4 | scaler | StandardFeatureScaler@1 | - | 逻辑回归必须先标准化，参考 Lesson:74e2da10。 |
| 5 | model | LogisticRegressionModel@1 | class_weight=balanced | 逻辑回归满足可解释性需求。 |

## 5. 能力生成与修复

| 生成算子 | 类型 | 用途 | 参考的历史实现 | 修复轮数 | 注册到能力库 |
| --- | --- | --- | --- | --- | --- |
| TotalChargesCleaner@2 | imputer | 处理总费用列的缺失值与数据格式问题 | TotalChargesCleaner@1 | 1 | 是 |

### 修复 TotalChargesCleaner@2 第 1 轮：✅ 已修复

- 诊断：`E_NAN`（contract）输出引入了新的 NaN/Inf
- 提示：本算子产生的缺失值或无穷值需要在本算子内处理（如 fillna）
- 参考经验：0 条

```diff
--- 修复前
+++ 修复后
@@ -7,5 +7,15 @@
 class TotalChargesCleaner(TransformerMixin, Operator):
-    meta = OperatorMeta(name='TotalChargesCleaner', kind=OperatorKind.IMPUTER, version=2, origin='synthesized', description='将字符串类型的总费用列转换为数值，处理空白异常值并增加在网时长分桶特征', conditions=['数据包含total_charges列', '存在tenure在网时长列'], dependencies=['pandas', 'numpy', 'sklearn'], params={'col': 'total_charges', 'tenure_col': 'tenure'}, parent='TotalChargesCleaner@1')
+    meta = OperatorMeta(
+        name="TotalChargesCleaner",
+        kind=OperatorKind.IMPUTER,
+        version=2,
+        origin="synthesized",
+        description="将字符串类型的总费用列转换为数值，处理空白异常值并增加在网时长分桶特征",
+        conditions=["数据包含total_charges列", "存在tenure在网时长列"],
+        dependencies=["pandas", "numpy", "sklearn"],
+        params={"col": "total_charges", "tenure_col": "tenure"},
+        parent="TotalChargesCleaner@1"
+    )
 
-    def __init__(self, col='total_charges', tenure_col='tenure'):
+    def __init__(self, col="total_charges", tenure_col="tenure"):
         self.col = col
@@ -14,3 +24,4 @@
     def fit(self, X, y=None):
-        self.mean_total_charges_ = 0.0
+        # 记录清洗后的均值或固定值用于兜底
+        self.mean_val_ = 0.0
         return self
@@ -18,13 +29,26 @@
     def transform(self, X):
-        _evaluation_raw = X['total_charges'].copy()
         X = X.copy()
+        
+        # 处理 total_charges 列
         if self.col in X.columns:
-            X[self.col] = pd.to_numeric(X[self.col].astype(str).str.strip(), errors='coerce').fillna(0.0)
-            X[self.col] = X[self.col].replace([np.inf, -np.inf], 0.0)
+            # 强制转换为数值，空白及非法字符变为 NaN
+            X[self.col] = pd.to_numeric(X[self.col].astype(str).str.strip(), errors="coerce")
+            # 处理 NaN 和 Inf，统一填充为 0.0
+            X[self.col] = X[self.col].fillna(0.0).replace([np.inf, -np.inf], 0.0)
+            
+        # 处理 tenure 分桶特征
         if self.tenure_col in X.columns:
-            bins = [0, 6, 12, 24, 48, 60, 100]
-            labels = ['0-6m', '7-12m', '13-24m', '25-48m', '49-60m', '60m+']
-            X['tenure_binned'] = pd.cut(X[self.tenure_col], bins=bins, labels=labels, include_lowest=True).astype(str)
-            X['tenure_binned'] = X['tenure_binned'].fillna('unknown')
-        X['total_charges'] = pd.to_numeric(_evaluation_raw.astype(str).str.strip(), errors='coerce')
+            bins = [0, 6, 12, 24, 48, 60, 1000]
+            labels = ["0-6m", "7-12m", "13-24m", "25-48m", "49-60m", "60m+"]
+            # 确保输入为数值进行分桶
+            tenure_series = pd.to_numeric(X[self.tenure_col], errors="coerce")
+            X["tenure_binned"] = pd.cut(
+                tenure_series, 
+                bins=bins, 
+                labels=labels, 
+                include_lowest=True
+            ).astype(str)
+            # 处理分桶过程中产生的 NaN (即原始数据中缺失或超出范围的 tenure)
+            X["tenure_binned"] = X["tenure_binned"].replace("nan", "unknown").fillna("unknown")
+            
         return X
```

## 6. 验证机制

| 级别 | 检查内容 | 本次配置 |
| --- | --- | --- |
| 1 静态检查 | 语法、import 白名单、禁止危险调用、接口与 meta 一致性 | 仅针对生成的算子 |
| 2 沙箱执行 | 独立子进程，限制超时与内存，清除含 KEY/TOKEN 的环境变量 | 超时 180 s，内存 8192 MB |
| 3 契约检查 | 不修改输入、行数一致、不引入 NaN/Inf、概率合法、能处理未见类别与缺失值 | 冒烟样本 300 行 |
| 4 指标检查 | 阈值、种子间稳定性、疑似泄漏、耗时与内存 | 5 折 × 种子 [42, 7, 2024]；std 上限 0.03 |

最优方案资源消耗：单次训练 0.105 s，预测 50.7 ms / 千行，峰值内存 259 MB。

修复前的 1 次验证结果保留在 state.json 的 history 中。

## 7. 知识沉淀

- 回写验证记录 ValidationRun：1 条
- 注册到能力库的新算子：TotalChargesCleaner@2
- 新提炼的经验：数据清洗算子必须保证输出中不包含 NaN 或 Inf 值（E_NAN）
- 本次提供的经验：0 条（累计检索命中，不代表采纳）

## 附录 A. 生成的算法代码（pipeline.py）

```python
"""预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75；正确处理累 —— 自动生成的算法流水线

需求：预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75；正确处理累计费用的空白字符串，并增加在网时长分桶特征。
方案：interpretable_logreg_pipeline
设计依据：该方案侧重于模型的可解释性，符合业务合规约束。
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


import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class TotalChargesCleaner(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="TotalChargesCleaner",
        kind=OperatorKind.IMPUTER,
        version=2,
        origin="synthesized",
        description="将字符串类型的总费用列转换为数值，处理空白异常值并增加在网时长分桶特征",
        conditions=["数据包含total_charges列", "存在tenure在网时长列"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"col": "total_charges", "tenure_col": "tenure"},
        parent="TotalChargesCleaner@1"
    )

    def __init__(self, col="total_charges", tenure_col="tenure"):
        self.col = col
        self.tenure_col = tenure_col

    def fit(self, X, y=None):
        # 记录清洗后的均值或固定值用于兜底
        self.mean_val_ = 0.0
        return self

    def transform(self, X):
        X = X.copy()

        # 处理 total_charges 列
        if self.col in X.columns:
            # 强制转换为数值，空白及非法字符变为 NaN
            X[self.col] = pd.to_numeric(X[self.col].astype(str).str.strip(), errors="coerce")
            # 处理 NaN 和 Inf，统一填充为 0.0
            X[self.col] = X[self.col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        # 处理 tenure 分桶特征
        if self.tenure_col in X.columns:
            bins = [0, 6, 12, 24, 48, 60, 1000]
            labels = ["0-6m", "7-12m", "13-24m", "25-48m", "49-60m", "60m+"]
            # 确保输入为数值进行分桶
            tenure_series = pd.to_numeric(X[self.tenure_col], errors="coerce")
            X["tenure_binned"] = pd.cut(
                tenure_series, 
                bins=bins, 
                labels=labels, 
                include_lowest=True
            ).astype(str)
            # 处理分桶过程中产生的 NaN (即原始数据中缺失或超出范围的 tenure)
            X["tenure_binned"] = X["tenure_binned"].replace("nan", "unknown").fillna("unknown")

        return X


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_imputer", TotalChargesCleaner()),  # 清理数值列，参考 Lesson:69d51358。
        ("1_imputer", MissingValueImputer()),  # 补充缺失数据。
        ("2_encoder", OneHotCategoricalEncoder(max_categories=20)),  # 线性模型需要编码化类别特征。
        ("3_scaler", StandardFeatureScaler()),  # 逻辑回归必须先标准化，参考 Lesson:74e2da10。
        ("4_model", LogisticRegressionModel(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)),  # 逻辑回归满足可解释性需求。
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

各阶段耗时：PLAN 0.1s，BUILD 3.8s，VERIFY 5.8s，ROUTE 0.0s，REPAIR 3.8s，CURATE 2.0s

```
[PLAN] 受控注入实验：固定种子运行的真实生成方案与源码，不重新规划或生成；本次独立调用真实 LLM 修复。
[PLAN] 完成，用时 0.1s → BUILD
[BUILD] 已验证原始源码后注入 NaN：TotalChargesCleaner@2；原始源码和补丁见 injection.json
[BUILD] 完成，用时 3.8s → VERIFY
[VERIFY] interpretable_logreg_pipeline：未通过，recall=nan；E_NAN 输出引入了新的 NaN/Inf
[VERIFY] 完成，用时 1.6s → ROUTE
[ROUTE] 完成，用时 0.0s → REPAIR
[REPAIR] 修复 TotalChargesCleaner@2 第 1 轮：E_NAN 输出引入了新的 NaN/Inf
[REPAIR] 完成，用时 3.8s → VERIFY
[VERIFY] interpretable_logreg_pipeline：通过，recall=0.7785
[VERIFY] 完成，用时 4.2s → ROUTE
[ROUTE] 完成，用时 0.0s → CURATE
[CURATE] 写入 KG：1 条 ValidationRun；注册新算子 ['TotalChargesCleaner@2']；新增经验 1 条
[CURATE] 完成，用时 2.0s → DONE
```

### LLM 调用来源

| 阶段 | 请求哈希 | 来源 | 本次耗时(s) | 录制耗时(s) | tokens |
| --- | --- | --- | --- | --- | --- |
| repair | b947dcebaea8dcc6 | remote | 3.759 | - | 2271 |
| lesson | 90fee5c2ea603eaf | remote | 1.945 | - | 2456 |
