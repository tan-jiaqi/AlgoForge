# AlgoForge 验证报告

| 项目 | 内容 |
| --- | --- |
| 需求 | 预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户，召回率不低于0.75 |
| 数据 | data/churn.csv |
| 运行 ID | repair_injected_3_reuse |
| LLM | gemini-3.5-flash |
| 状态 | ✅ 成功 |
| 最优方案 | interpretable_logreg_pipeline |
| 总耗时 | 92.1 s |

共验证 3 个候选方案，1 个通过；重新规划 0 次，修复 1 轮。最优方案 `interpretable_logreg_pipeline` 的 recall = 0.779（要求 ≥ 0.75）。

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
| imputer | MissingValueImputer@1 | 是 | 0.510 | 0.66 / 0.09 / 0.73 | 匹配条件：has_missing_values；历史验证 4 次，平均 recall 0.732 |
| imputer | TotalChargesCleaner@1 | 否（历史实现，可参考生成） | 0.482 | 1.00 / 0.27 / 0.00 | 匹配条件：has_missing_values、numeric_features、string_typed_numeric |
| imputer | TotalChargesCleaner@2 | 是 | 0.455 | 0.55 / 0.00 / 0.78 | 历史验证 1 次，平均 recall 0.778 |
| feature | TenureBucketizer@1 | 否（历史实现，可参考生成） | 0.646 | 0.81 / 0.27 / 0.80 | 匹配条件：interpretable、numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | ChargeRatioFeatureGenerator@1 | 否（历史实现，可参考生成） | 0.615 | 0.80 / 0.18 / 0.80 | 匹配条件：numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | ServiceCountFeature@1 | 否（历史实现，可参考生成） | 0.427 | 1.00 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| feature | IdColumnDropper@1 | 否（历史实现，可参考生成） | 0.350 | 0.81 / 0.09 / 0.00 | 匹配条件：id_columns |
| encoder | OneHotCategoricalEncoder@1 | 是 | 0.645 | 0.93 / 0.18 / 0.73 | 匹配条件：has_categorical_features、low_cardinality；历史验证 4 次，平均 recall 0.732 |
| encoder | KFoldTargetEncoder@1 | 否（历史实现，可参考生成） | 0.427 | 1.00 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| scaler | StandardFeatureScaler@1 | 是 | 0.650 | 1.00 / 0.09 / 0.74 | 匹配条件：numeric_features；历史验证 2 次，平均 recall 0.744 |
| model | BalancedLightGBMClassifier@1 | 否（历史实现，可参考生成） | 0.661 | 1.00 / 0.09 / 0.78 | 匹配条件：imbalanced；历史验证 1 次，平均 recall 0.780 |
| model | LogisticRegressionModel@1 | 是 | 0.637 | 0.97 / 0.09 / 0.74 | 匹配条件：interpretable；历史验证 2 次，平均 recall 0.744 |
| model | RandomForestModel@1 | 是 | 0.596 | 0.93 / 0.09 / 0.66 | 匹配条件：mixed_features；历史验证 1 次，平均 recall 0.660 |
| model | LightGBMModel@1 | 是 | 0.560 | 0.73 / 0.09 / 0.80 | 匹配条件：imbalanced；历史验证 1 次，平均 recall 0.800 |

### 参考的历史经验

- **样本不平衡导致默认模型召回率偏低**（M_THRESHOLD）：启用平衡类权重（如 class_weight='balanced' 或 BalancedLightGBMClassifier）以提升召回率
- **数值列含空白字符导致类型转换失败**（E_RUNTIME）：先 str.strip()，再 pd.to_numeric(errors='coerce')，缺失值填补为 0
- **逻辑回归标准化与可解释性**：必须先进行特征标准化（如 StandardFeatureScaler），在 0.1~10 之间搜索正则化强度 C，可直接输出系数给业务方。
- **未剔除客户 ID 导致 One-Hot 编码后维度暴涨与内存溢出**（E_TIMEOUT）：建模前剔除 ID 等标识列，并在编码器中设置最大类别数
- **数据清洗算子必须保证输出中不包含 NaN 或 Inf 值**（E_NAN）：在数据转换逻辑中，必须在返回结果前显式调用 fillna() 或 replace([np.inf, -np.inf], ...) 对所有潜在的非数值非法值进行填充，确保算子输出的完整性。

### 知识引用检查

检索并传入上下文不等于采纳。以下引用由模型声明，程序仅核对可见范围与结构化适用对象；不证明语义落实或指标收益。

| 方案 | 步骤 | 模型引用 | 范围检查 | 语义落实 |
| --- | --- | --- | --- | --- |
| interpretable_logreg_pipeline | 1 | Operator:TotalChargesCleaner@2、ValidationRun:repair_injected_3-interpretable_logreg_pipeline | 通过 | 未验证 |
| interpretable_logreg_pipeline | 2 | Operator:MissingValueImputer@1 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 3 | Operator:IdColumnDropper@1 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 4 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 5 | Operator:StandardFeatureScaler@1 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 6 | Operator:LogisticRegressionModel@1 | 通过 | 未验证 |
| robust_lgbm_pipeline | 1 | Operator:TotalChargesCleaner@2 | 通过 | 未验证 |
| robust_lgbm_pipeline | 2 | Operator:MissingValueImputer@1 | 通过 | 未验证 |
| robust_lgbm_pipeline | 3 | Operator:ServiceCountFeature@1 | 通过 | 未验证 |
| robust_lgbm_pipeline | 4 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| robust_lgbm_pipeline | 5 | Operator:LightGBMModel@1 | 通过 | 未验证 |
| balanced_rf_pipeline | 1 | Operator:TotalChargesCleaner@2 | 通过 | 未验证 |
| balanced_rf_pipeline | 2 | Operator:TenureBucketizer@1 | 通过 | 未验证 |
| balanced_rf_pipeline | 3 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| balanced_rf_pipeline | 4 | Operator:RandomForestModel@1 | 通过 | 未验证 |

## 3. 候选方案比较

排序规则：先看是否通过全部检查，再比主指标，再比次要指标之和，最后训练耗时短者优先。指标为多随机种子 × 分层 K 折交叉验证的均值 ± 种子间标准差。

| 排名 | 方案 | 结果 | recall | roc_auc | precision | f1 | recall_at_20 | 单次训练 (s) | 峰值内存 (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ★ interpretable_logreg_pipeline | 通过 | 0.779 ± 0.002 | 0.814 ± 0.001 | 0.483 ± 0.002 | 0.596 ± 0.002 | 0.454 ± 0.002 | 0.076 | 253 |
| 2 | balanced_rf_pipeline | 未通过 | 0.718 ± 0.003 | 0.803 ± 0.001 | 0.491 ± 0.003 | 0.583 ± 0.001 | 0.451 ± 0.003 | 0.817 | 273 |
| 3 | robust_lgbm_pipeline | 未通过 | 0.652 ± 0.008 | 0.786 ± 0.003 | 0.491 ± 0.004 | 0.560 ± 0.005 | 0.426 ± 0.003 | 0.335 | 271 |

```
recall（| 为阈值 0.75）
interpretable_logreg_pipeline        ■■■■■■■■■■■■■■■|···· 0.779
balanced_rf_pipeline                 ■■■■■■■■■■■■■■·|···· 0.718
robust_lgbm_pipeline                 ■■■■■■■■■■■■■··|···· 0.652
```

### 各方案的结构与诊断

- `interpretable_logreg_pipeline`：TotalChargesCleaner → MissingValueImputer → 生成:DropIdColumn → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel。诊断：无
- `balanced_rf_pipeline`：TotalChargesCleaner → 生成:TenureBucket → OneHotCategoricalEncoder → RandomForestModel。诊断：`M_THRESHOLD` recall = 0.7176 低于阈值 0.75
- `robust_lgbm_pipeline`：TotalChargesCleaner → MissingValueImputer → 生成:AddServiceCount → OneHotCategoricalEncoder → LightGBMModel。诊断：`M_THRESHOLD` recall = 0.6519 低于阈值 0.75

## 4. 最优方案：interpretable_logreg_pipeline

该方案利用逻辑回归的可解释性和特征标准化，结合针对性的数据清洗，确保模型在满足召回率阈值的前提下具备良好的业务解释能力。

| # | 步骤 | 算子 | 参数 | 选择理由 |
| --- | --- | --- | --- | --- |
| 1 | imputer | TotalChargesCleaner@2 | - | 处理总费用列的字符串异常并集成时长特征，满足业务需求且有历史验证记录。 |
| 2 | imputer | MissingValueImputer@1 | - | 兜底填补其他数值与类别列缺失值，确保数据完整性。 |
| 3 | feature | 生成 DropIdColumn | - | 剔除ID标识列以避免编码维度暴涨，符合 Lesson:88299a38。 |
| 4 | encoder | OneHotCategoricalEncoder@1 | max_categories=10 | 对类别特征进行One-Hot编码，限制维度以提升模型表现。 |
| 5 | scaler | StandardFeatureScaler@1 | - | 逻辑回归必须标准化，以便输出可解释的系数。 |
| 6 | model | LogisticRegressionModel@1 | C=0.5, class_weight=balanced | 使用逻辑回归提供高可解释性，并设置 balanced 权重以满足召回率需求。 |

## 5. 能力生成与修复

| 生成算子 | 类型 | 用途 | 参考的历史实现 | 修复轮数 | 注册到能力库 |
| --- | --- | --- | --- | --- | --- |
| DropIdColumn@1 | feature | 删除客户ID列以防止过拟合和特征维度暴涨 | IdColumnDropper@1 | 0 | 是 |
| AddServiceCount@1 | feature | 计算增值服务总数以捕捉用户粘性特征 | ServiceCountFeature@1 | 1 | 否 |
| TenureBucket@1 | feature | 将时长字段离散化处理 | TenureBucketizer@1 | 0 | 否 |

### 修复 AddServiceCount@1 第 1 轮：✅ 已修复

- 诊断：`E_INTERFACE`（static）meta.params 与 __init__ 默认值不一致：["service_cols: meta=['online_security', 'tech_support', 'streaming_tv'], __init__=('online_security', 'tech_support', 'streaming_tv')"]
- 提示：实例化时 meta.params 会覆盖 __init__ 默认值，两处必须一致
- 参考经验：0 条

```diff
--- 修复前
+++ 修复后
@@ -19,3 +19,3 @@
 
-    def __init__(self, service_cols=("online_security", "tech_support", "streaming_tv")):
+    def __init__(self, service_cols=["online_security", "tech_support", "streaming_tv"]):
         self.service_cols = service_cols
@@ -29,4 +29,3 @@
         # 统计增值服务数量
-        # 根据业务经验，通常判定逻辑为值等于"Yes"或"是"
-        # 兼容处理：将缺失值视为未订购
+        # 根据业务经验，通常判定逻辑为值等于"Yes"、"是"或"1"
         count = np.zeros(len(X))
@@ -34,4 +33,4 @@
             if col in X.columns:
-                # 统一转为字符串并剔除空格进行匹配
-                val = X[col].astype(str).str.strip().str.lower()
+                # 统一转为字符串并剔除空格进行匹配，兼容缺失值
+                val = X[col].fillna("no").astype(str).str.strip().str.lower()
                 count += (val.isin(["yes", "是", "1"])).astype(int)
@@ -40,3 +39,3 @@
         
-        # 处理可能的缺失值或异常数值（虽然此逻辑不产生NaN，但遵循规范兜底）
+        # 处理可能的异常数值（虽然此逻辑不产生NaN/inf，但遵循规范兜底）
         X["n_services"] = X["n_services"].fillna(0).replace([np.inf, -np.inf], 0)
```

## 6. 验证机制

| 级别 | 检查内容 | 本次配置 |
| --- | --- | --- |
| 1 静态检查 | 语法、import 白名单、禁止危险调用、接口与 meta 一致性 | 仅针对生成的算子 |
| 2 沙箱执行 | 独立子进程，限制超时与内存，清除含 KEY/TOKEN 的环境变量 | 超时 180 s，内存 8192 MB |
| 3 契约检查 | 不修改输入、行数一致、不引入 NaN/Inf、概率合法、能处理未见类别与缺失值 | 冒烟样本 300 行 |
| 4 指标检查 | 阈值、种子间稳定性、疑似泄漏、耗时与内存 | 5 折 × 种子 [42, 7, 2024]；std 上限 0.03 |

最优方案资源消耗：单次训练 0.076 s，预测 36.5 ms / 千行，峰值内存 253 MB。

修复前的 1 次验证结果保留在 state.json 的 history 中。

## 7. 知识沉淀

- 回写验证记录 ValidationRun：3 条
- 注册到能力库的新算子：DropIdColumn@1
- 新提炼的经验：确保算子元数据与构造函数默认参数类型一致（E_INTERFACE）
- 本次提供的经验：6 条（累计检索命中，不代表采纳）

## 附录 A. 生成的算法代码（pipeline.py）

```python
"""预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户，召回率不低于0.75 —— 自动生成的算法流水线

需求：预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户，召回率不低于0.75
方案：interpretable_logreg_pipeline
设计依据：该方案利用逻辑回归的可解释性和特征标准化，结合针对性的数据清洗，确保模型在满足召回率阈值的前提下具备良好的业务解释能力。
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
from factory.schemas import OperatorKind

class DropIdColumn(TransformerMixin, Operator):
    """
    删除客户ID及高基数标识列，防止后续处理（如One-Hot编码）导致维度爆炸。
    """
    meta = OperatorMeta(
        name="DropIdColumn",
        kind=OperatorKind.FEATURE,
        version=1,
        origin="synthesized",
        description="删除指定的ID列或高基数标识列，防止维度爆炸并保证数据完整性。",
        conditions=["数据包含customer_id等标识列"],
        dependencies=["pandas", "numpy"],
        params={"id_cols": ("customer_id",)},
        parent="IdColumnDropper@1"
    )

    def __init__(self, id_cols=("customer_id",)):
        self.id_cols = id_cols

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = X.copy()

        # 识别存在于列中的待删除ID列
        cols_to_drop = [c for c in self.id_cols if c in X.columns]
        X = X.drop(columns=cols_to_drop)

        # 根据 Lesson:cf247d47，确保输出不包含 NaN 或 Inf
        # 对于数值列，填充缺失值为0或中位数，此处选择填充为0以保证鲁棒性
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
            X[numeric_cols] = X[numeric_cols].fillna(0)

        return X

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
        ("0_imputer", TotalChargesCleaner(col='total_charges', tenure_col='tenure')),  # 处理总费用列的字符串异常并集成时长特征，满足业务需求且有历史验证记录。
        ("1_imputer", MissingValueImputer()),  # 兜底填补其他数值与类别列缺失值，确保数据完整性。
        ("2_feature", DropIdColumn()),  # 剔除ID标识列以避免编码维度暴涨，符合 Lesson:88299a38。
        ("3_encoder", OneHotCategoricalEncoder(max_categories=10)),  # 对类别特征进行One-Hot编码，限制维度以提升模型表现。
        ("4_scaler", StandardFeatureScaler()),  # 逻辑回归必须标准化，以便输出可解释的系数。
        ("5_model", LogisticRegressionModel(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)),  # 使用逻辑回归提供高可解释性，并设置 balanced 权重以满足召回率需求。
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

各阶段耗时：PLAN 6.0s，BUILD 7.3s，VERIFY 29.9s，ROUTE 0.0s，REPAIR 35.1s，CURATE 13.7s

```
[PLAN] 场景 telecom_churn；画像标签 ['has_missing_values', 'numeric_features', 'has_categorical_features', 'low_cardinality', 'string_typed_numeric', 'id_columns', 'imbalanced', 'time_tenure', 'mixed_features']
[PLAN] TaskSpec：binary_classification，目标 churn，主指标 recall >= 0.75
[PLAN] 候选方案：['interpretable_logreg_pipeline(TotalChargesCleaner → MissingValueImputer → *DropIdColumn → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel)', 'robust_lgbm_pipeline(TotalChargesCleaner → MissingValueImputer → *AddServiceCount → OneHotCategoricalEncoder → LightGBMModel)', 'balanced_rf_pipeline(TotalChargesCleaner → *TenureBucket → OneHotCategoricalEncoder → RandomForestModel)']
[PLAN] 完成，用时 6.0s → BUILD
[BUILD] 生成算子 DropIdColumn@1（参考 IdColumnDropper@1）：删除客户ID列以防止过拟合和特征维度暴涨
[BUILD] 生成算子 AddServiceCount@1（参考 ServiceCountFeature@1）：计算增值服务总数以捕捉用户粘性特征
[BUILD] 生成算子 TenureBucket@1（参考 TenureBucketizer@1）：将时长字段离散化处理
[BUILD] 完成，用时 7.3s → VERIFY
[VERIFY] interpretable_logreg_pipeline：通过，recall=0.7793
[VERIFY] robust_lgbm_pipeline：未通过，recall=nan；E_INTERFACE meta.params 与 __init__ 默认值不一致：["service_cols: meta=['online_security', 'tech_support', 'streaming_tv'], __init__=('online_security', 'tech_support', 'streaming_tv')"]
[VERIFY] balanced_rf_pipeline：未通过，recall=0.7176；M_THRESHOLD recall = 0.7176 低于阈值 0.75
[VERIFY] 完成，用时 22.1s → ROUTE
[ROUTE] 完成，用时 0.0s → REPAIR
[REPAIR] 修复 AddServiceCount@1 第 1 轮：E_INTERFACE meta.params 与 __init__ 默认值不一致：["service_cols: meta=['online_security', 'tech_sup
[REPAIR] 完成，用时 35.1s → VERIFY
[VERIFY] robust_lgbm_pipeline：未通过，recall=0.6519；M_THRESHOLD recall = 0.6519 低于阈值 0.75
[VERIFY] 完成，用时 7.9s → ROUTE
[ROUTE] 完成，用时 0.0s → CURATE
[CURATE] 写入 KG：3 条 ValidationRun；注册新算子 ['DropIdColumn@1']；新增经验 1 条
[CURATE] 完成，用时 13.7s → DONE
```

### LLM 调用来源

| 阶段 | 请求哈希 | 来源 | 本次耗时(s) | 录制耗时(s) | tokens |
| --- | --- | --- | --- | --- | --- |
| plan | c9b5ea5f1f946385 | remote | 5.879 | - | 12627 |
| synthesize | d4183d758c6c9b73 | remote | 2.473 | - | 2515 |
| synthesize | 90361bef1d420efc | remote | 2.332 | - | 2531 |
| synthesize | abf26839b8645851 | remote | 2.507 | - | 2684 |
| repair | f82578cab63cbb2d | remote | 35.059 | - | 2029 |
| lesson | 94046b360a136683 | remote | 13.702 | - | 1688 |
