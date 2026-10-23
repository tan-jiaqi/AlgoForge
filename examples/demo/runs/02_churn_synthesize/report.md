# AlgoForge 验证报告

| 项目 | 内容 |
| --- | --- |
| 需求 | 预测30天内可能流失的客户，需可解释，召回优先 |
| 数据 | data/churn.csv |
| 运行 ID | 02_churn_synthesize |
| LLM | gemini-3.5-flash |
| 状态 | ✅ 成功 |
| 最优方案 | interpretable_logistic_regression |
| 总耗时 | 63.1 s |

共验证 3 个候选方案，1 个通过；重新规划 0 次，修复 0 轮。最优方案 `interpretable_logistic_regression` 的 recall = 0.778（要求 ≥ 0.75）。

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
- **未剔除客户 ID 导致 One-Hot 编码后维度暴涨与内存溢出**（E_TIMEOUT）：建模前剔除 ID 等标识列，并在编码器中设置最大类别数

## 3. 候选方案比较

排序规则：先看是否通过全部检查，再比主指标，再比次要指标之和，最后训练耗时短者优先。指标为多随机种子 × 分层 K 折交叉验证的均值 ± 种子间标准差。

| 排名 | 方案 | 结果 | recall | roc_auc | precision | f1 | recall_at_20 | 单次训练 (s) | 峰值内存 (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ★ interpretable_logistic_regression | 通过 | 0.778 ± 0.001 | 0.814 ± 0.001 | 0.482 ± 0.001 | 0.595 ± 0.001 | 0.455 ± 0.001 | 0.088 | 261 |
| 2 | cost_ratio_random_forest | 未通过 | 0.714 ± 0.003 | 0.804 ± 0.001 | 0.493 ± 0.002 | 0.583 ± 0.002 | 0.455 ± 0.001 | 0.612 | 269 |
| 3 | balanced_lightgbm_boosting | 未通过 | 0.630 ± 0.003 | 0.783 ± 0.004 | 0.490 ± 0.001 | 0.551 ± 0.002 | 0.420 ± 0.008 | 0.259 | 265 |

```
recall（| 为阈值 0.75）
interpretable_logistic_regression    ■■■■■■■■■■■■■■■|···· 0.778
cost_ratio_random_forest             ■■■■■■■■■■■■■■·|···· 0.714
balanced_lightgbm_boosting           ■■■■■■■■■■■■■··|···· 0.630
```

### 各方案的结构与诊断

- `interpretable_logistic_regression`：生成:TotalChargesCleaner → MissingValueImputer → 生成:TenureBucketizer → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel。诊断：无
- `cost_ratio_random_forest`：生成:TotalChargesCleaner → MissingValueImputer → 生成:ChargeRatioFeatureGenerator → OneHotCategoricalEncoder → RandomForestModel。诊断：`M_THRESHOLD` recall = 0.7143 低于阈值 0.75
- `balanced_lightgbm_boosting`：生成:TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → 生成:BalancedLightGBMClassifier。诊断：`M_THRESHOLD` recall = 0.6298 低于阈值 0.75；`M_THRESHOLD` roc_auc = 0.7830 低于阈值 0.8

## 4. 最优方案：interpretable_logistic_regression

针对高可解释性需求与召回优先目标，结合在网时长分箱特征与标准化的线性模型，配合平衡类别权重实现高召回率与透明的特征权重解释。

| # | 步骤 | 算子 | 参数 | 选择理由 |
| --- | --- | --- | --- | --- |
| 1 | imputer | 生成 TotalChargesCleaner | - | 根据 lesson 经验，必须先清洗 total_charges 中的空白字符并转为浮点数，防止后续处理类型转换失败。 |
| 2 | imputer | MissingValueImputer@1 | - | 依据匹配条件 has_missing_values，对其余数值和类别缺失列进行中位数和众数填补。 |
| 3 | feature | 生成 TenureBucketizer | - | 历史验证召回率达 0.800，分箱在网时长有助于线性模型捕捉非线性流失拐点并提升可解释性。 |
| 4 | encoder | OneHotCategoricalEncoder@1 | max_categories=20 | 对低基数类别特征进行独热编码，并限制类别数避免维度过高。 |
| 5 | scaler | StandardFeatureScaler@1 | - | 根据 lesson 经验，逻辑回归模型必须在拟合前对特征进行标准化。 |
| 6 | model | LogisticRegressionModel@1 | C=1.0, class_weight=balanced, max_iter=1000, random_state=42 | 满足 interpretable 约束，且设置 class_weight='balanced' 解决不平衡导致的召回率不足问题。 |

## 5. 能力生成与修复

| 生成算子 | 类型 | 用途 | 参考的历史实现 | 修复轮数 | 注册到能力库 |
| --- | --- | --- | --- | --- | --- |
| TotalChargesCleaner@2 | imputer | 将包含空白字符串的数值列安全转换为浮点型，避免数值转换异常并将空缺值填充为0.0 | TotalChargesCleaner@1 | 0 | 是 |
| TenureBucketizer@2 | feature | 将连续的在网时长字段分箱为离散区间，增强线性模型的表达能力与业务可解释性 | TenureBucketizer@1 | 0 | 是 |
| BalancedLightGBMClassifier@2 | model | 使用带平衡权重的 LightGBM 训练流失分类模型，提供特征重要性分析并优化召回 | BalancedLightGBMClassifier@1 | 0 | 否 |
| ChargeRatioFeatureGenerator@2 | feature | 计算月度资费与历史平均费用的比率，刻画用户近期资费变动敏感度 | ChargeRatioFeatureGenerator@1 | 0 | 否 |

## 6. 验证机制

| 级别 | 检查内容 | 本次配置 |
| --- | --- | --- |
| 1 静态检查 | 语法、import 白名单、禁止危险调用、接口与 meta 一致性 | 仅针对生成的算子 |
| 2 沙箱执行 | 独立子进程，限制超时与内存，清除含 KEY/TOKEN 的环境变量 | 超时 180 s，内存 8192 MB |
| 3 契约检查 | 不修改输入、行数一致、不引入 NaN/Inf、概率合法、能处理未见类别与缺失值 | 冒烟样本 300 行 |
| 4 指标检查 | 阈值、种子间稳定性、疑似泄漏、耗时与内存 | 5 折 × 种子 [42, 7, 2024]；std 上限 0.03 |

最优方案资源消耗：单次训练 0.088 s，预测 42.8 ms / 千行，峰值内存 261 MB。

## 7. 知识沉淀

- 回写验证记录 ValidationRun：3 条
- 注册到能力库的新算子：TotalChargesCleaner@2、TenureBucketizer@2
- 新提炼的经验：无
- 本次使用的经验：7 条（命中次数已累计）

## 附录 A. 生成的算法代码（pipeline.py）

```python
"""预测30天内可能流失的客户，需可解释，召回优先 —— 自动生成的算法流水线

需求：预测30天内可能流失的客户，需可解释，召回优先
方案：interpretable_logistic_regression
设计依据：针对高可解释性需求与召回优先目标，结合在网时长分箱特征与标准化的线性模型，配合平衡类别权重实现高召回率与透明的特征权重解释。
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
        description="安全清洗并转换含空白字符串的累计消费等数值列为浮点型，并填充缺失值为指定默认值",
        conditions=["string_typed_numeric", "has_missing_values"],
        dependencies=[],
        params={"col": "total_charges", "fill_value": 0.0},
        parent="TotalChargesCleaner@1",
    )

    def __init__(self, col: str = "total_charges", fill_value: float = 0.0):
        self.col = col
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.col in X.columns:
            cleaned_series = (
                pd.to_numeric(X[self.col].astype(str).str.strip(), errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(self.fill_value)
                .astype(float)
            )
            X[self.col] = cleaned_series
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
        description="将在网时长字段分箱为离散区间特征以增强线性模型的表达能力与业务可解释性",
        conditions=["time_tenure", "numeric_features"],
        dependencies=["pandas", "numpy"],
        params={
            "col": "tenure",
            "bins": (-float("inf"), 6.0, 12.0, 24.0, 48.0, float("inf")),
            "labels": ("0-6", "6-12", "12-24", "24-48", "48+"),
            "drop_original": False,
        },
        parent="TenureBucketizer@1",
    )

    def __init__(
        self,
        col: str = "tenure",
        bins: tuple = (-float("inf"), 6.0, 12.0, 24.0, 48.0, float("inf")),
        labels: tuple = ("0-6", "6-12", "12-24", "24-48", "48+"),
        drop_original: bool = False,
    ):
        self.col = col
        self.bins = bins
        self.labels = labels
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_name_ = f"{self.col}_bucket"
        self.bins_ = list(self.bins) if self.bins is not None else [-float("inf"), 6.0, 12.0, 24.0, 48.0, float("inf")]
        self.labels_ = list(self.labels) if self.labels is not None else ["0-6", "6-12", "12-24", "24-48", "48+"]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.col not in X.columns:
            return X

        # 安全数值转换，兼顾脏数据、空白字符串与缺失值
        tenure_series = X[self.col]
        if tenure_series.dtype == object:
            tenure_series = tenure_series.astype(str).str.strip()
        numeric_tenure = pd.to_numeric(tenure_series, errors="coerce")

        # 分箱并处理缺失区间
        bucketed = pd.cut(numeric_tenure, bins=self.bins_, labels=self.labels_)
        bucketed_str = bucketed.astype(str)
        bucketed_str = bucketed_str.replace("nan", "unknown").fillna("unknown")

        X[self.feature_name_] = bucketed_str

        if self.drop_original:
            X = X.drop(columns=[self.col])

        return X


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_imputer", TotalChargesCleaner()),  # 根据 lesson 经验，必须先清洗 total_charges 中的空白字符并转为浮点数，防止后续处理类型转换失败。
        ("1_imputer", MissingValueImputer()),  # 依据匹配条件 has_missing_values，对其余数值和类别缺失列进行中位数和众数填补。
        ("2_feature", TenureBucketizer()),  # 历史验证召回率达 0.800，分箱在网时长有助于线性模型捕捉非线性流失拐点并提升可解释性。
        ("3_encoder", OneHotCategoricalEncoder(max_categories=20)),  # 对低基数类别特征进行独热编码，并限制类别数避免维度过高。
        ("4_scaler", StandardFeatureScaler()),  # 根据 lesson 经验，逻辑回归模型必须在拟合前对特征进行标准化。
        ("5_model", LogisticRegressionModel(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)),  # 满足 interpretable 约束，且设置 class_weight='balanced' 解决不平衡导致的召回率不足问题。
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

各阶段耗时：PLAN 14.2s，BUILD 24.2s，VERIFY 24.7s，ROUTE 0.0s，CURATE 0.0s

```
[PLAN] 场景 telecom_churn；画像标签 ['has_missing_values', 'numeric_features', 'has_categorical_features', 'low_cardinality', 'string_typed_numeric', 'id_columns', 'imbalanced', 'time_tenure', 'mixed_features']
[PLAN] TaskSpec：binary_classification，目标 churn，主指标 recall >= 0.75
[PLAN] 候选方案：['interpretable_logistic_regression(*TotalChargesCleaner → MissingValueImputer → *TenureBucketizer → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel)', 'balanced_lightgbm_boosting(*TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → *BalancedLightGBMClassifier)', 'cost_ratio_random_forest(*TotalChargesCleaner → MissingValueImputer → *ChargeRatioFeatureGenerator → OneHotCategoricalEncoder → RandomForestModel)']
[PLAN] 完成，用时 14.2s → BUILD
[BUILD] 生成算子 TotalChargesCleaner@2（参考 TotalChargesCleaner@1）：将包含空白字符串的数值列安全转换为浮点型，避免数值转换异常并将空缺值填充为0.0
[BUILD] 生成算子 TenureBucketizer@2（参考 TenureBucketizer@1）：将连续的在网时长字段分箱为离散区间，增强线性模型的表达能力与业务可解释性
[BUILD] 生成算子 BalancedLightGBMClassifier@2（参考 BalancedLightGBMClassifier@1）：使用带平衡权重的 LightGBM 训练流失分类模型，提供特征重要性分析并优化召回
[BUILD] 生成算子 ChargeRatioFeatureGenerator@2（参考 ChargeRatioFeatureGenerator@1）：计算月度资费与历史平均费用的比率，刻画用户近期资费变动敏感度
[BUILD] 完成，用时 24.2s → VERIFY
[VERIFY] interpretable_logistic_regression：通过，recall=0.7778
[VERIFY] balanced_lightgbm_boosting：未通过，recall=0.6298；M_THRESHOLD recall = 0.6298 低于阈值 0.75；M_THRESHOLD roc_auc = 0.7830 低于阈值 0.8
[VERIFY] cost_ratio_random_forest：未通过，recall=0.7143；M_THRESHOLD recall = 0.7143 低于阈值 0.75
[VERIFY] 完成，用时 24.7s → ROUTE
[ROUTE] 完成，用时 0.0s → CURATE
[CURATE] 写入 KG：3 条 ValidationRun；注册新算子 ['TotalChargesCleaner@2', 'TenureBucketizer@2']；新增经验 0 条
[CURATE] 完成，用时 0.0s → DONE
```
