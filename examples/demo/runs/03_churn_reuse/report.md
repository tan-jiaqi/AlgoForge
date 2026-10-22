# AlgoForge 验证报告

| 项目 | 内容 |
| --- | --- |
| 需求 | 预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户 |
| 数据 | data/churn.csv |
| 运行 ID | 03_churn_reuse |
| LLM | gemini-3.5-flash |
| 状态 | ✅ 成功 |
| 最优方案 | interpretable_logistic_regression |
| 总耗时 | 10.4 s |

共验证 2 个候选方案，1 个通过；重新规划 0 次，修复 0 轮。最优方案 `interpretable_logistic_regression` 的 recall = 0.778（要求 ≥ 0.75）。

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
| imputer | TotalChargesCleaner@2 | 是 | 0.599 | 0.78 / 0.18 / 0.78 | 匹配条件：has_missing_values、string_typed_numeric；历史验证 1 次，平均 recall 0.778 |
| imputer | MissingValueImputer@1 | 是 | 0.497 | 0.62 / 0.09 / 0.73 | 匹配条件：has_missing_values；历史验证 4 次，平均 recall 0.732 |
| imputer | TotalChargesCleaner@1 | 否（历史实现，可参考生成） | 0.482 | 1.00 / 0.27 / 0.00 | 匹配条件：has_missing_values、numeric_features、string_typed_numeric |
| feature | TenureBucketizer@1 | 否（历史实现，可参考生成） | 0.616 | 0.73 / 0.27 / 0.80 | 匹配条件：interpretable、numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | ChargeRatioFeatureGenerator@1 | 否（历史实现，可参考生成） | 0.589 | 0.73 / 0.18 / 0.80 | 匹配条件：numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | TenureBucketizer@2 | 是 | 0.576 | 0.72 / 0.18 / 0.78 | 匹配条件：numeric_features、time_tenure；历史验证 1 次，平均 recall 0.778 |
| encoder | OneHotCategoricalEncoder@1 | 是 | 0.645 | 0.93 / 0.18 / 0.73 | 匹配条件：has_categorical_features、low_cardinality；历史验证 4 次，平均 recall 0.732 |
| encoder | KFoldTargetEncoder@1 | 否（历史实现，可参考生成） | 0.427 | 1.00 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| scaler | StandardFeatureScaler@1 | 是 | 0.650 | 1.00 / 0.09 / 0.74 | 匹配条件：numeric_features；历史验证 2 次，平均 recall 0.744 |
| model | BalancedLightGBMClassifier@1 | 否（历史实现，可参考生成） | 0.661 | 1.00 / 0.09 / 0.78 | 匹配条件：imbalanced；历史验证 1 次，平均 recall 0.780 |
| model | LogisticRegressionModel@1 | 是 | 0.638 | 0.97 / 0.09 / 0.74 | 匹配条件：interpretable；历史验证 2 次，平均 recall 0.744 |
| model | RandomForestModel@1 | 是 | 0.602 | 0.94 / 0.09 / 0.66 | 匹配条件：mixed_features；历史验证 1 次，平均 recall 0.660 |

### 参考的历史经验

- **数值列含空白字符导致类型转换失败**（E_RUNTIME）：先 str.strip()，再 pd.to_numeric(errors='coerce')，缺失值填补为 0
- **样本不平衡导致默认模型召回率偏低**（M_THRESHOLD）：启用平衡类权重（如 class_weight='balanced' 或 BalancedLightGBMClassifier）以提升召回率
- **逻辑回归标准化与可解释性**：必须先进行特征标准化（如 StandardFeatureScaler），在 0.1~10 之间搜索正则化强度 C，可直接输出系数给业务方。
- **未剔除客户 ID 导致 One-Hot 编码后维度暴涨与内存溢出**（E_TIMEOUT）：建模前剔除 ID 等标识列，并在编码器中设置最大类别数
- **避免在沙箱中使用 getattr 等受限反射函数，访问可选属性应使用 try/except AttributeError**（E_UNSAFE）：避免使用 getattr、setattr 等受限反射函数；获取可能未初始化的属性时，改用直接属性访问并配合 try/except AttributeError 来提供默认值

## 3. 候选方案比较

排序规则：先看是否通过全部检查，再比主指标，再比次要指标之和，最后训练耗时短者优先。指标为多随机种子 × 分层 K 折交叉验证的均值 ± 种子间标准差。

| 排名 | 方案 | 结果 | recall | roc_auc | precision | f1 | recall_at_20 | 单次训练 (s) | 峰值内存 (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ★ interpretable_logistic_regression | 通过 | 0.778 ± 0.000 | 0.814 ± 0.001 | 0.482 ± 0.001 | 0.595 ± 0.001 | 0.457 ± 0.001 | 0.080 | 256 |
| 2 | balanced_lightgbm | 未通过 | 0.624 ± 0.007 | 0.783 ± 0.003 | 0.492 ± 0.002 | 0.550 ± 0.003 | 0.420 ± 0.006 | 0.265 | 263 |

```
recall（| 为阈值 0.75）
interpretable_logistic_regression    ■■■■■■■■■■■■■■■|···· 0.778
balanced_lightgbm                    ■■■■■■■■■■■■···|···· 0.624
```

### 各方案的结构与诊断

- `interpretable_logistic_regression`：生成:IdColumnDropper → TotalChargesCleaner → MissingValueImputer → TenureBucketizer → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel。诊断：无
- `balanced_lightgbm`：生成:IdColumnDropper → TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → 生成:BalancedLightGBMClassifier。诊断：`M_THRESHOLD` recall = 0.6243 低于阈值 0.75；`M_THRESHOLD` roc_auc = 0.7835 低于阈值 0.8

## 4. 最优方案：interpretable_logistic_regression

针对电信客户流失预测任务的高可解释性与召回率优先（recall>=0.75）需求，首先清洗脏数据并剔除高基数ID列，通过在网时长分箱与One-Hot编码提升线性表达能力，经标准化后输入配置样本均衡权重的逻辑回归模型，系数直观透明且能有效捕捉流失客户。

| # | 步骤 | 算子 | 参数 | 选择理由 |
| --- | --- | --- | --- | --- |
| 1 | feature | 生成 IdColumnDropper | - | 根据经验提示，customer_id基数过高，直接编码会导致维度暴涨与内存溢出，需在预处理前剔除。 |
| 2 | imputer | TotalChargesCleaner@2 | - | 根据经验提示，total_charges列存在空白字符脏数据，需先清洗并安全转换为浮点数值。 |
| 3 | imputer | MissingValueImputer@1 | - | 处理 monthly_charges 与 payment_method 列的缺失值，保证数据完整性。 |
| 4 | feature | TenureBucketizer@2 | - | 将在网时长 tenure 字段按生命周期分箱离散化，增强线性模型的表达能力与业务可解释性。 |
| 5 | encoder | OneHotCategoricalEncoder@1 | - | 对低基数类别特征进行 One-Hot 编码，转换为线性分类器可学习的稀疏向量。 |
| 6 | scaler | StandardFeatureScaler@1 | - | 逻辑回归对特征尺度敏感，必须对数值特征进行 Z-score 标准化以保证权重可解释与收敛。 |
| 7 | model | LogisticRegressionModel@1 | class_weight=balanced, C=1.0 | 满足任务严格的可解释性约束，配置 class_weight='balanced' 应对样本不平衡以确保召回率达标。 |

## 5. 能力生成与修复

| 生成算子 | 类型 | 用途 | 参考的历史实现 | 修复轮数 | 注册到能力库 |
| --- | --- | --- | --- | --- | --- |
| IdColumnDropper@2 | feature | 剔除客户ID等标识列，防止One-Hot编码时维度暴涨引发内存溢出 | - | 0 | 是 |
| BalancedLightGBMClassifier@2 | model | 使用 LightGBM 训练二分类模型，设置 class_weight='balanced' 自动平衡正负样本权重以提升召回率，并输出特征重要性 | BalancedLightGBMClassifier@1 | 0 | 否 |

## 6. 验证机制

| 级别 | 检查内容 | 本次配置 |
| --- | --- | --- |
| 1 静态检查 | 语法、import 白名单、禁止危险调用、接口与 meta 一致性 | 仅针对生成的算子 |
| 2 沙箱执行 | 独立子进程，限制超时与内存，清除含 KEY/TOKEN 的环境变量 | 超时 180 s，内存 8192 MB |
| 3 契约检查 | 不修改输入、行数一致、不引入 NaN/Inf、概率合法、能处理未见类别与缺失值 | 冒烟样本 300 行 |
| 4 指标检查 | 阈值、种子间稳定性、疑似泄漏、耗时与内存 | 5 折 × 种子 [42, 7, 2024]；std 上限 0.03 |

最优方案资源消耗：单次训练 0.080 s，预测 35.8 ms / 千行，峰值内存 256 MB。

## 7. 知识沉淀

- 回写验证记录 ValidationRun：2 条
- 注册到能力库的新算子：IdColumnDropper@2
- 新提炼的经验：无
- 本次使用的经验：9 条（命中次数已累计）

## 附录 A. 生成的算法代码（pipeline.py）

```python
"""预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户 —— 自动生成的算法流水线

需求：预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户
方案：interpretable_logistic_regression
设计依据：针对电信客户流失预测任务的高可解释性与召回率优先（recall>=0.75）需求，首先清洗脏数据并剔除高基数ID列，通过在网时长分箱与One-Hot编码提升线性表达能力，经标准化后输入配置样本均衡权重的逻辑回归模型，系数直观透明且能有效捕捉流失客户。
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


import re
from typing import List, Optional
import pandas as pd
from sklearn.base import TransformerMixin

from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind


class IdColumnDropper(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="IdColumnDropper",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="剔除客户ID等标识列以防止特征维度暴涨与过拟合",
        conditions=["id_columns"],
        dependencies=[],
        params={
            "columns": None,
            "auto_detect": True,
            "pattern": r"(?i)(?:^|_)(?:id|uuid|guid|code|no)(?:$|_)|^customer_id$|^user_id$",
            "high_cardinality_threshold": 0.98,
        },
    )

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        auto_detect: bool = True,
        pattern: str = r"(?i)(?:^|_)(?:id|uuid|guid|code|no)(?:$|_)|^customer_id$|^user_id$",
        high_cardinality_threshold: float = 0.98,
    ):
        self.columns = columns
        self.auto_detect = auto_detect
        self.pattern = pattern
        self.high_cardinality_threshold = high_cardinality_threshold

    def fit(self, X: pd.DataFrame, y=None):
        cols_to_drop = set()

        if self.columns:
            for col in self.columns:
                if col in X.columns:
                    cols_to_drop.add(col)

        if self.auto_detect and len(X) > 0:
            regex = re.compile(self.pattern)
            for col in X.columns:
                col_str = str(col)
                # 检查列名是否匹配 ID 命名模式
                is_name_match = bool(regex.search(col_str))

                # 计算非空值的唯一值比例
                non_null_series = X[col].dropna()
                n_count = len(non_null_series)
                ratio = (non_null_series.nunique() / n_count) if n_count > 0 else 0.0

                # 若命中了名称规则，或者属于高基数字符串/对象标识列且唯一率极高
                if is_name_match:
                    cols_to_drop.add(col)
                elif pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_string_dtype(X[col]):
                    if ratio >= self.high_cardinality_threshold and n_count > 30:
                        # 辅以简单判断，避免误删长文本描述（均长较短更可能是标识符）
                        avg_len = non_null_series.astype(str).str.len().mean()
                        if avg_len <= 64:
                            cols_to_drop.add(col)

        self.dropped_columns_ = sorted(list(cols_to_drop))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if hasattr(self, "dropped_columns_") and self.dropped_columns_:
            existing_cols = [c for c in self.dropped_columns_ if c in X.columns]
            if existing_cols:
                X = X.drop(columns=existing_cols)
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
        description="清洗包含空白字符串的数值列并转换为浮点型，缺失值填充为指定值",
        conditions=["string_typed_numeric", "has_missing_values"],
        dependencies=[],
        params={"column": "total_charges", "fill_value": 0.0},
        parent="TotalChargesCleaner@1",
    )

    def __init__(self, column: str = "total_charges", fill_value: float = 0.0):
        self.column = column
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = list(X.columns)
        self.column_present_ = self.column in X.columns
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.column in X.columns:
            cleaned = pd.to_numeric(
                X[self.column].astype(str).str.strip(),
                errors="coerce"
            ).fillna(self.fill_value)
            X[self.column] = cleaned.astype(float)
        return X

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind, Modality


class TenureBucketizer(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="TenureBucketizer",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="将在网时长tenure字段按业务经验分箱为离散区间段，增强模型表现与可解释性",
        conditions=["time_tenure", "numeric_features"],
        dependencies=[],
        params={
            "input_col": "tenure",
            "output_col": "tenure_bucket",
            "bins": (-1, 6, 12, 24, 48, float("inf")),
            "labels": ("0-6", "6-12", "12-24", "24-48", "48+"),
            "fill_missing": "Unknown",
        },
        parent="TenureBucketizer@1",
    )

    def __init__(
        self,
        input_col="tenure",
        output_col="tenure_bucket",
        bins=(-1, 6, 12, 24, 48, float("inf")),
        labels=("0-6", "6-12", "12-24", "24-48", "48+"),
        fill_missing="Unknown",
    ):
        self.input_col = input_col
        self.output_col = output_col
        self.bins = bins
        self.labels = labels
        self.fill_missing = fill_missing

    def fit(self, X, y=None):
        self.bins_ = list(self.bins) if self.bins is not None else [-1, 6, 12, 24, 48, float("inf")]
        self.labels_ = list(self.labels) if self.labels is not None else ["0-6", "6-12", "12-24", "24-48", "48+"]
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        out_col = self.output_col if self.output_col else f"{self.input_col}_bucket"

        try:
            bins = self.bins_
        except AttributeError:
            bins = list(self.bins) if self.bins is not None else [-1, 6, 12, 24, 48, float("inf")]

        try:
            labels = self.labels_
        except AttributeError:
            labels = list(self.labels) if self.labels is not None else ["0-6", "6-12", "12-24", "24-48", "48+"]

        if self.input_col not in X.columns:
            X[out_col] = str(self.fill_missing)
            return X

        s = pd.to_numeric(X[self.input_col], errors="coerce")
        cut_res = pd.cut(s, bins=bins, labels=labels)

        bucket_s = cut_res.astype(str)
        bucket_s = bucket_s.replace({"nan": str(self.fill_missing), "NaN": str(self.fill_missing), "<NA>": str(self.fill_missing)})

        X[out_col] = bucket_s
        return X


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_feature", IdColumnDropper()),  # 根据经验提示，customer_id基数过高，直接编码会导致维度暴涨与内存溢出，需在预处理前剔除。
        ("1_imputer", TotalChargesCleaner(column='total_charges', fill_value=0.0)),  # 根据经验提示，total_charges列存在空白字符脏数据，需先清洗并安全转换为浮点数值。
        ("2_imputer", MissingValueImputer()),  # 处理 monthly_charges 与 payment_method 列的缺失值，保证数据完整性。
        ("3_feature", TenureBucketizer(input_col='tenure', output_col='tenure_bucket', bins=(-1, 6, 12, 24, 48, float('inf')), labels=('0-6', '6-12', '12-24', '24-48', '48+'), fill_missing='Unknown')),  # 将在网时长 tenure 字段按生命周期分箱离散化，增强线性模型的表达能力与业务可解释性。
        ("4_encoder", OneHotCategoricalEncoder(max_categories=20)),  # 对低基数类别特征进行 One-Hot 编码，转换为线性分类器可学习的稀疏向量。
        ("5_scaler", StandardFeatureScaler()),  # 逻辑回归对特征尺度敏感，必须对数值特征进行 Z-score 标准化以保证权重可解释与收敛。
        ("6_model", LogisticRegressionModel(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)),  # 满足任务严格的可解释性约束，配置 class_weight='balanced' 应对样本不平衡以确保召回率达标。
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

各阶段耗时：PLAN 0.2s，BUILD 0.0s，VERIFY 10.2s，ROUTE 0.0s，CURATE 0.0s

```
[PLAN] 场景 telecom_churn；画像标签 ['has_missing_values', 'numeric_features', 'has_categorical_features', 'low_cardinality', 'string_typed_numeric', 'id_columns', 'imbalanced', 'time_tenure', 'mixed_features']
[PLAN] TaskSpec：binary_classification，目标 churn，主指标 recall >= 0.75
[PLAN] 候选方案：['interpretable_logistic_regression(*IdColumnDropper → TotalChargesCleaner → MissingValueImputer → TenureBucketizer → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel)', 'balanced_lightgbm(*IdColumnDropper → TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → *BalancedLightGBMClassifier)']
[PLAN] 完成，用时 0.2s → BUILD
[BUILD] 生成算子 IdColumnDropper@2：剔除客户ID等标识列，防止One-Hot编码时维度暴涨引发内存溢出
[BUILD] 生成算子 BalancedLightGBMClassifier@2（参考 BalancedLightGBMClassifier@1）：使用 LightGBM 训练二分类模型，设置 class_weight='balanced' 自动平衡正负样本权重以提升召回率，并输出特征重要性
[BUILD] 完成，用时 0.0s → VERIFY
[VERIFY] interpretable_logistic_regression：通过，recall=0.7785
[VERIFY] balanced_lightgbm：未通过，recall=0.6243；M_THRESHOLD recall = 0.6243 低于阈值 0.75；M_THRESHOLD roc_auc = 0.7835 低于阈值 0.8
[VERIFY] 完成，用时 10.2s → ROUTE
[ROUTE] 完成，用时 0.0s → CURATE
[CURATE] 写入 KG：2 条 ValidationRun；注册新算子 ['IdColumnDropper@2']；新增经验 0 条
[CURATE] 完成，用时 0.0s → DONE
```
