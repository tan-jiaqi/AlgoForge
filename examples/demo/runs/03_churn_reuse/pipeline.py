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
