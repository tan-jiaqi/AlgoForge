"""预测30天内可能流失的客户，需可解释，召回优先 —— 自动生成的算法流水线

需求：预测30天内可能流失的客户，需可解释，召回优先
方案：lr_interpretable_pipeline
设计依据：该方案采用逻辑回归模型，通过特征分箱、标准化和平衡样本权重，在保证极佳可解释性的同时，优化流失预测的召回率。
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
        ("0_imputer", TotalChargesCleaner()),  # 解决total_charges中空白字符导致转换失败的问题，符合数值列含空白字符的经验。
        ("1_imputer", MissingValueImputer()),  # 对其他列的缺失值进行中位数和众数填补。
        ("2_feature", TenureBucketizer()),  # 对tenure进行分箱，提升逻辑回归等线性模型的表现并增强可解释性。
        ("3_encoder", OneHotCategoricalEncoder(max_categories=20)),  # 对低基数类别特征进行One-Hot编码。
        ("4_scaler", StandardFeatureScaler()),  # 逻辑回归模型必须先进行特征标准化，以保证系数可解释性。
        ("5_model", LogisticRegressionModel(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)),  # 使用逻辑回归满足可解释性约束，并设置class_weight='balanced'以应对样本不平衡，提升召回率。
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
