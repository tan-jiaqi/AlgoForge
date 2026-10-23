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
