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
