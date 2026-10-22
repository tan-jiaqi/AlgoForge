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
