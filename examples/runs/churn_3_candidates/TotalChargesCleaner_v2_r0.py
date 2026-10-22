import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class TotalChargesCleaner(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="TotalChargesCleaner",
        kind=OperatorKind.IMPUTER,
        version=2,
        origin="synthesized",
        description="将 total_charges 中的空白字符处理为缺失并转换为浮点型，填补为0.0",
        conditions=["string_typed_numeric"],
        dependencies=[],
        params={"col": "total_charges"},
        parent="TotalChargesCleaner@1"
    )

    def __init__(self, col="total_charges"):
        self.col = col

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        if self.col in X.columns:
            # 先将列转换为字符串并去除首尾空白字符，再转换为数值型（无法转换的变NaN），最后填补为0.0
            X[self.col] = pd.to_numeric(X[self.col].astype(str).str.strip(), errors="coerce").fillna(0.0)
        return X
