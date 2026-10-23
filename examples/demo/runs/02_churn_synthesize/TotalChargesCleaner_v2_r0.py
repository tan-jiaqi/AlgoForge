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
