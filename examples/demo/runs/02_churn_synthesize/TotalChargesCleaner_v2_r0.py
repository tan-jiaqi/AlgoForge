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
