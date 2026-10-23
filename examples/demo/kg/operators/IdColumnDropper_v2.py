from typing import Sequence
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
        description="移除客户ID等无业务预测意义的标识列，防止编码维度暴涨和过拟合",
        conditions=["id_columns"],
        dependencies=[],
        params={"id_cols": ("customer_id",)},
        parent="IdColumnDropper@1",
    )

    def __init__(self, id_cols: Sequence[str] = ("customer_id",)):
        self.id_cols = id_cols

    def fit(self, X: pd.DataFrame, y=None):
        cols_to_check = set(self.id_cols) if self.id_cols is not None else set()
        self.drop_cols_ = [col for col in X.columns if col in cols_to_check]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cols_to_drop = [c for c in self.drop_cols_ if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
        return X
