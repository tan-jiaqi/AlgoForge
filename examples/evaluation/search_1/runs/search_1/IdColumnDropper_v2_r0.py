import pandas as pd
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class IdColumnDropper(TransformerMixin, Operator):
    """
    删除客户ID等标识列，防止维度爆炸并避免过拟合。
    遵循 Lesson:88299a38，在特征工程初期强制剔除标识列。
    """
    meta = OperatorMeta(
        name="IdColumnDropper",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="删除客户ID等标识列，防止维度爆炸并避免过拟合",
        conditions=["id_columns"],
        dependencies=["pandas", "sklearn"],
        params={"id_cols": ("customer_id",)},
        parent="IdColumnDropper@1"
    )

    def __init__(self, id_cols=("customer_id",)):
        self.id_cols = id_cols

    def fit(self, X, y=None):
        # 记录已识别的待删除列名
        self.columns_to_drop_ = [c for c in self.id_cols if c in X.columns]
        return self

    def transform(self, X):
        X = X.copy()
        # 过滤掉不存在的列，防止推理时报错
        actual_drop = [c for c in self.columns_to_drop_ if c in X.columns]
        if actual_drop:
            X = X.drop(columns=actual_drop)
        return X
