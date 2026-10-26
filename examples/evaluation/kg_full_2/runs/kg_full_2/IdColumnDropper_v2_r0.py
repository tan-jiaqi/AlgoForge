import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class IdColumnDropper(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="IdColumnDropper",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="剔除客户 ID 等高基数标识列，防止维度爆炸和过拟合",
        conditions=["数据包含高基数标识列"],
        dependencies=[],
        params={"id_cols": ["customer_id"]},
        parent="IdColumnDropper@1"
    )

    def __init__(self, id_cols=["customer_id"]):
        self.id_cols = id_cols

    def fit(self, X, y=None):
        """记录输入列以供校验，此处仅返回self"""
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):
        """
        剔除指定的 ID 列。
        根据踩坑经验：必须先 copy 原始数据，防止原地修改影响下游；
        处理时需确保即使列不存在也不报错。
        """
        X = X.copy()
        cols_to_drop = [c for c in self.id_cols if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
        return X
