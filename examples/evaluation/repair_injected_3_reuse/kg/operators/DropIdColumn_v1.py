import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class DropIdColumn(TransformerMixin, Operator):
    """
    删除客户ID及高基数标识列，防止后续处理（如One-Hot编码）导致维度爆炸。
    """
    meta = OperatorMeta(
        name="DropIdColumn",
        kind=OperatorKind.FEATURE,
        version=1,
        origin="synthesized",
        description="删除指定的ID列或高基数标识列，防止维度爆炸并保证数据完整性。",
        conditions=["数据包含customer_id等标识列"],
        dependencies=["pandas", "numpy"],
        params={"id_cols": ("customer_id",)},
        parent="IdColumnDropper@1"
    )

    def __init__(self, id_cols=("customer_id",)):
        self.id_cols = id_cols

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = X.copy()
        
        # 识别存在于列中的待删除ID列
        cols_to_drop = [c for c in self.id_cols if c in X.columns]
        X = X.drop(columns=cols_to_drop)
        
        # 根据 Lesson:cf247d47，确保输出不包含 NaN 或 Inf
        # 对于数值列，填充缺失值为0或中位数，此处选择填充为0以保证鲁棒性
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
            X[numeric_cols] = X[numeric_cols].fillna(0)
            
        return X
