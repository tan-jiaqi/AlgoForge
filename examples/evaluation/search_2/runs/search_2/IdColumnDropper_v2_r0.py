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
        description="删除客户ID等无预测价值的特征，防止后续建模出现维度爆炸。",
        conditions=["数据包含客户ID等唯一标识列"],
        dependencies=[],
        params={"id_cols": ("customer_id",)},
        parent="IdColumnDropper@1"
    )

    def __init__(self, id_cols=("customer_id",)):
        self.id_cols = id_cols

    def fit(self, X, y=None):
        """记录应当删除的列，不执行计算。"""
        self.cols_to_drop_ = [c for c in self.id_cols if c in X.columns]
        return self

    def transform(self, X):
        """安全删除指定列，确保不修改原数据。"""
        if not isinstance(X, pd.DataFrame):
            return X
        
        X_copy = X.copy()
        existing_cols_to_drop = [c for c in self.cols_to_drop_ if c in X_copy.columns]
        if existing_cols_to_drop:
            X_copy = X_copy.drop(columns=existing_cols_to_drop)
            
        return X_copy
