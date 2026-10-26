import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class IdColumnDropper(TransformerMixin, Operator):
    """
    删除客户ID及其他高基数标识列，防止后续编码时出现维度爆炸，并确保输出数据结构稳定。
    """
    meta = OperatorMeta(
        name="IdColumnDropper",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="删除指定的标识列，防止高基数特征导致的维度爆炸问题。",
        conditions=["数据包含标识列", "需要进行特征编码"],
        dependencies=[],
        params={"id_cols": ("customer_id",)},
        parent="IdColumnDropper@1"
    )

    def __init__(self, id_cols=("customer_id",)):
        self.id_cols = id_cols

    def fit(self, X, y=None):
        self.id_cols_ = list(self.id_cols)
        return self

    def transform(self, X):
        X = X.copy()
        # 识别当前 DataFrame 中存在的标识列
        cols_to_drop = [c for c in self.id_cols_ if c in X.columns]
        
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
        
        # 确保算子输出的健壮性：处理可能的 NaN/Inf
        # 尽管此算子仅做删除，但考虑到数据流完整性，在此处进行显式兜底校验
        X = X.replace([np.inf, -np.inf], np.nan)
        
        return X
