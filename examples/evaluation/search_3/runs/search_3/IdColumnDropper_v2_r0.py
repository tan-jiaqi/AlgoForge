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
        description="自动识别并剔除高基数ID类列，防止模型过拟合及维度灾难",
        conditions=["数据集包含唯一标识列"],
        dependencies=["pandas", "numpy"],
        params={"id_cols": ("customer_id",)},
        parent="IdColumnDropper@1"
    )

    def __init__(self, id_cols=("customer_id",)):
        self.id_cols = id_cols

    def fit(self, X, y=None):
        """
        验证输入数据结构并记录状态。
        """
        self.cols_to_drop_ = [c for c in self.id_cols if c in X.columns]
        return self

    def transform(self, X):
        """
        执行删除操作。
        """
        X = X.copy()
        
        # 剔除预定义的ID列
        cols_actually_dropped = [c for c in self.cols_to_drop_ if c in X.columns]
        X = X.drop(columns=cols_actually_dropped)
        
        # 逻辑兜底：检查是否存在其他疑似高基数ID的列（超过行数90%的唯一值）
        # 避免在线上推理阶段因为动态列名导致的问题
        potential_ids = [
            col for col in X.select_dtypes(include=['object']).columns 
            if X[col].nunique() / len(X) > 0.9
        ]
        X = X.drop(columns=potential_ids)
        
        return X
