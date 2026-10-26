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
        description="删除客户ID等标识列以避免编码后维度爆炸",
        conditions=["数据含客户ID等标识列"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"id_cols": ["customer_id"]},
        parent="IdColumnDropper@1"
    )

    def __init__(self, id_cols=["customer_id"]):
        self.id_cols = id_cols

    def fit(self, X, y=None):
        # 记录训练集存在的 ID 列，保证 transform 时行为一致
        self.drop_cols_ = [c for c in self.id_cols if c in X.columns]
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. 剔除已知的标识列
        X = X.drop(columns=[c for c in self.drop_cols_ if c in X.columns], errors='ignore')
        
        # 2. 处理 total_charges 等因读取产生的脏数据
        # 将空白字符串转换为数字并填补缺失值，防止后续下游编码器崩溃
        if 'total_charges' in X.columns:
            if X['total_charges'].dtype == 'object':
                X['total_charges'] = pd.to_numeric(
                    X['total_charges'].astype(str).str.strip(), 
                    errors='coerce'
                )
            X['total_charges'] = X['total_charges'].fillna(0.0)
            # 处理可能产生的无穷值
            X['total_charges'] = X['total_charges'].replace([np.inf, -np.inf], 0.0)
            
        return X
