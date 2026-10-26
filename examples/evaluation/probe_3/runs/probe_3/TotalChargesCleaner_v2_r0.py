import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind, Modality

class TotalChargesCleaner(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="TotalChargesCleaner",
        kind=OperatorKind.IMPUTER,
        version=2,
        origin="synthesized",
        description="将TotalCharges列的字符串清洗并转换为数值，同时生成累计费用与月费的比值特征",
        conditions=["数据包含 total_charges 和 monthly_charges 列"],
        dependencies=["pandas", "numpy"],
        params={"col": "total_charges", "monthly_col": "monthly_charges"},
        parent="TotalChargesCleaner@1"
    )

    def __init__(self, col="total_charges", monthly_col="monthly_charges"):
        self.col = col
        self.monthly_col = monthly_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # 清洗 total_charges：去空格、转数值、空值填0
        X[self.col] = pd.to_numeric(
            X[self.col].astype(str).str.strip(), 
            errors="coerce"
        ).fillna(0.0)
        
        # 确保 monthly_charges 也处理缺失值
        X[self.monthly_col] = pd.to_numeric(X[self.monthly_col], errors="coerce").fillna(0.0)
        
        # 特征工程：生成费用比值特征
        # 避免除以0，若月费为0则比值为0
        ratio_col = f"{self.col}_to_{self.monthly_col}_ratio"
        X[ratio_col] = np.where(
            X[self.monthly_col] > 0,
            X[self.col] / X[self.monthly_col],
            0.0
        )
        
        # 处理异常值（如无穷大）
        X[ratio_col] = X[ratio_col].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        
        return X
