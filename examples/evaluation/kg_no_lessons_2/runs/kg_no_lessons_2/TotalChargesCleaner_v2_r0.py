import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class TotalChargesCleaner(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="TotalChargesCleaner",
        kind=OperatorKind.IMPUTER,
        version=2,
        origin="synthesized",
        description="将total_charges列中的脏字符串数据转换为浮点型，并使用0填充缺失值",
        conditions=["数据包含字符串类型的数值列", "存在包含空白字符的无效数据"],
        dependencies=["pandas", "numpy"],
        params={"col": "total_charges"},
        parent="TotalChargesCleaner@1"
    )

    def __init__(self, col="total_charges"):
        self.col = col

    def fit(self, X, y=None):
        """记录列名并返回自身，无需学习状态"""
        return self

    def transform(self, X):
        """将指定列转换为数值类型，处理无效字符串并填充缺失值"""
        X = X.copy()
        if self.col in X.columns:
            # 1. 强制转为字符串并去除首尾空白
            # 2. to_numeric 将无法转换的字符串(如空字符、非数值串)转为 NaN
            # 3. 用 0.0 填充转换后产生的 NaN (针对新客户或脏数据)
            X[self.col] = pd.to_numeric(
                X[self.col].astype(str).str.strip(), 
                errors="coerce"
            ).fillna(0.0)
            
            # 处理可能产生的无穷大值，确保数据健壮性
            X[self.col] = X[self.col].replace([np.inf, -np.inf], 0.0)
            
        return X
