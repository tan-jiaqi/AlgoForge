import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class TotalChargesCleaner(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="TotalChargesCleaner",
        kind=OperatorKind.IMPUTER,
        version=2,
        origin="synthesized",
        description="将字符串类型的总费用列转换为数值，处理空白异常值并增加在网时长分桶特征",
        conditions=["数据包含total_charges列", "存在tenure在网时长列"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"col": "total_charges", "tenure_col": "tenure"},
        parent="TotalChargesCleaner@1"
    )

    def __init__(self, col="total_charges", tenure_col="tenure"):
        self.col = col
        self.tenure_col = tenure_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # 处理 total_charges 列：转换为数值，强制处理非数值为 0，并确保结果无缺失或无穷大
        if self.col in X.columns:
            # 强制转换为数值类型，无法转换的转为 NaN
            clean_series = pd.to_numeric(X[self.col].astype(str).str.strip(), errors='coerce')
            # 填充 NaN 和无穷值为 0
            X[self.col] = clean_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 处理 tenure_col 列：生成分桶特征
        if self.tenure_col in X.columns:
            bins = [0, 6, 12, 24, 48, 60, 1000]
            labels = ['0-6m', '7-12m', '13-24m', '25-48m', '49-60m', '60m+']
            
            tenure_vals = pd.to_numeric(X[self.tenure_col], errors='coerce').fillna(0)
            X['tenure_binned'] = pd.cut(
                tenure_vals, 
                bins=bins, 
                labels=labels, 
                include_lowest=True
            ).astype(str)
            
            # 确保分桶结果没有空值
            X['tenure_binned'] = X['tenure_binned'].fillna('unknown')

        return X
