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
        self.mean_total_charges_ = 0.0
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. 清洗 total_charges 列
        # Lesson:69d51358 - 处理空白字符串导致转换失败的问题
        if self.col in X.columns:
            X[self.col] = pd.to_numeric(
                X[self.col].astype(str).str.strip(), 
                errors="coerce"
            ).fillna(0.0)
            
            # 确保不存在无穷值
            X[self.col] = X[self.col].replace([np.inf, -np.inf], 0.0)

        # 2. 增加在网时长分桶特征 (tenure_binned)
        # 增加可解释性特征，将持续时间分为阶段
        if self.tenure_col in X.columns:
            bins = [0, 6, 12, 24, 48, 60, 100]
            labels = ['0-6m', '7-12m', '13-24m', '25-48m', '49-60m', '60m+']
            X['tenure_binned'] = pd.cut(
                X[self.tenure_col], 
                bins=bins, 
                labels=labels, 
                include_lowest=True
            ).astype(str)
            
            # 处理分桶后的缺失值
            X['tenure_binned'] = X['tenure_binned'].fillna('unknown')

        return X
