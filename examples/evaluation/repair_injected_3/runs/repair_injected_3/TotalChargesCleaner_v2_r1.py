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
        # 记录清洗后的均值或固定值用于兜底
        self.mean_val_ = 0.0
        return self

    def transform(self, X):
        X = X.copy()
        
        # 处理 total_charges 列
        if self.col in X.columns:
            # 强制转换为数值，空白及非法字符变为 NaN
            X[self.col] = pd.to_numeric(X[self.col].astype(str).str.strip(), errors="coerce")
            # 处理 NaN 和 Inf，统一填充为 0.0
            X[self.col] = X[self.col].fillna(0.0).replace([np.inf, -np.inf], 0.0)
            
        # 处理 tenure 分桶特征
        if self.tenure_col in X.columns:
            bins = [0, 6, 12, 24, 48, 60, 1000]
            labels = ["0-6m", "7-12m", "13-24m", "25-48m", "49-60m", "60m+"]
            # 确保输入为数值进行分桶
            tenure_series = pd.to_numeric(X[self.tenure_col], errors="coerce")
            X["tenure_binned"] = pd.cut(
                tenure_series, 
                bins=bins, 
                labels=labels, 
                include_lowest=True
            ).astype(str)
            # 处理分桶过程中产生的 NaN (即原始数据中缺失或超出范围的 tenure)
            X["tenure_binned"] = X["tenure_binned"].replace("nan", "unknown").fillna("unknown")
            
        return X
