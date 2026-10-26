import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class TenureBucket(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="TenureBucket",
        kind=OperatorKind.FEATURE,
        version=1,
        origin="synthesized",
        description="将在网时长tenure进行分桶处理，便于模型学习流失高发期的非线性规律。",
        conditions=["输入包含tenure列", "tenure为数值型"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"col": "tenure"},
        parent="TenureBucketizer@1"
    )

    def __init__(self, col="tenure"):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.col not in X.columns:
            return X
        
        # 确保输入为数值类型，处理可能的非数值脏值
        temp_col = pd.to_numeric(X[self.col], errors="coerce")
        
        bins = [-1, 6, 12, 24, 48, np.inf]
        labels = ["0-6", "6-12", "12-24", "24-48", "48+"]
        
        # 分桶并处理缺失值
        bucketed = pd.cut(temp_col, bins=bins, labels=labels).astype(str)
        
        # 根据 lesson 经验，必须处理切分产生的 NaN (变为 'nan') 
        # 为了模型的鲁棒性，将其统一填充为一个分类标签
        bucketed = bucketed.replace("nan", "unknown")
        
        X[self.col + "_bucket"] = bucketed
        
        # 兜底处理可能产生的非法值
        X[self.col + "_bucket"] = X[self.col + "_bucket"].fillna("unknown")
        
        return X
