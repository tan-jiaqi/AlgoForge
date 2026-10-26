import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class TenureBucketizer(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="TenureBucketizer",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="将数值型在网时长特征划分为业务导向的区间，增强线性模型的解释性。",
        conditions=["输入必须包含数值型tenure列"],
        dependencies=["numpy", "pandas", "sklearn"],
        params={"col": "tenure"},
        parent="TenureBucketizer@1"
    )

    def __init__(self, col="tenure"):
        self.col = col

    def fit(self, X, y=None):
        """记录算子状态，本算子为无状态转换，仅返回self。"""
        self.bins_ = [-1, 6, 12, 24, 48, np.inf]
        self.labels_ = ["0-6", "6-12", "12-24", "24-48", "48+"]
        return self

    def transform(self, X):
        """
        将tenure列分桶，处理缺失值及非数值情况。
        遵循数据清洗规范：先处理潜在脏数据，确保输出类型一致。
        """
        X = X.copy()
        
        # 处理脏数据：先确保转为数值，无法转换的(含空值)处理为-1（即处于0-6区间内）
        raw_tenure = pd.to_numeric(X[self.col].astype(str).str.strip(), errors='coerce')
        raw_tenure = raw_tenure.fillna(-1)
        
        # 分桶操作
        bucket_col = f"{self.col}_bucket"
        X[bucket_col] = pd.cut(
            raw_tenure, 
            bins=self.bins_, 
            labels=self.labels_
        ).astype(str)
        
        # 确保分桶后的缺失值被处理（pd.cut可能在NaN输入下产生nan字符串）
        X[bucket_col] = X[bucket_col].replace("nan", "0-6")
        
        return X
