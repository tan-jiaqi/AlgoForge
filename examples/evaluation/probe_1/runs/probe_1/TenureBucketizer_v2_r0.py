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
        description="对在网时长进行分箱离散化，将连续变量转化为业务可解释的分段特征，并处理非数值型脏数据。",
        conditions=["数据包含在网时长字段", "需要提升模型的可解释性"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"col": "tenure"},
        parent="TenureBucketizer@1"
    )

    def __init__(self, col="tenure"):
        self.col = col

    def fit(self, X, y=None):
        # 记录分箱逻辑状态
        self.bins_ = [-1, 6, 12, 24, 48, np.inf]
        self.labels_ = ["0-6", "6-12", "12-24", "24-48", "48+"]
        return self

    def transform(self, X):
        X = X.copy()
        if self.col not in X.columns:
            return X
        
        # 处理脏数据：将可能的非数值型字符串转为数值
        series = X[self.col].astype(str).str.strip()
        series = pd.to_numeric(series, errors='coerce')
        
        # 填补缺失值/无法转换的值：默认置为0
        series = series.fillna(0)
        
        # 执行分箱逻辑
        bucket_col = f"{self.col}_bucket"
        X[bucket_col] = pd.cut(
            series, 
            bins=self.bins_, 
            labels=self.labels_
        ).astype(str)
        
        # 处理可能产生的异常值：确保没有空值
        X[bucket_col] = X[bucket_col].replace("nan", "Unknown")
        
        return X
