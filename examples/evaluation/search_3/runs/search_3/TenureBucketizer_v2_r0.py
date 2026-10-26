import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class TenureBucketizer(TransformerMixin, Operator):
    """
    在网时长分箱，将 tenure 离散化为 [0-6, 6-12, 12-24, 24-48, 48+]，提升模型对不同生命周期用户的识别。
    """
    meta = OperatorMeta(
        name="TenureBucketizer",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="在网时长分箱，提升模型对不同生命周期用户的识别",
        conditions=["数据包含数值列 tenure"],
        dependencies=["pandas", "numpy"],
        params={"col": "tenure"},
        parent="TenureBucketizer@1"
    )

    def __init__(self, col="tenure"):
        self.col = col

    def fit(self, X, y=None):
        # 记录转换逻辑所需状态
        self.bins_ = [-1, 6, 12, 24, 48, np.inf]
        self.labels_ = ["0-6", "6-12", "12-24", "24-48", "48+"]
        return self

    def transform(self, X):
        X = X.copy()
        
        # 处理可能的缺失值和类型异常：先确保数值化
        if self.col in X.columns:
            # 使用 pd.to_numeric 处理可能存在的字符串格式或非法值
            col_data = pd.to_numeric(X[self.col], errors='coerce')
            
            # 对缺失值补 0 或其他填充策略（参考 lessons: 填补为 0）
            col_data = col_data.fillna(0)
            
            # 执行分箱逻辑
            bucket_name = f"{self.col}_bucket"
            X[bucket_name] = pd.cut(
                col_data, 
                bins=self.bins_, 
                labels=self.labels_
            ).astype(str)
            
            # 处理分箱后可能产生的 NaN (例如超出 bin 范围)
            X[bucket_name] = X[bucket_name].fillna("unknown")
            
        return X
