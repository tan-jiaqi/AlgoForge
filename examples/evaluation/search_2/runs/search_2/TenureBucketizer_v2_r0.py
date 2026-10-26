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
        description="对tenure进行分箱以辅助模型识别流失周期",
        conditions=["存在tenure数值列"],
        dependencies=["numpy", "pandas", "sklearn"],
        params={"col": "tenure"},
        parent="TenureBucketizer@1"
    )

    def __init__(self, col="tenure"):
        self.col = col
        self.bins_ = [-1, 6, 12, 24, 48, np.inf]
        self.labels_ = ["0-6", "6-12", "12-24", "24-48", "48+"]

    def fit(self, X, y=None):
        # 记录转换状态，以便后续验证
        self.fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        if self.col not in X.columns:
            return X
            
        # 确保输入列为数值型，处理脏数据
        col_data = pd.to_numeric(X[self.col], errors='coerce')
        
        # 分箱处理，包含缺失值处理（缺失值会被切分为 NaN，astype(str)转为'nan'）
        # 如果需要保持原列类型，可以保留缺失值而不填充
        bucketed = pd.cut(col_data, bins=self.bins_, labels=self.labels_)
        
        # 将分箱结果合并，将未匹配的（含原缺失值）统一映射为 'unknown' 或保持字符串化
        X[f"{self.col}_bucket"] = bucketed.astype(str).replace('nan', 'unknown')
        
        return X
