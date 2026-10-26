import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind, Modality

class TotalChargesCleaner(TransformerMixin, Operator):
    """
    清理包含空白符或脏字符的total_charges列并转换为浮点型，缺失值默认为0。
    """
    meta = OperatorMeta(
        name="TotalChargesCleaner",
        kind=OperatorKind.IMPUTER,
        version=2,
        origin="synthesized",
        description="清理包含空白符的total_charges列并转换为浮点型",
        conditions=["数据存在字符串格式的数值列", "存在包含空白字符的缺失值"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"col": "total_charges"},
        parent="TotalChargesCleaner@1"
    )

    def __init__(self, col="total_charges"):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.col in X.columns:
            # 根据Lesson:69d51358经验：先strip，再转数值（强制转换空值），最后填充缺失值
            X[self.col] = pd.to_numeric(
                X[self.col].astype(str).str.strip(), 
                errors="coerce"
            ).fillna(0.0)
            
            # 确保转换后没有无穷值，保持特征稳定性
            X[self.col] = X[self.col].replace([np.inf, -np.inf], 0.0)
            
        return X
