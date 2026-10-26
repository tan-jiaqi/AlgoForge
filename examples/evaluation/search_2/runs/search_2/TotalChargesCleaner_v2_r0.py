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
        description="将total_charges字段中的空白字符处理为0.0并转为数值类型",
        conditions=["数据包含字符串类型的数值列total_charges"],
        dependencies=["pandas", "numpy"],
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
            # 根据 Lesson:69d51358：先去除空白字符，再转为数值，最后填充缺失值
            X[self.col] = pd.to_numeric(
                X[self.col].astype(str).str.strip(), 
                errors="coerce"
            ).fillna(0.0)
            
            # 处理可能产生的无穷值
            X[self.col] = X[self.col].replace([np.inf, -np.inf], 0.0)
        return X
