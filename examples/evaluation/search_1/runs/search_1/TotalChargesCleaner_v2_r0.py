import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind, Modality

class TotalChargesCleaner(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="TotalChargesCleaner",
        kind=OperatorKind.IMPUTER,
        version=2,
        origin="synthesized",
        description="将total_charges列中的空白字符串转换为数值，并将缺失值填充为0.0",
        conditions=["数据中存在包含空白字符的数值列"],
        dependencies=["pandas", "numpy"],
        params={"col": "total_charges"},
        parent="TotalChargesCleaner@1"
    )

    def __init__(self, col="total_charges"):
        self.col = col

    def fit(self, X, y=None):
        """记录列名以供后续调用"""
        self.col_ = self.col
        return self

    def transform(self, X):
        """
        对输入数据进行深拷贝，将目标列转为float类型，
        空白字符串及无法转换的值转为NaN，随后填充为0.0。
        """
        X = X.copy()
        if self.col_ in X.columns:
            # 依据Lesson:69d51358：先去除首尾空白，利用coerce转为数值，最后填充缺失为0
            X[self.col_] = pd.to_numeric(
                X[self.col_].astype(str).str.strip(), 
                errors="coerce"
            ).fillna(0.0)
            
            # 确保即使存在inf值也进行清理，防止后续模型报错
            X[self.col_] = X[self.col_].replace([np.inf, -np.inf], 0.0)
            
        return X
