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
        description="将total_charges列中的脏字符串（含空白符）转换为数值，并用0填补缺失值",
        conditions=["数据包含字符串类型的数值列", "存在因空白符导致的转换错误"],
        dependencies=["pandas", "numpy"],
        params={"col": "total_charges"},
        parent="TotalChargesCleaner@1"
    )

    def __init__(self, col="total_charges"):
        self.col = col

    def fit(self, X, y=None):
        """记录列名，该算子无需学习统计量"""
        self.col_ = self.col
        return self

    def transform(self, X):
        """
        执行转换：
        1. 复制输入以保证不可变性
        2. 处理字符串中的空白符并强制转换为浮点数
        3. 处理转换产生的缺失值（赋值为0）
        """
        X = X.copy()
        
        if self.col_ in X.columns:
            # 依据 Lessons 修复：先 strip，再 coerce，最后用 0 填充
            # 确保转换过程不抛出异常
            X[self.col_] = pd.to_numeric(
                X[self.col_].astype(str).str.strip(), 
                errors="coerce"
            ).fillna(0.0)
            
            # 处理可能存在的无穷值，确保输出的数值稳定性
            X[self.col_] = X[self.col_].replace([np.inf, -np.inf], 0.0)
            
        return X
