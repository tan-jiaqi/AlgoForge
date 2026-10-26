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
        description="将字符串类型的 total_charges 列转换为浮点型，处理空白值并填充缺失值",
        conditions=["包含 total_charges 字符串列"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"col": "total_charges"},
        parent="TotalChargesCleaner@1"
    )

    def __init__(self, col="total_charges"):
        self.col = col

    def fit(self, X, y=None):
        """记录列名并返回 self，该算子无状态学习。"""
        self.col_ = self.col
        return self

    def transform(self, X):
        """
        处理逻辑：
        1. 复制输入数据防止污染原数据
        2. 将指定列转为字符串并去除首尾空白
        3. 使用 pd.to_numeric 将无法解析的值转为 NaN
        4. 使用 0.0 填充转换后的 NaN（针对新客户逻辑）
        5. 处理可能存在的无穷值并填充为 0.0
        """
        X = X.copy()
        if self.col_ in X.columns:
            # 1. 清洗及类型转换
            clean_series = pd.to_numeric(
                X[self.col_].astype(str).str.strip(), 
                errors="coerce"
            )
            
            # 2. 填充缺失值
            clean_series = clean_series.fillna(0.0)
            
            # 3. 处理可能产生的无穷值
            clean_series = clean_series.replace([np.inf, -np.inf], 0.0)
            
            X[self.col_] = clean_series
            
        return X
