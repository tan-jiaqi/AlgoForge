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
        description="将total_charges字段中的空白字符串转换为数值，并处理异常值以符合后续模型计算需求",
        conditions=["数据包含 total_charges 列且该列含有空白字符或被识别为字符串"],
        dependencies=["pandas", "numpy"],
        params={"col": "total_charges"},
        parent="TotalChargesCleaner@1"
    )

    def __init__(self, col="total_charges"):
        self.col = col

    def fit(self, X, y=None):
        """记录列名并返回 self"""
        return self

    def transform(self, X):
        """
        修正 total_charges：
        1. 复制输入数据以防原地修改
        2. 将字符串两端的空白符清除
        3. 利用 pd.to_numeric 将非数值转为 NaN
        4. 使用 0 填充缺失值（代表新入网用户）
        """
        X = X.copy()
        if self.col in X.columns:
            # 执行类型转换：先转字符串并去除空格，coerce 非法值为 NaN，最后填补为 0
            X[self.col] = pd.to_numeric(
                X[self.col].astype(str).str.strip(), 
                errors="coerce"
            ).fillna(0.0)
            
            # 确保最终列为浮点型，并处理任何可能产生的 inf 值
            X[self.col] = X[self.col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
        return X
