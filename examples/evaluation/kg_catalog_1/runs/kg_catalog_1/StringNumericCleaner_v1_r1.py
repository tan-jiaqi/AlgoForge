import pandas as pd
import numpy as np
import re
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class StringNumericCleaner(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="StringNumericCleaner",
        kind=OperatorKind.FEATURE,
        version=1,
        origin="synthesized",
        description="将字符串格式的数值列清洗并转换为浮点数，并确保输出无缺失值",
        conditions=["数据包含数值特征被错误读取为字符串"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"columns": None}
    )

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        # 记录需要处理的列，若未指定则自动识别 object 类型列
        if self.columns is None:
            self.numeric_cols_ = X.select_dtypes(include=['object']).columns.tolist()
        else:
            self.numeric_cols_ = self.columns
        
        # 计算每列中位数以便 transform 阶段填充
        self.medians_ = {}
        for col in self.numeric_cols_:
            if col in X.columns:
                # 尝试将原列转为数值获取中位数，若全为非数值则设为 0
                temp = pd.to_numeric(X[col], errors='coerce')
                self.medians_[col] = temp.median() if not temp.isnull().all() else 0.0
        return self

    def transform(self, X):
        X = X.copy()
        
        for col in self.numeric_cols_:
            if col not in X.columns:
                continue
            
            # 清洗字符串并转数值
            col_data = X[col].astype(str).str.strip()
            col_data = col_data.replace(r'^\s*$', np.nan, regex=True)
            
            # 转换为 float，非法值变 NaN
            converted = pd.to_numeric(col_data, errors='coerce')
            
            # 处理 inf
            converted = converted.replace([np.inf, -np.inf], np.nan)
            
            # 填充缺失值
            fill_val = self.medians_.get(col, 0.0)
            if np.isnan(fill_val):
                fill_val = 0.0
            
            X[col] = converted.fillna(fill_val)
            
        return X
