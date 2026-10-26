import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class DataPreprocessor(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="DataPreprocessor",
        kind=OperatorKind.FEATURE,
        version=1,
        origin="synthesized",
        description="删除标识列并处理字符串格式的数值特征，同时填补缺失值。",
        conditions=["输入包含customer_id及字符串数值列"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"id_columns": ["customer_id"], "numeric_cols": ["total_charges"]}
    )

    def __init__(self, id_columns=["customer_id"], numeric_cols=["total_charges"]):
        self.id_columns = id_columns
        self.numeric_cols = numeric_cols

    def fit(self, X, y=None):
        X = X.copy()
        self.medians_ = {}
        for col in self.numeric_cols:
            if col in X.columns:
                series = pd.to_numeric(X[col], errors='coerce')
                self.medians_[col] = series.median() if not series.isna().all() else 0.0
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. 删除标识列
        X = X.drop(columns=[c for c in self.id_columns if c in X.columns], errors='ignore')
        
        # 2. 转换指定数值列
        for col in self.numeric_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].replace([np.inf, -np.inf], np.nan)
                X[col] = X[col].fillna(self.medians_.get(col, 0.0))
        
        # 3. 对整体DataFrame其他列处理（规避复杂dtype导致issubdtype报错）
        for col in X.columns:
            if X[col].isnull().any():
                # 使用类型检查而非复杂的issubdtype，优先处理数值列
                if pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = X[col].fillna(X[col].median() if not X[col].dropna().empty else 0.0)
                else:
                    # 显式转换为对象类型以处理复杂类型缺失
                    X[col] = X[col].fillna("Missing").astype(str)
                    
        return X
