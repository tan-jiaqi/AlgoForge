import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class ChargeRatioFeatureGenerator(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="ChargeRatioFeatureGenerator",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="生成消费比值特征，通过处理total_charges脏数据并计算总费与月费比例",
        conditions=["数据中包含total_charges, monthly_charges, tenure字段"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"monthly": "monthly_charges", "total": "total_charges", "tenure": "tenure"},
        parent="ChargeRatioFeatureGenerator@1"
    )

    def __init__(self, monthly="monthly_charges", total="total_charges", tenure="tenure"):
        self.monthly = monthly
        self.total = total
        self.tenure = tenure

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # 处理 total_charges 脏数据 (Lesson:69d51358)
        # 将空白字符串和非法字符转为数值，缺失值填补为 0
        if self.total in X.columns:
            X[self.total] = pd.to_numeric(X[self.total].astype(str).str.strip(), errors='coerce').fillna(0.0)
            
        # 确保 monthly_charges 和 tenure 为数值类型
        X[self.monthly] = pd.to_numeric(X[self.monthly], errors='coerce').fillna(0.0)
        X[self.tenure] = pd.to_numeric(X[self.tenure], errors='coerce').fillna(0.0)
        
        # 计算特征: avg_monthly = total / tenure (tenure=0时为0)
        # 避免除以0，tenure为0时比值记为1
        tenure_safe = X[self.tenure].replace(0, np.nan)
        avg_calc = X[self.total] / tenure_safe
        
        # 计算比值: (avg_calc / monthly)
        # 处理无穷值和缺失值
        ratio = avg_calc / X[self.monthly]
        X["charge_ratio"] = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        
        return X
