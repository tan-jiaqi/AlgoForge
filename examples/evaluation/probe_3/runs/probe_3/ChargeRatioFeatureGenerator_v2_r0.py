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
        description="构造累计平均费用与月费的比值特征，自动处理类型转换、零值及缺失值。",
        conditions=["数据包含 monthly_charges, total_charges, tenure 列"],
        dependencies=["numpy", "pandas", "sklearn"],
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
        
        # 转换 total_charges 为数值，处理空白符
        if self.total in X.columns:
            X[self.total] = pd.to_numeric(X[self.total].astype(str).str.strip(), errors='coerce').fillna(0)
            
        # 确保其他列为数值
        X[self.monthly] = pd.to_numeric(X[self.monthly], errors='coerce').fillna(0)
        X[self.tenure] = pd.to_numeric(X[self.tenure], errors='coerce').fillna(0)
        
        # 计算 avg_charges = total_charges / tenure
        # tenure 为 0 或缺失时，avg_charges 处理为 0
        avg_charges = X[self.total] / X[self.tenure].replace(0, np.nan)
        avg_charges = avg_charges.fillna(0)
        
        # 计算 ratio = avg_charges / monthly_charges
        # 若 monthly_charges 为 0，结果为 inf，随后统一处理
        ratio = avg_charges / X[self.monthly].replace(0, np.nan)
        
        # 处理 inf 和 NaN，缺失值填补为 1.0 (代表平均值与月费持平)
        X["charge_ratio"] = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        
        return X
