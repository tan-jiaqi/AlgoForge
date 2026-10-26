import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class ChargeRatioFeatureGenerator(TransformerMixin, Operator):
    """生成消费比值特征以捕捉行为特征，通过月均消费与当前月费的比值衡量资费变化。"""
    
    meta = OperatorMeta(
        name="ChargeRatioFeatureGenerator",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="计算月均消费与当前月费的比值，衡量用户近期资费变动情况。",
        conditions=["数据包含monthly_charges, total_charges, tenure字段"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"monthly": "monthly_charges", "total": "total_charges", "tenure": "tenure"},
        parent="ChargeRatioFeatureGenerator@1"
    )

    def __init__(self, monthly="monthly_charges", total="total_charges", tenure="tenure"):
        self.monthly = monthly
        self.total = total
        self.tenure = tenure

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. 确保目标列为数值类型，处理潜在的空格字符串和缺失值
        # 根据 Lesson:69d51358，进行 strip 和强制类型转换
        total_col = pd.to_numeric(X[self.total].astype(str).str.strip(), errors='coerce').fillna(0.0)
        monthly_col = pd.to_numeric(X[self.monthly].astype(str).str.strip(), errors='coerce').fillna(0.0)
        tenure_col = pd.to_numeric(X[self.tenure].astype(str).str.strip(), errors='coerce')
        
        # 2. 计算平均费用，处理 tenure 为 0 的情况
        # 避免除以 0，当 tenure 为 0 时，假设平均费用为 0 或由 monthly_charges 决定
        avg_charge = total_col / tenure_col.replace(0, np.nan)
        
        # 3. 计算比值
        # 防止除以 0 (monthly_charges 为 0)
        ratio = avg_charge / monthly_col.replace(0, np.nan)
        
        # 4. 根据 Lesson:6375318f，显式处理 Inf 和 NaN
        # 如果 tenure 为 0 或其他异常情况导致比值无效，统一填补为 1.0 (代表资费持平)
        X["charge_ratio"] = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        
        return X
