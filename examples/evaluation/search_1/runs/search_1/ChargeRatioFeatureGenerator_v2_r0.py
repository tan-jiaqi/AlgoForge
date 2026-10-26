import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class ChargeRatioFeatureGenerator(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="ChargeRatioFeatureGenerator",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="基于月费与累计消费构造资费波动比值特征，处理数值转换异常与除零风险",
        conditions=["数据包含tenure, monthly_charges, total_charges列"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"monthly": "monthly_charges", "total": "total_charges", "tenure": "tenure"},
        parent="ChargeRatioFeatureGenerator@1"
    )

    def __init__(self, monthly="monthly_charges", total="total_charges", tenure="tenure"):
        self.monthly = monthly
        self.total = total
        self.tenure = tenure

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. 确保数值列处理：先剔除空格，转数值，缺失值填充为0 (Lesson:69d51358)
        # 将 total_charges 和 monthly_charges 强制转为 float
        for col in [self.total, self.monthly, self.tenure]:
            if col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = pd.to_numeric(X[col].astype(str).str.strip(), errors='coerce')
                X[col] = X[col].fillna(0)

        # 2. 计算平均费用，处理 tenure 为 0 的情况
        # avg = total / tenure。若 tenure 为 0，视为 0 消费
        tenure_safe = X[self.tenure].replace(0, np.nan)
        avg_charge = X[self.total] / tenure_safe
        
        # 3. 计算资费比值：avg / current_monthly
        # 如果当前月费为0，避免除以0产生inf
        monthly_safe = X[self.monthly].replace(0, np.nan)
        ratio = avg_charge / monthly_safe
        
        # 4. 处理无穷大及空值 (Lesson:69d51358)
        # 若计算出现inf或NaN，填充为1.0（表示当前月费与历史平均一致）
        X["charge_ratio"] = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        
        return X
