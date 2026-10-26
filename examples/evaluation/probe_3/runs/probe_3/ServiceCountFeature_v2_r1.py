import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class ServiceCountFeature(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="ServiceCountFeature",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="统计客户订购的增值服务数量并计算累计费用与月费的比值，以量化客户粘性",
        conditions=["数据包含增值服务列", "数据包含月费与总费用列"],
        dependencies=["numpy", "pandas", "sklearn"],
        params={
            "service_cols": ["online_security", "tech_support", "streaming_tv"],
            "monthly_col": "monthly_charges",
            "total_col": "total_charges"
        },
        parent="ServiceCountFeature@1"
    )

    def __init__(self, service_cols=["online_security", "tech_support", "streaming_tv"], monthly_col="monthly_charges", total_col="total_charges"):
        self.service_cols = service_cols
        self.monthly_col = monthly_col
        self.total_col = total_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. 统计增值服务数量 (假设“是”为订购)
        # 将非“是”的值视作未订购，处理缺失值风险
        X["n_services"] = 0
        for col in self.service_cols:
            if col in X.columns:
                X["n_services"] += (X[col].astype(str).str.strip() == "是").astype(int)
        
        # 2. 处理数值列：清理空白字符并转为 float
        # 针对 total_charges 可能存在的 ' ' 空格字符串问题
        for col in [self.monthly_col, self.total_col]:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col].astype(str).str.strip(), errors='coerce').fillna(0.0)
        
        # 3. 计算累计费用/月费比值 (隐含在网时长特征)
        # 规避零分母：若 monthly_charges 为 0 则比值设为 0
        denominator = X[self.monthly_col].replace(0, np.nan)
        X["charge_ratio"] = (X[self.total_col] / denominator).fillna(0.0)
        
        # 4. 处理最终可能产生的无穷值或缺失值
        X["charge_ratio"] = X["charge_ratio"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        
        return X
