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
        description="构建当前月费与在网期间月均消费的比值特征，反映资费变动压力",
        conditions=["has_missing_values", "numeric_features", "string_typed_numeric"],
        dependencies=[],
        params={
            "monthly_col": "monthly_charges",
            "total_col": "total_charges",
            "tenure_col": "tenure",
            "new_col_name": "charge_ratio"
        },
        parent="ChargeRatioFeatureGenerator@1"
    )

    def __init__(self, monthly_col="monthly_charges", total_col="total_charges", tenure_col="tenure", new_col_name="charge_ratio"):
        self.monthly_col = monthly_col
        self.total_col = total_col
        self.tenure_col = tenure_col
        self.new_col_name = new_col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # 安全转换 total_charges（处理可能含空格的字符串型数值）
        total_series = X[self.total_col]
        if total_series.dtype == object:
            total_series = total_series.astype(str).str.strip()
        total = pd.to_numeric(total_series, errors='coerce').fillna(0.0)

        # 安全转换 tenure，并将 0 替换为 NaN 避免除零
        tenure_series = X[self.tenure_col]
        if tenure_series.dtype == object:
            tenure_series = tenure_series.astype(str).str.strip()
        tenure = pd.to_numeric(tenure_series, errors='coerce').replace(0, np.nan)

        # 安全转换 monthly_charges
        monthly_series = X[self.monthly_col]
        if monthly_series.dtype == object:
            monthly_series = monthly_series.astype(str).str.strip()
        monthly = pd.to_numeric(monthly_series, errors='coerce')

        # 计算在网期间的月均消费
        avg = total / tenure

        # 计算当前月费与月均消费的比值，处理无穷值与缺失值，默认填充为 1.0
        ratio = (avg / monthly).replace([np.inf, -np.inf], np.nan).fillna(1.0)

        X[self.new_col_name] = ratio
        return X
