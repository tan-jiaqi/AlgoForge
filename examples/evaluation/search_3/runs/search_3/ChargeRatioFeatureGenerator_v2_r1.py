import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class ChargeRatioFeatureGenerator(TransformerMixin, Operator):
    """
    计算月均消费与当前月费的比值特征，衡量用户消费变动趋势。
    """
    meta = OperatorMeta(
        name="ChargeRatioFeatureGenerator",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="构造月均消费与当前月费的比值特征，捕捉消费变动趋势",
        conditions=["输入数据包含月费、总费用及在网时长字段"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"monthly": "monthly_charges", "total": "total_charges", "tenure": "tenure"},
        parent="ChargeRatioFeatureGenerator@1"
    )

    def __init__(self, monthly="monthly_charges", total="total_charges", tenure="tenure"):
        self.monthly = monthly
        self.total = total
        self.tenure = tenure

    def fit(self, X, y=None):
        self.columns_ = X.columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        
        # 预处理：确保数值列类型正确，处理空白字符串及非数值
        # 先转为字符串去除空格，再转为数值，缺失值填充为0
        total_series = pd.to_numeric(X[self.total].astype(str).str.strip(), errors='coerce').fillna(0.0)
        tenure_series = pd.to_numeric(X[self.tenure], errors='coerce').fillna(0.0)
        monthly_series = pd.to_numeric(X[self.monthly], errors='coerce').fillna(0.0)
        
        # 计算逻辑：平均消费 = 总消费 / 在网时长
        # 为避免除以0，使用 np.where 处理 tenure 为 0 的情况
        avg_charge = np.where(tenure_series > 0, total_series / tenure_series, 0.0)
        
        # 计算比值：charge_ratio = avg_charge / current_monthly_charge
        # 为避免除以0，当月费为0时，结果置为 1.0 (表示无变动)
        ratio = np.where(monthly_series > 0, avg_charge / monthly_series, 1.0)
        
        # 处理可能产生的无穷值与缺失值
        res = pd.Series(ratio, index=X.index)
        X["charge_ratio"] = res.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        
        return X
