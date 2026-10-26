import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind, Modality

class ChargeRatioFeatureGenerator(TransformerMixin, Operator):
    """
    生成消费比值特征：月均消费(total/tenure) / 当前月费。
    衡量客户长期平均消费水平与当前资费水平的差异，用于捕捉流失倾向。
    """
    meta = OperatorMeta(
        name="ChargeRatioFeatureGenerator",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="基于月费、总费及在网时长构造比值特征，衡量资费变动。",
        conditions=["输入列包含 monthly_charges, total_charges, tenure"],
        dependencies=["numpy", "pandas", "sklearn"],
        params={
            "monthly": "monthly_charges",
            "total": "total_charges",
            "tenure": "tenure"
        },
        parent="ChargeRatioFeatureGenerator@1"
    )

    def __init__(self, monthly="monthly_charges", total="total_charges", tenure="tenure"):
        self.monthly = monthly
        self.total = total
        self.tenure = tenure

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()

        def clean_numeric(series):
            # 严格处理：先转字符串，去除空格，再转数值，强制coerce无法转换的为NaN
            if series.dtype == 'object':
                series = series.astype(str).str.strip()
                # 排除可能存在的空字符串，将其视为 NaN
                series = series.replace(r'^\s*$', np.nan, regex=True)
                series = pd.to_numeric(series, errors='coerce')
            else:
                series = pd.to_numeric(series, errors='coerce')
            return series.fillna(0.0)

        # 获取并清洗数据
        monthly_series = clean_numeric(X[self.monthly])
        total_series = clean_numeric(X[self.total])
        tenure_series = clean_numeric(X[self.tenure])

        # 计算 avg = total / tenure。
        # tenure 为 0 时结果为 inf，后续统一处理
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_series = total_series / tenure_series
            # 计算 ratio = avg / monthly
            ratio = avg_series / monthly_series

        # 处理 inf, -inf 和 NaN，保证鲁棒性
        # 如果无法计算（例如 tenure 或 monthly 均为 0），默认填充 1.0 代表比率持平
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        ratio = ratio.fillna(1.0)
        
        # 将结果赋值回数据框
        X["charge_ratio"] = ratio
        
        return X
