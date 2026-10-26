import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class AddServiceCount(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="AddServiceCount",
        kind=OperatorKind.FEATURE,
        version=1,
        origin="synthesized",
        description="计算客户订购的增值服务总数，捕捉用户粘性特征。",
        conditions=["数据包含增值服务相关类别列"],
        dependencies=["pandas", "numpy"],
        params={"service_cols": ["online_security", "tech_support", "streaming_tv"]},
        parent="ServiceCountFeature@1"
    )

    def __init__(self, service_cols=["online_security", "tech_support", "streaming_tv"]):
        self.service_cols = service_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # 统计增值服务数量
        # 根据业务经验，通常判定逻辑为值等于"Yes"、"是"或"1"
        count = np.zeros(len(X))
        for col in self.service_cols:
            if col in X.columns:
                # 统一转为字符串并剔除空格进行匹配，兼容缺失值
                val = X[col].fillna("no").astype(str).str.strip().str.lower()
                count += (val.isin(["yes", "是", "1"])).astype(int)
        
        X["n_services"] = count
        
        # 处理可能的异常数值（虽然此逻辑不产生NaN/inf，但遵循规范兜底）
        X["n_services"] = X["n_services"].fillna(0).replace([np.inf, -np.inf], 0)
        
        return X
