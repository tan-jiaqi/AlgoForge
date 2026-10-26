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
        description="统计客户订购的增值服务数量，捕捉客户对平台的粘性特征",
        conditions=["数据包含增值服务相关类别列"],
        dependencies=[],
        params={"service_cols": ("online_security", "tech_support", "streaming_tv")},
        parent="ServiceCountFeature@1"
    )

    def __init__(self, service_cols=("online_security", "tech_support", "streaming_tv")):
        self.service_cols = service_cols

    def fit(self, X, y=None):
        # 无需训练状态，记录列信息即可
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        
        # 统计服务计数
        # 逻辑：对于指定的列，若值为"Yes"或"是"则记为1，处理缺失值或未见类别不会报错
        n_services = pd.Series(0, index=X.index)
        for col in self.service_cols:
            if col in X.columns:
                # 统一转字符串并剔除空格，兼容不同语言编码
                col_vals = X[col].astype(str).str.strip().str.lower()
                n_services += (col_vals.isin(["yes", "是"])).astype(int)
        
        X["n_services"] = n_services
        
        # 处理新特征可能产生的缺失/无穷值（本逻辑下不会产生，但遵守规范）
        X["n_services"] = X["n_services"].fillna(0).replace([np.inf, -np.inf], 0)
        
        return X
