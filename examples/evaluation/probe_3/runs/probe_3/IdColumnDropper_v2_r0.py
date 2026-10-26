import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class IdColumnDropper(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="IdColumnDropper",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="自动识别并剔除高基数的标识列（ID），防止后续特征工程（如 One-Hot）导致维度爆炸。",
        conditions=["适用于存在唯一标识符的结构化数据"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"id_columns": ["customer_id"]},
        parent="IdColumnDropper@1"
    )

    def __init__(self, id_columns=None):
        self.id_columns = id_columns if id_columns is not None else ["customer_id"]

    def fit(self, X, y=None):
        # 记录训练集中存在的 ID 列，保证后续转换的一致性
        self.id_columns_ = [c for c in self.id_columns if c in X.columns]
        return self

    def transform(self, X):
        X = X.copy()
        
        # 过滤掉不存在的列名，防止报错
        cols_to_drop = [c for c in self.id_columns_ if c in X.columns]
        
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
            
        # 确保输出不会包含任何无穷值或异常处理逻辑引起的非数值问题
        # 注意：此算子仅负责删除，后续数值处理应由对应的数值处理器完成
        return X
