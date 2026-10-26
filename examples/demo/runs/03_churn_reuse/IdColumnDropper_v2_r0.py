import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class IdColumnDropper(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="IdColumnDropper",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="自动识别并剔除数据中的标识类(ID)特征，防止后续编码导致维度灾难。",
        conditions=["数据包含标识列"],
        dependencies=[],
        params={"id_columns": None}
    )

    def __init__(self, id_columns=None):
        self.id_columns = id_columns

    def fit(self, X, y=None):
        # 如果未指定id_columns，通过启发式规则自动识别：
        # 若某列唯一值数量等于行数，或列名包含id且唯一值极高，则视为ID列
        if self.id_columns is None:
            n_rows = len(X)
            self.id_columns_ = [
                col for col in X.columns 
                if X[col].nunique() == n_rows or ('id' in col.lower() and X[col].nunique() > n_rows * 0.9)
            ]
        else:
            self.id_columns_ = self.id_columns
        return self

    def transform(self, X):
        X = X.copy()
        # 移除标识列，过滤掉不存在于X中的列名以防报错
        cols_to_drop = [c for c in self.id_columns_ if c in X.columns]
        X = X.drop(columns=cols_to_drop)
        
        # 确保算子输出无无穷值(虽通常不产生，但按规范清理)
        # 注意：此处仅处理数值列，类别列不受影响
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        return X
