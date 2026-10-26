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
        description="删除指定的ID列或包含ID特征的列，以防止模型在ID上过拟合。",
        conditions=["数据集中存在标识性列（如ticket_id）"],
        dependencies=[],
        params={"columns": None}
    )

    def __init__(self, columns=None):
        """
        初始化 IdColumnDropper。
        
        Args:
            columns (list of str, optional): 需要删除的列名列表。
        """
        self.columns = columns

    def fit(self, X, y=None):
        """
        保存状态。
        """
        self.columns_ = self.columns
        return self

    def transform(self, X):
        """
        删除指定的列，返回剔除后的数据集副本。
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
            
        X = X.copy()
        
        # 如果没有明确指定列，尝试自动识别常见的ID列名，或者如果已指定则执行删除
        if self.columns_ is not None:
            cols_to_drop = [c for c in self.columns_ if c in X.columns]
            X = X.drop(columns=cols_to_drop)
        else:
            # 默认逻辑：识别名称中含有 'id' 且不区分大小写的列
            id_cols = [c for c in X.columns if 'id' in str(c).lower()]
            X = X.drop(columns=id_cols)
            
        return X
