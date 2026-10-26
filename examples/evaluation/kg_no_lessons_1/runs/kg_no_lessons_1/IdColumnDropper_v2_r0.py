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
        description="删除指定的客户ID或高基数标识列，防止模型过拟合噪声及特征空间膨胀。",
        conditions=["数据含客户 ID 等标识列", "数据含高基数唯一标识"],
        dependencies=[],
        params={"id_cols": ["customer_id"]},
        parent="IdColumnDropper@1"
    )

    def __init__(self, id_cols=["customer_id"]):
        self.id_cols = id_cols

    def fit(self, X, y=None):
        # 记录 fit 状态，本算子为无状态转换，但符合 sklearn 规范
        self.id_cols_ = list(self.id_cols)
        return self

    def transform(self, X):
        # 复制数据，保证输入不变性
        X = X.copy()
        
        # 识别出存在于列中的目标列
        cols_to_drop = [c for c in self.id_cols_ if c in X.columns]
        
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
            
        return X
