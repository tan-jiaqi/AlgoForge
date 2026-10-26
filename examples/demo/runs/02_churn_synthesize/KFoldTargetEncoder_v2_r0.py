import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.model_selection import KFold
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class KFoldTargetEncoder(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="KFoldTargetEncoder",
        kind=OperatorKind.ENCODER,
        version=2,
        origin="synthesized",
        description="基于K折交叉验证的高基数类别目标编码，支持平滑处理以防止过拟合。",
        conditions=["数据包含类别特征", "目标变量为二分类"],
        dependencies=["pandas", "sklearn", "numpy"],
        params={"columns": None, "n_splits": 5, "smoothing": 10.0},
        parent="KFoldTargetEncoder@1"
    )

    def __init__(self, columns=None, n_splits=5, smoothing=10.0):
        self.columns = columns
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.mapping_ = {}
        self.prior_ = None

    def fit(self, X, y=None):
        X = X.copy()
        if self.columns is None:
            self.columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        target = y.name if isinstance(y, pd.Series) else "target"
        df = X.copy()
        df[target] = y
        self.prior_ = df[target].mean()

        for col in self.columns:
            self.mapping_[col] = {}
            # 计算全量统计用于测试集映射
            stats = df.groupby(col)[target].agg(["mean", "count"])
            smooth = (stats["mean"] * stats["count"] + self.prior_ * self.smoothing) / (stats["count"] + self.smoothing)
            self.mapping_[col]["global"] = smooth
            
            # 计算训练集K折编码
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            train_enc = pd.Series(index=df.index, dtype=float)
            for fit_idx, val_idx in kf.split(df):
                fold_stats = df.iloc[fit_idx].groupby(col)[target].agg(["mean", "count"])
                fold_smooth = (fold_stats["mean"] * fold_stats["count"] + self.prior_ * self.smoothing) / (fold_stats["count"] + self.smoothing)
                train_enc.iloc[val_idx] = df.iloc[val_idx][col].map(fold_smooth)
            
            self.mapping_[col]["train_enc"] = train_enc
        
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in self.mapping_:
                # 训练集返回K折编码，测试集返回全局平均编码
                if X.index.equals(self.mapping_[col]["train_enc"].index):
                    X[col] = self.mapping_[col]["train_enc"]
                else:
                    X[col] = X[col].map(self.mapping_[col]["global"])
                
                # 处理未见类别和缺失值
                X[col] = X[col].fillna(self.prior_)
                # 确保无无穷值
                X[col] = X[col].replace([np.inf, -np.inf], self.prior_)
        
        return X
