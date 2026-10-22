import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import ClassifierMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class BalancedLightGBMClassifier(ClassifierMixin, Operator):
    meta = OperatorMeta(
        name="BalancedLightGBMClassifier",
        kind=OperatorKind.MODEL,
        version=2,
        origin="synthesized",
        description="训练带有 balanced 类权重的 LightGBM 二分类模型以强化召回率",
        conditions=["imbalanced", "numeric_features", "has_categorical_features"],
        dependencies=["lightgbm"],
        params={
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "class_weight": "balanced",
            "random_state": 42,
            "verbose": -1
        },
        parent="BalancedLightGBMClassifier@1"
    )

    def __init__(self, n_estimators=400, learning_rate=0.05, num_leaves=31, class_weight="balanced", random_state=42, verbose=-1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.class_weight = class_weight
        self.random_state = random_state
        self.verbose = verbose

    def _preprocess(self, X):
        X = X.copy()
        # 将无穷值替换为 NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        # 将 object/string 类型列转换为 category 类型，以便 LightGBM 原生支持
        for col in X.columns:
            if X[col].dtype == 'object' or isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].astype('category')
        return X

    def fit(self, X, y):
        X_clean = self._preprocess(X)
        
        # 确保 y 为 1D 数组
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        
        self.classes_ = np.unique(y)
        self.feature_names_in_ = np.array(X_clean.columns)
        self.categorical_cols_ = list(X_clean.select_dtypes(include=['category']).columns)
        
        self.model_ = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            class_weight=self.class_weight,
            random_state=self.random_state,
            verbose=self.verbose
        )
        
        self.model_.fit(
            X_clean, 
            y, 
            categorical_feature=self.categorical_cols_ if self.categorical_cols_ else 'auto'
        )
        return self

    def _align_features(self, X):
        X_clean = self._preprocess(X)
        if hasattr(self, "feature_names_in_"):
            # 对齐训练时的列，缺失列自动填充 NaN，多余列自动丢弃
            X_clean = X_clean.reindex(columns=self.feature_names_in_)
            # 确保类别型列在对齐后依然保持 category 类型
            for col in self.categorical_cols_:
                X_clean[col] = X_clean[col].astype('category')
        return X_clean

    def predict_proba(self, X):
        X_aligned = self._align_features(X)
        return self.model_.predict_proba(X_aligned)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
