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
        description="使用LightGBM训练分类模型，自动平衡正负样本权重以应对不平衡数据",
        conditions=["imbalanced", "numeric_features", "has_categorical_features"],
        dependencies=["lightgbm", "scikit-learn", "pandas", "numpy"],
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

    def __init__(
        self, 
        n_estimators=400, 
        learning_rate=0.05, 
        num_leaves=31, 
        class_weight="balanced", 
        random_state=42, 
        verbose=-1
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.class_weight = class_weight
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        # 识别类别型特征
        self.categorical_cols_ = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        self.categories_ = {}
        
        X_fit = X.copy()
        for col in self.categorical_cols_:
            X_fit[col] = X_fit[col].astype('category')
            self.categories_[col] = X_fit[col].cat.categories

        self.model_ = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            class_weight=self.class_weight,
            random_state=self.random_state,
            verbose=self.verbose
        )
        
        self.model_.fit(X_fit, y)
        self.classes_ = self.model_.classes_
        return self

    def _preprocess(self, X):
        X_pred = X.copy()
        for col in self.categorical_cols_:
            if col in X_pred.columns:
                # 使用训练集中的类别对齐，未见类别和缺失值会自动转化为 NaN，LightGBM 可原生处理
                X_pred[col] = pd.Categorical(X_pred[col], categories=self.categories_[col])
        return X_pred

    def predict(self, X):
        X_pred = self._preprocess(X)
        return self.model_.predict(X_pred)

    def predict_proba(self, X):
        X_pred = self._preprocess(X)
        return self.model_.predict_proba(X_pred)

    def transform(self, X):
        X_copy = X.copy()
        preds = self.predict(X_copy)
        return pd.DataFrame({"prediction": preds}, index=X_copy.index)
