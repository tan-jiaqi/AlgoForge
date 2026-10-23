import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import ClassifierMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind, Modality


class BalancedLightGBMClassifier(ClassifierMixin, Operator):
    meta = OperatorMeta(
        name="BalancedLightGBMClassifier",
        kind=OperatorKind.MODEL,
        version=2,
        origin="synthesized",
        description="基于LightGBM的平衡分类器，支持正负样本权重自动平衡与特征重要性输出",
        conditions=["imbalanced", "classification", "tabular"],
        dependencies=["lightgbm", "scikit-learn"],
        params={
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "class_weight": "balanced",
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1,
        },
        parent="BalancedLightGBMClassifier@1",
    )

    def __init__(
        self,
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        class_weight="balanced",
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        X_df = pd.DataFrame(X).copy()
        y_arr = np.asarray(y)

        self.classes_ = np.unique(y_arr)
        self.feature_names_in_ = np.array(X_df.columns, dtype=str)
        self.n_features_in_ = X_df.shape[1]

        # 确保列名无特殊 JSON 字符干扰 LightGBM
        clean_cols = [str(c).replace(":", "_").replace(" ", "_") for c in X_df.columns]
        X_df.columns = clean_cols
        self.cleaned_feature_names_ = clean_cols

        self.model_ = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        self.model_.fit(X_df, y_arr)
        self.feature_importances_ = self.model_.feature_importances_
        return self

    def _prepare_input(self, X):
        X_df = pd.DataFrame(X).copy()
        if hasattr(self, "cleaned_feature_names_"):
            X_df.columns = self.cleaned_feature_names_
        return X_df

    def predict_proba(self, X):
        X_df = self._prepare_input(X)
        proba = self.model_.predict_proba(X_df)
        return np.asarray(proba, dtype=np.float64)

    def predict(self, X):
        X_df = self._prepare_input(X)
        pred = self.model_.predict(X_df)
        return np.asarray(pred)
