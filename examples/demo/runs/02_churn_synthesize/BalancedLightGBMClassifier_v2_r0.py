import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
import lightgbm as lgb

from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind, Modality


class BalancedLightGBMClassifier(ClassifierMixin, Operator):
    meta = OperatorMeta(
        name="BalancedLightGBMClassifier",
        kind=OperatorKind.MODEL,
        version=2,
        origin="synthesized",
        description="使用带类别平衡权重的LightGBM分类器进行建模，优化正类召回率并提供概率预测",
        conditions=["imbalanced", "tabular", "classification"],
        dependencies=["lightgbm", "scikit-learn", "numpy", "pandas"],
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
            "threshold": 0.5,
        },
        parent="BalancedLightGBMClassifier@1",
    )

    def __init__(
        self,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        class_weight: str = "balanced",
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        verbose: int = -1,
        threshold: float = 0.5,
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
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=str)
            X_mat = X.copy()
        else:
            X_mat = pd.DataFrame(X)
            self.feature_names_in_ = np.array(X_mat.columns, dtype=str)

        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)

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

        self.model_.fit(X_mat, y_arr)
        self.feature_importances_ = self.model_.feature_importances_
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_mat = X.copy()
            if hasattr(self, "feature_names_in_"):
                # 保持与训练时特征一致，缺失特征补 NaN
                for col in self.feature_names_in_:
                    if col not in X_mat.columns:
                        X_mat[col] = np.nan
                X_mat = X_mat[list(self.feature_names_in_)]
        else:
            X_mat = pd.DataFrame(X)

        proba = self.model_.predict_proba(X_mat)
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        if len(self.classes_) == 2:
            pos_idx = 1
            preds = np.where(
                proba[:, pos_idx] >= self.threshold,
                self.classes_[1],
                self.classes_[0],
            )
            return preds
        return self.classes_[np.argmax(proba, axis=1)]
