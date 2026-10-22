from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind, Modality
from sklearn.base import ClassifierMixin
import numpy as np
import pandas as pd
import lightgbm as lgb


class BalancedLightGBMClassifier(ClassifierMixin, Operator):
    meta = OperatorMeta(
        name="BalancedLightGBMClassifier",
        kind=OperatorKind.MODEL,
        version=2,
        origin="synthesized",
        description="基于LightGBM的平衡二分类模型，通过样本类别加权提升不平衡数据下的召回率并提供特征重要性",
        conditions=["imbalanced", "classification", "tabular"],
        dependencies=["lightgbm", "numpy", "pandas", "scikit-learn"],
        params={
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        },
        parent="BalancedLightGBMClassifier@1",
    )

    def __init__(
        self,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        class_weight: str = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = -1,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _prepare_data(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col in X_out.columns:
            if X_out[col].dtype == "object":
                X_out[col] = X_out[col].astype("category")
        return X_out

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_proc = self._prepare_data(X)
        y_proc = np.asarray(y)

        self.feature_names_in_ = np.array(X_proc.columns, dtype=str)
        self.n_features_in_ = X_proc.shape[1]

        self.model_ = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        self.model_.fit(X_proc, y_proc)

        self.classes_ = np.asarray(self.model_.classes_)
        self.feature_importances_ = np.asarray(self.model_.feature_importances_)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_proc = self._prepare_data(X)
        proba = self.model_.predict_proba(X_proc)
        return np.asarray(proba, dtype=np.float64)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_proc = self._prepare_data(X)
        preds = self.model_.predict(X_proc)
        return np.asarray(preds)
