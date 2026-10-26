"""内置模型算子（流水线的最后一步）。"""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from factory.operators.base import Operator, OperatorMeta, register
from factory.schemas import Modality, OperatorKind

BOTH = [Modality.TABULAR, Modality.TEXT]


def _align_class_keys(weight: dict, y) -> dict:
    """方案参数经过 JSON 序列化，{0: 1, 1: 3} 会变成 {"0": 1, "1": 3}；按字符串形式对回 y 中的真实类别。"""
    by_str = {str(c): c for c in np.unique(y)}
    return {by_str.get(str(k), k): v for k, v in weight.items()}


class _WrappedClassifier(ClassifierMixin, Operator):
    """公共封装：子类实现 ``_build``，返回一个 sklearn 分类器。"""

    def _build(self):
        raise NotImplementedError

    def fit(self, X, y):
        model = self._build()
        weight = getattr(model, "class_weight", None)
        if isinstance(weight, dict):
            model.set_params(class_weight=_align_class_keys(weight, y))
        self.model_ = model.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X) -> np.ndarray:
        return self.model_.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model_.predict_proba(X)


@register
class LogisticRegressionModel(_WrappedClassifier):
    meta = OperatorMeta(
        name="LogisticRegressionModel",
        kind=OperatorKind.MODEL,
        modalities=BOTH,
        description="L2 逻辑回归：系数可解释，也是稀疏文本特征上的强基线。",
        outputs="labels / class probabilities",
        conditions=["interpretable", "scaled_features", "sparse_features"],
        dependencies=["scikit-learn"],
        params={"C": 1.0, "class_weight": "balanced", "max_iter": 1000, "random_state": 42},
    )

    def __init__(self, C: float = 1.0, class_weight="balanced", max_iter: int = 1000, random_state: int = 42):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state

    def _build(self):
        return LogisticRegression(
            C=self.C, class_weight=self.class_weight, max_iter=self.max_iter, random_state=self.random_state
        )


@register
class RandomForestModel(_WrappedClassifier):
    meta = OperatorMeta(
        name="RandomForestModel",
        kind=OperatorKind.MODEL,
        description="随机森林：不受特征尺度影响，能拟合非线性，可输出特征重要性。",
        outputs="labels / class probabilities",
        conditions=["nonlinear", "mixed_features", "feature_importance"],
        dependencies=["scikit-learn"],
        params={"n_estimators": 300, "max_depth": None, "class_weight": "balanced", "random_state": 42},
    )

    def __init__(self, n_estimators: int = 300, max_depth=None, class_weight="balanced", random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state

    def _build(self):
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )


@register
class LightGBMModel(_WrappedClassifier):
    meta = OperatorMeta(
        name="LightGBMModel",
        kind=OperatorKind.MODEL,
        description="梯度提升树（LightGBM）：通常是表格数据上最强的选择。",
        outputs="labels / class probabilities",
        conditions=["nonlinear", "large_tabular", "imbalanced"],
        dependencies=["lightgbm"],
        params={
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "class_weight": "balanced",
            "random_state": 42,
        },
    )

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        class_weight="balanced",
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.class_weight = class_weight
        self.random_state = random_state

    def _build(self):
        return LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            class_weight=self.class_weight,
            random_state=self.random_state,
            verbose=-1,
        )
