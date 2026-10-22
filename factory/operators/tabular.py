"""内置表格预处理算子（输入、输出均为 DataFrame）。"""

from __future__ import annotations

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from factory.operators.base import Operator, OperatorMeta, register, split_columns
from factory.schemas import OperatorKind


@register
class MissingValueImputer(TransformerMixin, Operator):
    """数值列缺失用中位数填补，类别列缺失用众数填补。"""

    meta = OperatorMeta(
        name="MissingValueImputer",
        kind=OperatorKind.IMPUTER,
        description="数值列用中位数、类别列用众数填补缺失值。",
        conditions=["has_missing_values"],
        dependencies=["pandas"],
    )

    def fit(self, X: pd.DataFrame, y=None):
        numeric, categorical = split_columns(X)
        self.fill_values_ = {c: X[c].median() for c in numeric}
        for c in categorical:
            mode = X[c].mode(dropna=True)
            self.fill_values_[c] = mode.iloc[0] if not mode.empty else "missing"
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.fillna(value=self.fill_values_)


@register
class OneHotCategoricalEncoder(TransformerMixin, Operator):
    """对非数值列做 one-hot 编码，数值列原样保留。"""

    meta = OperatorMeta(
        name="OneHotCategoricalEncoder",
        kind=OperatorKind.ENCODER,
        description="类别列 one-hot 编码，忽略未见类别。",
        conditions=["has_categorical_features", "low_cardinality"],
        dependencies=["scikit-learn"],
        params={"max_categories": 20},
    )

    def __init__(self, max_categories: int = 20):
        self.max_categories = max_categories

    def fit(self, X: pd.DataFrame, y=None):
        self.numeric_, self.categorical_ = split_columns(X)
        self.encoder_ = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            max_categories=self.max_categories,
            sparse_output=False,
        )
        if self.categorical_:
            self.encoder_.fit(X[self.categorical_].astype(str))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric = X[self.numeric_]
        if not self.categorical_:
            return numeric.copy()
        encoded = pd.DataFrame(
            self.encoder_.transform(X[self.categorical_].astype(str)),
            columns=self.encoder_.get_feature_names_out(self.categorical_),
            index=X.index,
        )
        return pd.concat([numeric, encoded], axis=1)


@register
class StandardFeatureScaler(TransformerMixin, Operator):
    """把数值列标准化为零均值、单位方差。"""

    meta = OperatorMeta(
        name="StandardFeatureScaler",
        kind=OperatorKind.SCALER,
        description="数值列 Z-score 标准化，线性模型和基于距离的模型需要。",
        conditions=["linear_model", "numeric_features"],
        dependencies=["scikit-learn"],
    )

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_, _ = split_columns(X)
        self.scaler_ = StandardScaler().fit(X[self.columns_]) if self.columns_ else None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if self.scaler_ is not None:
            out[self.columns_] = self.scaler_.transform(X[self.columns_])
        return out
