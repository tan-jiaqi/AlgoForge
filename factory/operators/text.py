"""内置文本算子。"""

from __future__ import annotations

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from factory.operators.base import Operator, OperatorMeta, register
from factory.schemas import Modality, OperatorKind


@register
class TfidfTextVectorizer(TransformerMixin, Operator):
    """对一个文本列提取 TF-IDF 特征（字符级 n-gram 无需分词即可处理中文）。"""

    meta = OperatorMeta(
        name="TfidfTextVectorizer",
        kind=OperatorKind.VECTORIZER,
        modalities=[Modality.TEXT],
        description="文本列 TF-IDF 向量化，默认字符级 n-gram，对中文稳健。",
        outputs="scipy.sparse matrix",
        conditions=["text_input"],
        dependencies=["scikit-learn"],
        params={"column": "text", "analyzer": "char_wb", "ngram_range": (1, 2), "max_features": 20000},
    )

    def __init__(
        self,
        column: str = "text",
        analyzer: str = "char_wb",
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int = 20000,
    ):
        self.column = column
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.max_features = max_features

    def fit(self, X: pd.DataFrame, y=None):
        self.vectorizer_ = TfidfVectorizer(
            analyzer=self.analyzer, ngram_range=tuple(self.ngram_range), max_features=self.max_features
        ).fit(X[self.column].fillna("").astype(str))
        return self

    def transform(self, X: pd.DataFrame):
        return self.vectorizer_.transform(X[self.column].fillna("").astype(str))
