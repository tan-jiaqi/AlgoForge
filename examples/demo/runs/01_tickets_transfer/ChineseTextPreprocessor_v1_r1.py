import re
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind, Modality


class ChineseTextPreprocessor(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="ChineseTextPreprocessor",
        kind=OperatorKind.FEATURE,
        version=1,
        origin="synthesized",
        description="对中文文本进行标点过滤、字符规整及空格分字预处理以提升TF-IDF特征质量",
        conditions=["text_input"],
        dependencies=[],
        params={
            "text_columns": None,
            "remove_punctuation": True,
            "remove_digits": False,
            "to_lower": True,
            "space_between_chars": True,
        },
    )

    def __init__(
        self,
        text_columns=None,
        remove_punctuation=True,
        remove_digits=False,
        to_lower=True,
        space_between_chars=True,
    ):
        self.text_columns = text_columns
        self.remove_punctuation = remove_punctuation
        self.remove_digits = remove_digits
        self.to_lower = to_lower
        self.space_between_chars = space_between_chars

    def fit(self, X, y=None):
        if self.text_columns is not None:
            self.text_columns_ = [col for col in self.text_columns if col in X.columns]
        else:
            self.text_columns_ = [
                col for col in X.columns if X[col].dtype == "object" or pd.api.types.is_string_dtype(X[col])
            ]
        return self

    def _clean_text(self, text):
        if pd.isna(text):
            return ""
        s = str(text)
        if self.to_lower:
            s = s.lower()
        if self.remove_punctuation:
            s = re.sub(r"[^\w\s\u4e00-\u9fa5]", " ", s)
        if self.remove_digits:
            s = re.sub(r"\d+", " ", s)
        if self.space_between_chars:
            s = re.sub(r"([\u4e00-\u9fa5])", r" \1 ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def transform(self, X):
        X = X.copy()
        try:
            cols = self.text_columns_
        except AttributeError:
            if self.text_columns is not None:
                cols = [col for col in self.text_columns if col in X.columns]
            else:
                cols = [
                    col for col in X.columns if X[col].dtype == "object" or pd.api.types.is_string_dtype(X[col])
                ]

        for col in cols:
            if col in X.columns:
                X[col] = X[col].apply(self._clean_text)

        return X
