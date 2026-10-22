import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind, Modality


class ServiceCountFeature(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="ServiceCountFeature",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="统计客户订购的增值服务总数，反映客户业务粘性",
        conditions=["has_categorical_features"],
        dependencies=[],
        params={
            "service_cols": ("online_security", "tech_support", "streaming_tv"),
            "positive_values": ("Yes", "是", "1"),
            "output_col": "n_services",
        },
        parent="ServiceCountFeature@1",
    )

    def __init__(
        self,
        service_cols=("online_security", "tech_support", "streaming_tv"),
        positive_values=("Yes", "是", "1"),
        output_col="n_services",
    ):
        self.service_cols = service_cols
        self.positive_values = positive_values
        self.output_col = output_col

    def fit(self, X, y=None):
        cols = list(self.service_cols) if self.service_cols is not None else []
        self.available_cols_ = [c for c in cols if c in X.columns]
        return self

    def transform(self, X):
        X = X.copy()
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        pos_set = set(self.positive_values) if self.positive_values is not None else {"Yes", "是", "1"}
        pos_str_set = {str(v).strip().lower() for v in pos_set}

        count_series = pd.Series(0, index=X.index, dtype=np.int64)

        for col in self.available_cols_:
            if col in X.columns:
                series = X[col]
                is_pos = series.isin(pos_set) | series.astype(str).str.strip().str.lower().isin(pos_str_set)
                count_series = count_series + is_pos.fillna(False).astype(np.int64)

        X[self.output_col] = count_series
        return X
