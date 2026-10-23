import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind


class ChargeRatioFeatureGenerator(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="ChargeRatioFeatureGenerator",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="计算历史平均月度费用与当前资费的比率特征，刻画客户近期资费变动敏感度",
        conditions=["numeric_features", "time_tenure", "string_typed_numeric"],
        dependencies=["numpy", "pandas", "scikit-learn"],
        params={
            "monthly_col": "monthly_charges",
            "total_col": "total_charges",
            "tenure_col": "tenure",
            "output_col": "charge_ratio",
            "fill_value": 1.0,
        },
        parent="ChargeRatioFeatureGenerator@1",
    )

    def __init__(
        self,
        monthly_col: str = "monthly_charges",
        total_col: str = "total_charges",
        tenure_col: str = "tenure",
        output_col: str = "charge_ratio",
        fill_value: float = 1.0,
    ):
        self.monthly_col = monthly_col
        self.total_col = total_col
        self.tenure_col = tenure_col
        self.output_col = output_col
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = list(X.columns)
        return self

    def _clean_numeric(self, series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce")
        return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        required_cols = [self.monthly_col, self.total_col, self.tenure_col]
        missing_cols = [col for col in required_cols if col not in X.columns]

        if missing_cols:
            X[self.output_col] = float(self.fill_value)
            return X

        monthly = self._clean_numeric(X[self.monthly_col])
        total = self._clean_numeric(X[self.total_col])
        tenure = self._clean_numeric(X[self.tenure_col])

        safe_tenure = tenure.replace(0, np.nan)
        avg_charge = total / safe_tenure

        safe_monthly = monthly.replace(0, np.nan)
        ratio = avg_charge / safe_monthly

        X[self.output_col] = (
            ratio.replace([np.inf, -np.inf], np.nan).fillna(self.fill_value).astype(float)
        )
        return X
