import re
from typing import List, Optional
import pandas as pd
from sklearn.base import TransformerMixin

from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind


class IdColumnDropper(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="IdColumnDropper",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="剔除客户ID等标识列以防止特征维度暴涨与过拟合",
        conditions=["id_columns"],
        dependencies=[],
        params={
            "columns": None,
            "auto_detect": True,
            "pattern": r"(?i)(?:^|_)(?:id|uuid|guid|code|no)(?:$|_)|^customer_id$|^user_id$",
            "high_cardinality_threshold": 0.98,
        },
    )

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        auto_detect: bool = True,
        pattern: str = r"(?i)(?:^|_)(?:id|uuid|guid|code|no)(?:$|_)|^customer_id$|^user_id$",
        high_cardinality_threshold: float = 0.98,
    ):
        self.columns = columns
        self.auto_detect = auto_detect
        self.pattern = pattern
        self.high_cardinality_threshold = high_cardinality_threshold

    def fit(self, X: pd.DataFrame, y=None):
        cols_to_drop = set()

        if self.columns:
            for col in self.columns:
                if col in X.columns:
                    cols_to_drop.add(col)

        if self.auto_detect and len(X) > 0:
            regex = re.compile(self.pattern)
            for col in X.columns:
                col_str = str(col)
                # 检查列名是否匹配 ID 命名模式
                is_name_match = bool(regex.search(col_str))

                # 计算非空值的唯一值比例
                non_null_series = X[col].dropna()
                n_count = len(non_null_series)
                ratio = (non_null_series.nunique() / n_count) if n_count > 0 else 0.0

                # 若命中了名称规则，或者属于高基数字符串/对象标识列且唯一率极高
                if is_name_match:
                    cols_to_drop.add(col)
                elif pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_string_dtype(X[col]):
                    if ratio >= self.high_cardinality_threshold and n_count > 30:
                        # 辅以简单判断，避免误删长文本描述（均长较短更可能是标识符）
                        avg_len = non_null_series.astype(str).str.len().mean()
                        if avg_len <= 64:
                            cols_to_drop.add(col)

        self.dropped_columns_ = sorted(list(cols_to_drop))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if hasattr(self, "dropped_columns_") and self.dropped_columns_:
            existing_cols = [c for c in self.dropped_columns_ if c in X.columns]
            if existing_cols:
                X = X.drop(columns=existing_cols)
        return X
