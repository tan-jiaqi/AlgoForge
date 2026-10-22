"""数据画像（Data Profile）：确定性地分析数据集，产出列角色和适用条件标签。

画像结果有两个用途：
1. 作为 KG 检索的 ``conditions``，例如 ``has_missing_values``、``imbalanced``；
2. 作为 Planner prompt 中的数据摘要，让 LLM 基于事实规划，而不是靠猜。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from factory.kg import CONDITION_TAGS

ID_HINTS = ("id", "编号", "uuid")
TIME_HINTS = ("tenure", "days", "months", "duration", "时长")
TARGET_HINTS = ("churn", "label", "target", "y", "class", "流失")


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    role: str  # numeric / categorical / string_numeric / text / id
    missing_rate: float
    n_unique: int


@dataclass
class DataProfile:
    path: str
    n_rows: int
    target: str | None
    class_counts: dict[str, int]
    columns: list[ColumnProfile]
    tags: list[str] = field(default_factory=list)

    @property
    def modality(self) -> str:
        return "text" if "text_input" in self.tags else "tabular"

    @property
    def text_column(self) -> str | None:
        return next((c.name for c in self.columns if c.role == "text"), None)

    def summary(self) -> dict:
        """供 prompt 使用的精简摘要。"""
        return {
            "rows": self.n_rows,
            "target": self.target,
            "class_counts": self.class_counts,
            "tags": {t: CONDITION_TAGS.get(t, t) for t in self.tags},
            "columns": [
                f"{c.name}: {c.role}, {c.dtype}, missing {c.missing_rate:.1%}, unique {c.n_unique}" for c in self.columns
            ],
        }


def load_table(path: str | Path) -> pd.DataFrame:
    """统一的读数入口：父进程画像与沙箱子进程必须读得一模一样。"""
    return pd.read_csv(path)


def _missing_rate(s: pd.Series) -> float:
    blank = s.astype(str).str.strip().eq("") if s.dtype == object else False
    return float((s.isna() | blank).mean())


def _role(name: str, s: pd.Series, n_rows: int) -> str:
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    nonblank = s.dropna().astype(str).str.strip()
    nonblank = nonblank[nonblank != ""]
    if len(nonblank) and pd.to_numeric(nonblank, errors="coerce").notna().mean() > 0.9:
        return "string_numeric"
    uniq = s.nunique(dropna=True) / max(n_rows, 1)
    if uniq > 0.95 and any(h in name.lower() for h in ID_HINTS):
        return "id"
    if uniq > 0.3 and nonblank.str.len().mean() > 8:
        return "text"
    if uniq > 0.95:
        return "id"
    return "categorical"


def guess_target(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if col.lower() in TARGET_HINTS:
            return col
    return None


def profile(path: str | Path, target: str | None = None) -> DataProfile:
    df = load_table(path)
    target = target if target in df.columns else guess_target(df)
    n = len(df)
    cols = [
        ColumnProfile(c, str(df[c].dtype), _role(c, df[c], n), _missing_rate(df[c]), int(df[c].nunique(dropna=True)))
        for c in df.columns
        if c != target
    ]
    counts = {str(k): int(v) for k, v in df[target].value_counts().items()} if target else {}

    roles = {c.role for c in cols}
    cats = [c for c in cols if c.role == "categorical"]
    tags = []
    if any(c.missing_rate > 0 for c in cols):
        tags.append("has_missing_values")
    if "numeric" in roles or "string_numeric" in roles:
        tags.append("numeric_features")
    if cats:
        tags.append("has_categorical_features")
        tags.append("high_cardinality" if any(c.n_unique > 50 for c in cats) else "low_cardinality")
    if "string_numeric" in roles:
        tags.append("string_typed_numeric")
    if "id" in roles:
        tags.append("id_columns")
    if "text" in roles:
        tags.append("text_input")
    if counts and min(counts.values()) / n < 0.35:
        tags.append("imbalanced")
    if any(h in c.name.lower() for c in cols for h in TIME_HINTS):
        tags.append("time_tenure")
    if cats and "numeric" in roles:
        tags.append("mixed_features")
    if n > 50_000:
        tags.append("large_tabular")
    return DataProfile(str(path), n, target, counts, cols, tags)
