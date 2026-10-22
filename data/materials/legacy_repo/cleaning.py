import pandas as pd


def clean_total_charges(df, col="total_charges"):
    """把累计消费列从字符串转为数值。

    新入网客户的 total_charges 为空白字符串，直接 astype(float) 会报
    ValueError: could not convert string to float。这里先 strip 再
    to_numeric(errors="coerce")，空值用 0 填充（新客户尚未产生消费）。
    """
    out = df.copy()
    out[col] = pd.to_numeric(out[col].astype(str).str.strip(), errors="coerce").fillna(0.0)
    return out


def drop_id_columns(df, id_cols=("customer_id",)):
    """删除客户 ID 等高基数标识列，避免被当作类别特征做 one-hot 导致维度爆炸。"""
    return df.drop(columns=[c for c in id_cols if c in df.columns])
