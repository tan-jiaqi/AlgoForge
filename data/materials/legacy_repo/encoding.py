import pandas as pd
from sklearn.model_selection import KFold


def target_encode(train, test, col, target="churn", n_splits=5, smoothing=10):
    """对高基数类别列做 K 折目标编码（带平滑）。

    注意：必须只在训练集内做折外编码再映射到测试集，否则会标签泄漏。
    2024-03 的实验里曾在全量数据上直接编码，AUC 虚高到 0.97，上线后崩到 0.71。
    """
    prior = train[target].mean()
    train_enc = pd.Series(index=train.index, dtype=float)
    for fit_idx, val_idx in KFold(n_splits, shuffle=True, random_state=0).split(train):
        stats = train.iloc[fit_idx].groupby(col)[target].agg(["mean", "count"])
        smooth = (stats["mean"] * stats["count"] + prior * smoothing) / (stats["count"] + smoothing)
        train_enc.iloc[val_idx] = train.iloc[val_idx][col].map(smooth).fillna(prior).values
    stats = train.groupby(col)[target].agg(["mean", "count"])
    smooth = (stats["mean"] * stats["count"] + prior * smoothing) / (stats["count"] + smoothing)
    return train_enc, test[col].map(smooth).fillna(prior)
