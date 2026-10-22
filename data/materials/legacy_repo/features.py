import numpy as np
import pandas as pd


def add_tenure_bucket(df, col="tenure"):
    """按在网时长分段：0-6、6-12、12-24、24-48、48+ 个月。

    业务经验：前 6 个月是流失高发期，分段后对线性模型更友好，也便于向业务方解释。
    """
    out = df.copy()
    bins = [-1, 6, 12, 24, 48, np.inf]
    labels = ["0-6", "6-12", "12-24", "24-48", "48+"]
    out[col + "_bucket"] = pd.cut(out[col], bins=bins, labels=labels).astype(str)
    return out


def add_charge_ratio(df, monthly="monthly_charges", total="total_charges", tenure="tenure"):
    """构造消费相关比值特征：月均消费 / 当前月费，衡量近期资费是否上涨。

    要求 total_charges 已为数值；tenure 为 0 时比值记为 1。
    """
    out = df.copy()
    avg = out[total] / out[tenure].replace(0, np.nan)
    out["charge_ratio"] = (avg / out[monthly]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return out


def count_services(df, service_cols=("online_security", "tech_support", "streaming_tv")):
    """统计客户订购的增值服务数量；订购越多，迁移成本越高，流失率越低。"""
    out = df.copy()
    out["n_services"] = sum((out[c] == "是").astype(int) for c in service_cols if c in out.columns)
    return out
