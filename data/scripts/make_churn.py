"""生成模拟的电信客户流失数据集（参照 IBM Telco Churn 字段，固定随机种子可复现）。

故意保留了真实数据中的"脏点"，供后续验证与修复环节使用：
- total_charges 为字符串列，新客户（tenure=0）为空白字符串 " "；
- monthly_charges / payment_method 有少量缺失；
- customer_id 为高基数 ID 列，不应参与建模。

用法：python data/scripts/make_churn.py [--rows 5000] [--out data/churn.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def make_churn(rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["月付", "一年", "两年"], rows, p=[0.55, 0.25, 0.20])
    # 合约越长，在网时长越长
    tenure_scale = np.select([contract == "月付", contract == "一年"], [12, 30], 50)
    tenure = np.clip(rng.exponential(tenure_scale), 1, 72).astype(int)
    tenure[rng.random(rows) < 0.015] = 0  # 新入网客户

    internet = rng.choice(["光纤", "DSL", "无"], rows, p=[0.45, 0.35, 0.20])
    has_net = internet != "无"
    online_security = np.where(has_net, rng.choice(["是", "否"], rows, p=[0.35, 0.65]), "无网络服务")
    tech_support = np.where(has_net, rng.choice(["是", "否"], rows, p=[0.35, 0.65]), "无网络服务")
    streaming_tv = np.where(has_net, rng.choice(["是", "否"], rows, p=[0.45, 0.55]), "无网络服务")
    payment = rng.choice(["电子支票", "邮寄支票", "银行转账", "信用卡"], rows, p=[0.35, 0.2, 0.22, 0.23])
    paperless = rng.choice(["是", "否"], rows, p=[0.6, 0.4])
    senior = (rng.random(rows) < 0.16).astype(int)
    partner = rng.choice(["是", "否"], rows)
    region = rng.choice(["华东", "华南", "华北", "西南", "西北", "东北"], rows)

    base = np.select([internet == "光纤", internet == "DSL"], [75, 50], 22)
    monthly = base + 8 * (streaming_tv == "是") + 5 * (online_security == "是") + rng.normal(0, 6, rows)
    monthly = np.round(np.clip(monthly, 18, 120), 2)
    total = np.round(monthly * tenure * rng.uniform(0.95, 1.05, rows), 2)

    # 流失的潜在逻辑：月付、光纤、电子支票、无安全/技术支持、在网短 → 风险高
    logit = (
        -2.3
        + 1.3 * (contract == "月付")
        - 0.9 * (contract == "两年")
        + 0.7 * (internet == "光纤")
        + 0.6 * (payment == "电子支票")
        + 0.4 * (online_security == "否")
        + 0.4 * (tech_support == "否")
        + 0.3 * senior
        + 0.2 * (paperless == "是")
        - 0.045 * tenure
        + 0.012 * (monthly - 65)
        + rng.normal(0, 0.6, rows)
    )
    churn = (rng.random(rows) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{100000 + i}" for i in range(rows)],
            "region": region,
            "senior_citizen": senior,
            "partner": partner,
            "tenure": tenure,
            "contract": contract,
            "internet_service": internet,
            "online_security": online_security,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "paperless_billing": paperless,
            "payment_method": payment,
            "monthly_charges": monthly,
            "total_charges": [" " if t == 0 else f"{v:.2f}" for t, v in zip(tenure, total)],
            "churn": churn,
        }
    )
    df.loc[rng.random(rows) < 0.03, "monthly_charges"] = np.nan
    df.loc[rng.random(rows) < 0.02, "payment_method"] = np.nan
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "churn.csv")
    args = parser.parse_args()
    df = make_churn(args.rows, args.seed)
    df.to_csv(args.out, index=False)
    print(f"写入 {args.out}：{len(df)} 行，流失率 {df['churn'].mean():.1%}")


if __name__ == "__main__":
    main()
