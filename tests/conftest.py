import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tabular_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "tenure": rng.integers(1, 72, n).astype(float),
            "monthly_charges": rng.normal(70, 20, n),
            "contract": rng.choice(["month", "one_year", "two_year"], n),
            "payment": rng.choice(["card", "bank", "check"], n),
        }
    )
    df.loc[::10, "monthly_charges"] = np.nan
    df.loc[::15, "contract"] = None
    df["churn"] = ((df["tenure"] < 20) & (df["contract"] == "month")).astype(int)
    return df


@pytest.fixture
def text_df() -> pd.DataFrame:
    texts = ["网络断了无法上网", "宽带很慢经常掉线", "话费扣错了要退款", "账单金额不对", "想办理套餐升级", "咨询新套餐资费"] * 10
    labels = ["network", "network", "billing", "billing", "plan", "plan"] * 10
    return pd.DataFrame({"text": texts, "label": labels})
