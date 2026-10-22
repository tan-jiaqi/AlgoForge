import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tabular_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 600
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
    signal = (df["tenure"] < 20) & (df["contract"] == "month")
    df["churn"] = np.where(rng.random(n) < 0.15, 1 - signal, signal).astype(int)  # 加 15% 噪声，避免指标虚高
    return df


@pytest.fixture
def text_df() -> pd.DataFrame:
    texts = ["网络断了无法上网", "宽带很慢经常掉线", "话费扣错了要退款", "账单金额不对", "想办理套餐升级", "咨询新套餐资费"] * 10
    labels = ["network", "network", "billing", "billing", "plan", "plan"] * 10
    return pd.DataFrame({"text": texts, "label": labels})


@pytest.fixture
def churn_csv(tmp_path, tabular_df):
    """小型表格数据 CSV：含缺失、类别列和空白字符串数值列。"""
    df = tabular_df.copy()
    df["customer_id"] = [f"C{i}" for i in range(len(df))]
    df["total_charges"] = [" " if i % 40 == 0 else f"{v:.1f}" for i, v in enumerate(df["tenure"] * 50)]
    path = tmp_path / "churn.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def fast_verify():
    from factory.verify import VerifyConfig

    return VerifyConfig(cv_folds=3, seeds=[0, 1], smoke_rows=120, timeout_seconds=60, report_metrics=["roc_auc"], n_threads=1,
                        max_metric_std=0.1)  # 小样本下种子间波动较大


@pytest.fixture(autouse=True)
def _isolate_registry():
    """测试中注册的生成算子不能泄漏到其他测试。"""
    from factory.operators import registry

    ops, sources = dict(registry._ops), dict(registry.sources)
    yield
    registry._ops.clear()
    registry._ops.update(ops)
    registry.sources.clear()
    registry.sources.update(sources)
