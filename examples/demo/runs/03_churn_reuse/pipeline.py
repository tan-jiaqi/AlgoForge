"""预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户 —— 自动生成的算法流水线

需求：预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户
方案：linear_interpretable_lr
设计依据：针对流失预测的高召回与高解释性要求，先清洗数据与分箱增强特征表达，再通过类别均衡的逻辑回归直观输出各特征权重系数。
主指标：recall >= 0.75

由 AlgoForge 自动生成。用法：
    python pipeline.py --data <train.csv>                      # 训练并保存 model.joblib
    python pipeline.py --predict <new.csv> --model model.joblib  # 预测
"""

import argparse

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from factory.operators.models import LogisticRegressionModel
from factory.operators.tabular import MissingValueImputer, OneHotCategoricalEncoder, StandardFeatureScaler


from typing import Sequence
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
        description="移除客户ID等无业务预测意义的标识列，防止编码维度暴涨和过拟合",
        conditions=["id_columns"],
        dependencies=[],
        params={"id_cols": ("customer_id",)},
        parent="IdColumnDropper@1",
    )

    def __init__(self, id_cols: Sequence[str] = ("customer_id",)):
        self.id_cols = id_cols

    def fit(self, X: pd.DataFrame, y=None):
        cols_to_check = set(self.id_cols) if self.id_cols is not None else set()
        self.drop_cols_ = [col for col in X.columns if col in cols_to_check]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cols_to_drop = [c for c in self.drop_cols_ if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
        return X

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind


class TotalChargesCleaner(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="TotalChargesCleaner",
        kind=OperatorKind.IMPUTER,
        version=2,
        origin="synthesized",
        description="安全清洗并转换含空白字符串的累计消费等数值列为浮点型，并填充缺失值为指定默认值",
        conditions=["string_typed_numeric", "has_missing_values"],
        dependencies=[],
        params={"col": "total_charges", "fill_value": 0.0},
        parent="TotalChargesCleaner@1",
    )

    def __init__(self, col: str = "total_charges", fill_value: float = 0.0):
        self.col = col
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.col in X.columns:
            cleaned_series = (
                pd.to_numeric(X[self.col].astype(str).str.strip(), errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(self.fill_value)
                .astype(float)
            )
            X[self.col] = cleaned_series
        return X

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind


class TenureBucketizer(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="TenureBucketizer",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="将在网时长字段分箱为离散区间特征以增强线性模型的表达能力与业务可解释性",
        conditions=["time_tenure", "numeric_features"],
        dependencies=["pandas", "numpy"],
        params={
            "col": "tenure",
            "bins": (-float("inf"), 6.0, 12.0, 24.0, 48.0, float("inf")),
            "labels": ("0-6", "6-12", "12-24", "24-48", "48+"),
            "drop_original": False,
        },
        parent="TenureBucketizer@1",
    )

    def __init__(
        self,
        col: str = "tenure",
        bins: tuple = (-float("inf"), 6.0, 12.0, 24.0, 48.0, float("inf")),
        labels: tuple = ("0-6", "6-12", "12-24", "24-48", "48+"),
        drop_original: bool = False,
    ):
        self.col = col
        self.bins = bins
        self.labels = labels
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_name_ = f"{self.col}_bucket"
        self.bins_ = list(self.bins) if self.bins is not None else [-float("inf"), 6.0, 12.0, 24.0, 48.0, float("inf")]
        self.labels_ = list(self.labels) if self.labels is not None else ["0-6", "6-12", "12-24", "24-48", "48+"]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.col not in X.columns:
            return X

        # 安全数值转换，兼顾脏数据、空白字符串与缺失值
        tenure_series = X[self.col]
        if tenure_series.dtype == object:
            tenure_series = tenure_series.astype(str).str.strip()
        numeric_tenure = pd.to_numeric(tenure_series, errors="coerce")

        # 分箱并处理缺失区间
        bucketed = pd.cut(numeric_tenure, bins=self.bins_, labels=self.labels_)
        bucketed_str = bucketed.astype(str)
        bucketed_str = bucketed_str.replace("nan", "unknown").fillna("unknown")

        X[self.feature_name_] = bucketed_str

        if self.drop_original:
            X = X.drop(columns=[self.col])

        return X


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_feature", IdColumnDropper()),  # 根据经验必须在编码前剔除 customer_id 等标识列，避免 One-Hot 导致特征维度爆炸与内存溢出。
        ("1_imputer", TotalChargesCleaner(col='total_charges', fill_value=0.0)),  # 解决数值列含空白字符导致类型转换失败的问题，安全转换 total_charges 为浮点型并填补缺省值。
        ("2_imputer", MissingValueImputer()),  # 对 monthly_charges 与 payment_method 的缺失值分别采用中位数与众数进行填补，确保数据完整性。
        ("3_feature", TenureBucketizer(col='tenure', bins=(float('-inf'), 6.0, 12.0, 24.0, 48.0, float('inf')), labels=('0-6', '6-12', '12-24', '24-48', '48+'), drop_original=False)),  # 将在网时长离散化分箱，提升线性模型的非线性拟合能力并增强业务分群解释性。
        ("4_encoder", OneHotCategoricalEncoder(max_categories=20)),  # 对低基数类别特征进行 One-Hot 编码，转换成适合线性模型计算的稀疏特征。
        ("5_scaler", StandardFeatureScaler()),  # 根据经验使用逻辑回归前必须先对数值特征进行 Z-score 标准化，保证收敛与权重可比。
        ("6_model", LogisticRegressionModel(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)),  # 配置 class_weight='balanced' 解决流失样本不平衡以达成高召回率要求，且回归系数直观可解释。
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="训练数据 CSV")
    parser.add_argument("--predict", help="待预测数据 CSV")
    parser.add_argument("--model", default="model.joblib")
    args = parser.parse_args()
    if args.data:
        df = pd.read_csv(args.data)
        pipe = build_pipeline().fit(df.drop(columns=[TARGET]), df[TARGET])
        joblib.dump(pipe, args.model)
        print(f"模型已保存到 {args.model}")
    if args.predict:
        pipe = joblib.load(args.model)
        df = pd.read_csv(args.predict)
        X = df.drop(columns=[TARGET], errors="ignore")
        out = pd.DataFrame(pipe.predict_proba(X), columns=[f"proba_{c}" for c in pipe.classes_])
        out["prediction"] = pipe.predict(X)
        print(out.head().to_string())


if __name__ == "__main__":
    main()
