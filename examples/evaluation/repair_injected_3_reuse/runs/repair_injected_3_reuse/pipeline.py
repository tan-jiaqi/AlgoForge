"""预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户，召回率不低于0.75 —— 自动生成的算法流水线

需求：预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户，召回率不低于0.75
方案：interpretable_logreg_pipeline
设计依据：该方案利用逻辑回归的可解释性和特征标准化，结合针对性的数据清洗，确保模型在满足召回率阈值的前提下具备良好的业务解释能力。
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


import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class DropIdColumn(TransformerMixin, Operator):
    """
    删除客户ID及高基数标识列，防止后续处理（如One-Hot编码）导致维度爆炸。
    """
    meta = OperatorMeta(
        name="DropIdColumn",
        kind=OperatorKind.FEATURE,
        version=1,
        origin="synthesized",
        description="删除指定的ID列或高基数标识列，防止维度爆炸并保证数据完整性。",
        conditions=["数据包含customer_id等标识列"],
        dependencies=["pandas", "numpy"],
        params={"id_cols": ("customer_id",)},
        parent="IdColumnDropper@1"
    )

    def __init__(self, id_cols=("customer_id",)):
        self.id_cols = id_cols

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = X.copy()

        # 识别存在于列中的待删除ID列
        cols_to_drop = [c for c in self.id_cols if c in X.columns]
        X = X.drop(columns=cols_to_drop)

        # 根据 Lesson:cf247d47，确保输出不包含 NaN 或 Inf
        # 对于数值列，填充缺失值为0或中位数，此处选择填充为0以保证鲁棒性
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
            X[numeric_cols] = X[numeric_cols].fillna(0)

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
        description="将字符串类型的总费用列转换为数值，处理空白异常值并增加在网时长分桶特征",
        conditions=["数据包含total_charges列", "存在tenure在网时长列"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"col": "total_charges", "tenure_col": "tenure"},
        parent="TotalChargesCleaner@1"
    )

    def __init__(self, col="total_charges", tenure_col="tenure"):
        self.col = col
        self.tenure_col = tenure_col

    def fit(self, X, y=None):
        # 记录清洗后的均值或固定值用于兜底
        self.mean_val_ = 0.0
        return self

    def transform(self, X):
        X = X.copy()

        # 处理 total_charges 列
        if self.col in X.columns:
            # 强制转换为数值，空白及非法字符变为 NaN
            X[self.col] = pd.to_numeric(X[self.col].astype(str).str.strip(), errors="coerce")
            # 处理 NaN 和 Inf，统一填充为 0.0
            X[self.col] = X[self.col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        # 处理 tenure 分桶特征
        if self.tenure_col in X.columns:
            bins = [0, 6, 12, 24, 48, 60, 1000]
            labels = ["0-6m", "7-12m", "13-24m", "25-48m", "49-60m", "60m+"]
            # 确保输入为数值进行分桶
            tenure_series = pd.to_numeric(X[self.tenure_col], errors="coerce")
            X["tenure_binned"] = pd.cut(
                tenure_series, 
                bins=bins, 
                labels=labels, 
                include_lowest=True
            ).astype(str)
            # 处理分桶过程中产生的 NaN (即原始数据中缺失或超出范围的 tenure)
            X["tenure_binned"] = X["tenure_binned"].replace("nan", "unknown").fillna("unknown")

        return X


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_imputer", TotalChargesCleaner(col='total_charges', tenure_col='tenure')),  # 处理总费用列的字符串异常并集成时长特征，满足业务需求且有历史验证记录。
        ("1_imputer", MissingValueImputer()),  # 兜底填补其他数值与类别列缺失值，确保数据完整性。
        ("2_feature", DropIdColumn()),  # 剔除ID标识列以避免编码维度暴涨，符合 Lesson:88299a38。
        ("3_encoder", OneHotCategoricalEncoder(max_categories=10)),  # 对类别特征进行One-Hot编码，限制维度以提升模型表现。
        ("4_scaler", StandardFeatureScaler()),  # 逻辑回归必须标准化，以便输出可解释的系数。
        ("5_model", LogisticRegressionModel(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)),  # 使用逻辑回归提供高可解释性，并设置 balanced 权重以满足召回率需求。
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
