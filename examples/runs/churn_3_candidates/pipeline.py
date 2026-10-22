"""预测30天内可能流失的客户，需可解释，召回优先，并比较多种建模方案 —— 自动生成的算法流水线

需求：预测30天内可能流失的客户，需可解释，召回优先，并比较多种建模方案
方案：interpretable_logistic_regression
设计依据：针对业务高可解释性及召回率诉求，首先安全清洗字符串数值列并填补缺失值，One-Hot编码后进行标准Z-Score缩放，最后采用平衡权重的逻辑回归模型输出线性可解释系数。
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

class TotalChargesCleaner(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="TotalChargesCleaner",
        kind=OperatorKind.IMPUTER,
        version=2,
        origin="synthesized",
        description="将 total_charges 中的空白字符处理为缺失并转换为浮点型，填补为0.0",
        conditions=["string_typed_numeric"],
        dependencies=[],
        params={"col": "total_charges"},
        parent="TotalChargesCleaner@1"
    )

    def __init__(self, col="total_charges"):
        self.col = col

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        if self.col in X.columns:
            # 先将列转换为字符串并去除首尾空白字符，再转换为数值型（无法转换的变NaN），最后填补为0.0
            X[self.col] = pd.to_numeric(X[self.col].astype(str).str.strip(), errors="coerce").fillna(0.0)
        return X


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_imputer", TotalChargesCleaner()),  # 根据 lessons，total_charges 包含空白字符直接转数值会报错，需先做清洗转换。
        ("1_imputer", MissingValueImputer()),  # 数据中 monthly_charges 和 payment_method 仍有缺失值，需要中位数和众数填充。
        ("2_encoder", OneHotCategoricalEncoder(max_categories=20)),  # 类别特征基数较低，适用 One-Hot 编码且需限制类别数防止维度发散。
        ("3_scaler", StandardFeatureScaler()),  # 根据经验，逻辑回归对特征尺度敏感，必须先进行标准化。
        ("4_model", LogisticRegressionModel(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)),  # 逻辑回归具有高可解释性，配置 class_weight='balanced' 应对样本不平衡以优先提升召回率。
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
