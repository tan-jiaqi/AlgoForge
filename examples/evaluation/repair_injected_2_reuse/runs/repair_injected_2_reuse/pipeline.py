"""预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户，召回率不低于0.75 —— 自动生成的算法流水线

需求：预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户，召回率不低于0.75
方案：interpretable_logreg_pipeline
设计依据：利用逻辑回归确保解释性，并遵循标准化与样本平衡的原则优化召回率。
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
        self.mean_total_charges_ = 0.0
        return self

    def transform(self, X):
        X = X.copy()

        # 处理 total_charges 列
        if self.col in X.columns:
            # 转换为数值，将无法转换的转为 NaN，然后填充为 0
            X[self.col] = pd.to_numeric(X[self.col].astype(str).str.strip(), errors="coerce")
            X[self.col] = X[self.col].fillna(0.0)
            # 处理无穷大
            X[self.col] = X[self.col].replace([np.inf, -np.inf], 0.0)

        # 处理 tenure 分桶
        if self.tenure_col in X.columns:
            # 确保 tenure 是数值，缺失值处理为 0
            tenure_series = pd.to_numeric(X[self.tenure_col], errors="coerce").fillna(0.0)
            bins = [0, 6, 12, 24, 48, 60, 1000]
            labels = ["0-6m", "7-12m", "13-24m", "25-48m", "49-60m", "60m+"]
            X["tenure_binned"] = pd.cut(
                tenure_series, 
                bins=bins, 
                labels=labels, 
                include_lowest=True
            ).astype(str)
            # 填充处理可能出现的 NaN
            X["tenure_binned"] = X["tenure_binned"].fillna("unknown")

        return X


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_imputer", TotalChargesCleaner(col='total_charges', tenure_col='tenure')),  # 处理总费用列的格式及缺失值，这是逻辑回归的先决步骤。
        ("1_imputer", MissingValueImputer()),  # 补全剩余的缺失值。
        ("2_encoder", OneHotCategoricalEncoder(max_categories=20)),  # 对类别特征进行编码以适应线性模型。
        ("3_scaler", StandardFeatureScaler()),  # 逻辑回归必须进行标准化以确保模型可解释性。
        ("4_model", LogisticRegressionModel(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)),  # 使用逻辑回归提供高解释性，并配置 balanced 权重优化召回率。
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
