"""预测客户流失，召回率不低于0.8；优先尝试可解释模型，未达标时探索不同模型或参数 —— 自动生成的算法流水线

需求：预测客户流失，召回率不低于0.8；优先尝试可解释模型，未达标时探索不同模型或参数。
方案：interpretable_logreg_pipeline
设计依据：针对可解释性约束，使用逻辑回归，并通过加大正类权重系数（class_weight）强行拉升召回率。
主指标：recall >= 0.8

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


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_imputer", MissingValueImputer()),
        ("1_encoder", OneHotCategoricalEncoder(max_categories=20)),
        ("2_scaler", StandardFeatureScaler()),
        ("3_model", LogisticRegressionModel(C=0.5, class_weight={'0': 1, '1': 4}, max_iter=1000, random_state=42)),
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
