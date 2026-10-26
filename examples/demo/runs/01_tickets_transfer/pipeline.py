"""把客服工单自动分到网络故障、账单争议、套餐变更、销户投诉四类，宏平均 F1 不低 —— 自动生成的算法流水线

需求：把客服工单自动分到网络故障、账单争议、套餐变更、销户投诉四类，宏平均 F1 不低于 0.8
方案：tfidf_logistic_baseline
设计依据：使用高效的文本向量化方案配合平衡权重逻辑回归，是处理中文短文本分类的经典基线方案。
主指标：f1_macro >= 0.8

由 AlgoForge 自动生成。用法：
    python pipeline.py --data <train.csv>                      # 训练并保存 model.joblib
    python pipeline.py --predict <new.csv> --model model.joblib  # 预测
"""

import argparse

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from factory.operators.models import LogisticRegressionModel
from factory.operators.text import TfidfTextVectorizer


TARGET = 'label'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_vectorizer", TfidfTextVectorizer(column='text', analyzer='char_wb', ngram_range=(1, 2), max_features=20000)),  # 针对中文短文本，采用字符级 1-2 gram TF-IDF 向量化处理，符合最佳实践。
        ("1_model", LogisticRegressionModel(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)),  # 针对样本不平衡问题，设置 class_weight='balanced' 以提升宏平均 F1 分数。
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
