"""把客服工单自动分到网络故障、账单争议、套餐变更、销户投诉四类，宏平均 F1 不低 —— 自动生成的算法流水线

需求：把客服工单自动分到网络故障、账单争议、套餐变更、销户投诉四类，宏平均 F1 不低于 0.8
方案：tfidf_ngram3_lr_tuned
设计依据：通过扩大 n-gram 窗口提升特征表现力，结合微调后的正则化强度，力求达到 F1-macro 0.8 以上目标。
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
        ("0_vectorizer", TfidfTextVectorizer(column='text', analyzer='char_wb', ngram_range=(1, 3), max_features=30000)),  # 扩展 n-gram 范围至 1-3 gram 并增加特征维度，更好捕捉工单特定词组与长搭配语义。
        ("1_model", LogisticRegressionModel(C=2.0, class_weight='balanced', max_iter=1000, random_state=42)),  # 适度减小 L2 正则化约束（增大 C 值）配合更高维文本特征，同时保留平衡权重处理不平衡数据。
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
