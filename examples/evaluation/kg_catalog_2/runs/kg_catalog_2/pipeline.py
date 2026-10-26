# 警告：该方案未通过验证，仅为得分最高的候选，供人工参考。
"""预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75 —— 自动生成的算法流水线

需求：预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75
方案：random_forest_churn_prediction
设计依据：通过预处理清理脏数据，使用随机森林模型平衡召回率与可解释性需求，并针对不平衡数据使用平衡权重策略。
主指标：recall >= 0.75

由 AlgoForge 自动生成。用法：
    python pipeline.py --data <train.csv>                      # 训练并保存 model.joblib
    python pipeline.py --predict <new.csv> --model model.joblib  # 预测
"""

import argparse

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from factory.operators.models import RandomForestModel
from factory.operators.tabular import MissingValueImputer, OneHotCategoricalEncoder


import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class DataPreprocessor(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="DataPreprocessor",
        kind=OperatorKind.FEATURE,
        version=1,
        origin="synthesized",
        description="删除标识列并处理字符串格式的数值特征，同时填补缺失值。",
        conditions=["输入包含customer_id及字符串数值列"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"id_columns": ["customer_id"], "numeric_cols": ["total_charges"]}
    )

    def __init__(self, id_columns=["customer_id"], numeric_cols=["total_charges"]):
        self.id_columns = id_columns
        self.numeric_cols = numeric_cols

    def fit(self, X, y=None):
        X = X.copy()
        self.medians_ = {}
        for col in self.numeric_cols:
            if col in X.columns:
                series = pd.to_numeric(X[col], errors='coerce')
                self.medians_[col] = series.median() if not series.isna().all() else 0.0
        return self

    def transform(self, X):
        X = X.copy()

        # 1. 删除标识列
        X = X.drop(columns=[c for c in self.id_columns if c in X.columns], errors='ignore')

        # 2. 转换指定数值列
        for col in self.numeric_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].replace([np.inf, -np.inf], np.nan)
                X[col] = X[col].fillna(self.medians_.get(col, 0.0))

        # 3. 对整体DataFrame其他列处理（规避复杂dtype导致issubdtype报错）
        for col in X.columns:
            if X[col].isnull().any():
                # 使用类型检查而非复杂的issubdtype，优先处理数值列
                if pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = X[col].fillna(X[col].median() if not X[col].dropna().empty else 0.0)
                else:
                    # 显式转换为对象类型以处理复杂类型缺失
                    X[col] = X[col].fillna("Missing").astype(str)

        return X


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_feature", DataPreprocessor()),  # 需要处理客户ID列带来的干扰以及 total_charges 列的字符串格式问题，这是数据预处理的第一步。
        ("1_imputer", MissingValueImputer()),  # 针对 payment_method 和 monthly_charges 等字段的缺失值进行中位数/众数填补。
        ("2_encoder", OneHotCategoricalEncoder(max_categories=20)),  # 将类别型特征转换为数值型以供模型读取，利用其对低基数特征的处理能力。
        ("3_model", RandomForestModel(n_estimators=300, max_depth=None, class_weight='balanced', random_state=42)),  # 随机森林在处理表格数据时稳健性好，且可通过特征重要性满足可解释性要求；class_weight='balanced' 针对不平衡数据进行优化，以提升召回率。
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
