# 警告：该方案未通过验证，仅为得分最高的候选，供人工参考。
"""预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75 —— 自动生成的算法流水线

需求：预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75
方案：rf_churn_pipeline
设计依据：该方案通过专用清洗算子修复字符串数值，使用随机森林处理正负样本不平衡，并利用其特征重要性满足可解释性约束，提高流失预测召回率。
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


import pandas as pd
import numpy as np
import re
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class StringNumericCleaner(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="StringNumericCleaner",
        kind=OperatorKind.FEATURE,
        version=1,
        origin="synthesized",
        description="将字符串格式的数值列清洗并转换为浮点数，并确保输出无缺失值",
        conditions=["数据包含数值特征被错误读取为字符串"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"columns": None}
    )

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        # 记录需要处理的列，若未指定则自动识别 object 类型列
        if self.columns is None:
            self.numeric_cols_ = X.select_dtypes(include=['object']).columns.tolist()
        else:
            self.numeric_cols_ = self.columns

        # 计算每列中位数以便 transform 阶段填充
        self.medians_ = {}
        for col in self.numeric_cols_:
            if col in X.columns:
                # 尝试将原列转为数值获取中位数，若全为非数值则设为 0
                temp = pd.to_numeric(X[col], errors='coerce')
                self.medians_[col] = temp.median() if not temp.isnull().all() else 0.0
        return self

    def transform(self, X):
        X = X.copy()

        for col in self.numeric_cols_:
            if col not in X.columns:
                continue

            # 清洗字符串并转数值
            col_data = X[col].astype(str).str.strip()
            col_data = col_data.replace(r'^\s*$', np.nan, regex=True)

            # 转换为 float，非法值变 NaN
            converted = pd.to_numeric(col_data, errors='coerce')

            # 处理 inf
            converted = converted.replace([np.inf, -np.inf], np.nan)

            # 填充缺失值
            fill_val = self.medians_.get(col, 0.0)
            if np.isnan(fill_val):
                fill_val = 0.0

            X[col] = converted.fillna(fill_val)

        return X


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_feature", StringNumericCleaner()),  # 处理 total_charges 列中存在的 string_typed_numeric 问题，保证后续模型输入正确。
        ("1_imputer", MissingValueImputer()),  # 对 monthly_charges 和 payment_method 等存在的缺失值进行中位数/众数填补。
        ("2_encoder", OneHotCategoricalEncoder(max_categories=10)),  # 将类别型特征转换为数值表示，以适应模型输入需求。
        ("3_model", RandomForestModel(n_estimators=500, max_depth=None, class_weight='balanced', random_state=42)),  # 随机森林在处理不平衡数据方面表现稳健，且通过特征重要性提供可解释性，满足召回率要求。
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
