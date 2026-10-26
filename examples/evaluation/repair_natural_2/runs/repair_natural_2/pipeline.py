# 警告：该方案未通过验证，仅为得分最高的候选，供人工参考。
"""预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75；将 tot —— 自动生成的算法流水线

需求：预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75；将 total_charges 的空白字符串转换并填补为合理数值。
方案：random_forest_pipeline
设计依据：利用随机森林处理复杂的非线性关系，引入消费比值特征以捕捉流失信号。
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
from factory.operators.tabular import OneHotCategoricalEncoder


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
        description="将total_charges列中的脏字符串（含空白符）转换为数值，并用0填补缺失值",
        conditions=["数据包含字符串类型的数值列", "存在因空白符导致的转换错误"],
        dependencies=["pandas", "numpy"],
        params={"col": "total_charges"},
        parent="TotalChargesCleaner@1"
    )

    def __init__(self, col="total_charges"):
        self.col = col

    def fit(self, X, y=None):
        """记录列名，该算子无需学习统计量"""
        self.col_ = self.col
        return self

    def transform(self, X):
        """
        执行转换：
        1. 复制输入以保证不可变性
        2. 处理字符串中的空白符并强制转换为浮点数
        3. 处理转换产生的缺失值（赋值为0）
        """
        X = X.copy()

        if self.col_ in X.columns:
            # 依据 Lessons 修复：先 strip，再 coerce，最后用 0 填充
            # 确保转换过程不抛出异常
            X[self.col_] = pd.to_numeric(
                X[self.col_].astype(str).str.strip(), 
                errors="coerce"
            ).fillna(0.0)

            # 处理可能存在的无穷值，确保输出的数值稳定性
            X[self.col_] = X[self.col_].replace([np.inf, -np.inf], 0.0)

        return X

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class ChargeRatioFeatureGenerator(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="ChargeRatioFeatureGenerator",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="生成消费比值特征，通过处理total_charges脏数据并计算总费与月费比例",
        conditions=["数据中包含total_charges, monthly_charges, tenure字段"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"monthly": "monthly_charges", "total": "total_charges", "tenure": "tenure"},
        parent="ChargeRatioFeatureGenerator@1"
    )

    def __init__(self, monthly="monthly_charges", total="total_charges", tenure="tenure"):
        self.monthly = monthly
        self.total = total
        self.tenure = tenure

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # 处理 total_charges 脏数据 (Lesson:69d51358)
        # 将空白字符串和非法字符转为数值，缺失值填补为 0
        if self.total in X.columns:
            X[self.total] = pd.to_numeric(X[self.total].astype(str).str.strip(), errors='coerce').fillna(0.0)

        # 确保 monthly_charges 和 tenure 为数值类型
        X[self.monthly] = pd.to_numeric(X[self.monthly], errors='coerce').fillna(0.0)
        X[self.tenure] = pd.to_numeric(X[self.tenure], errors='coerce').fillna(0.0)

        # 计算特征: avg_monthly = total / tenure (tenure=0时为0)
        # 避免除以0，tenure为0时比值记为1
        tenure_safe = X[self.tenure].replace(0, np.nan)
        avg_calc = X[self.total] / tenure_safe

        # 计算比值: (avg_calc / monthly)
        # 处理无穷值和缺失值
        ratio = avg_calc / X[self.monthly]
        X["charge_ratio"] = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        return X


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_imputer", TotalChargesCleaner()),  # 数据清洗第一步。
        ("1_feature", ChargeRatioFeatureGenerator()),  # 通过领域知识构造特征提升模型感知能力。
        ("2_encoder", OneHotCategoricalEncoder(max_categories=20)),  # 编码。
        ("3_model", RandomForestModel(n_estimators=400, max_depth=12, class_weight='balanced', random_state=42)),  # 利用随机森林非线性拟合能力，通过调整深度控制过拟合。
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
