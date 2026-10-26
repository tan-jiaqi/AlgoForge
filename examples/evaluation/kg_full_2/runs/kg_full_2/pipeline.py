# 警告：该方案未通过验证，仅为得分最高的候选，供人工参考。
"""预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75 —— 自动生成的算法流水线

需求：预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75
方案：lightgbm_churn_prediction
设计依据：该方案针对电信流失场景，优先解决脏数据和ID列导致的内存及转换问题，并采用经过验证的LightGBM模型以达到召回率阈值。
主指标：recall >= 0.75

由 AlgoForge 自动生成。用法：
    python pipeline.py --data <train.csv>                      # 训练并保存 model.joblib
    python pipeline.py --predict <new.csv> --model model.joblib  # 预测
"""

import argparse

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from factory.operators.models import LightGBMModel
from factory.operators.tabular import MissingValueImputer, OneHotCategoricalEncoder


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
        description="将字符串类型的 total_charges 列转换为浮点型，处理空白值并填充缺失值",
        conditions=["包含 total_charges 字符串列"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"col": "total_charges"},
        parent="TotalChargesCleaner@1"
    )

    def __init__(self, col="total_charges"):
        self.col = col

    def fit(self, X, y=None):
        """记录列名并返回 self，该算子无状态学习。"""
        self.col_ = self.col
        return self

    def transform(self, X):
        """
        处理逻辑：
        1. 复制输入数据防止污染原数据
        2. 将指定列转为字符串并去除首尾空白
        3. 使用 pd.to_numeric 将无法解析的值转为 NaN
        4. 使用 0.0 填充转换后的 NaN（针对新客户逻辑）
        5. 处理可能存在的无穷值并填充为 0.0
        """
        X = X.copy()
        if self.col_ in X.columns:
            # 1. 清洗及类型转换
            clean_series = pd.to_numeric(
                X[self.col_].astype(str).str.strip(), 
                errors="coerce"
            )

            # 2. 填充缺失值
            clean_series = clean_series.fillna(0.0)

            # 3. 处理可能产生的无穷值
            clean_series = clean_series.replace([np.inf, -np.inf], 0.0)

            X[self.col_] = clean_series

        return X

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class IdColumnDropper(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="IdColumnDropper",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="剔除客户 ID 等高基数标识列，防止维度爆炸和过拟合",
        conditions=["数据包含高基数标识列"],
        dependencies=[],
        params={"id_cols": ["customer_id"]},
        parent="IdColumnDropper@1"
    )

    def __init__(self, id_cols=["customer_id"]):
        self.id_cols = id_cols

    def fit(self, X, y=None):
        """记录输入列以供校验，此处仅返回self"""
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):
        """
        剔除指定的 ID 列。
        根据踩坑经验：必须先 copy 原始数据，防止原地修改影响下游；
        处理时需确保即使列不存在也不报错。
        """
        X = X.copy()
        cols_to_drop = [c for c in self.id_cols if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
        return X


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_imputer", TotalChargesCleaner()),  # 根据 Lesson:69d51358，需先处理 total_charges 列的脏数据，防止类型转换错误。
        ("1_imputer", MissingValueImputer()),  # 处理 monthly_charges 等其他列的缺失值。
        ("2_feature", IdColumnDropper()),  # 根据 Lesson:88299a38，建模前必须剔除 ID 列。
        ("3_encoder", OneHotCategoricalEncoder(max_categories=10)),  # 对类别特征进行编码，限制类别数量以控制特征维度。
        ("4_model", LightGBMModel(n_estimators=500, learning_rate=0.05, num_leaves=31, class_weight='balanced', random_state=42)),  # 使用 LightGBM 并配合 class_weight='balanced' 优化召回率，满足召回优先的业务目标，参考 Lesson:9e9e77d6。
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
