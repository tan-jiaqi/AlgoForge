# 警告：该方案未通过验证，仅为得分最高的候选，供人工参考。
"""预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75 —— 自动生成的算法流水线

需求：预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75
方案：lgbm_churn_prediction_pipeline
设计依据：本方案针对流失预测场景，采用高质量的预处理手段（TotalChargesCleaner）确保数据可用性，利用IdColumnDropper避免过拟合，并选择历史效果最好的LightGBM模型，通过class_weight均衡策略确保高召回率。
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
        description="将total_charges列中的脏字符串数据转换为浮点型，并使用0填充缺失值",
        conditions=["数据包含字符串类型的数值列", "存在包含空白字符的无效数据"],
        dependencies=["pandas", "numpy"],
        params={"col": "total_charges"},
        parent="TotalChargesCleaner@1"
    )

    def __init__(self, col="total_charges"):
        self.col = col

    def fit(self, X, y=None):
        """记录列名并返回自身，无需学习状态"""
        return self

    def transform(self, X):
        """将指定列转换为数值类型，处理无效字符串并填充缺失值"""
        X = X.copy()
        if self.col in X.columns:
            # 1. 强制转为字符串并去除首尾空白
            # 2. to_numeric 将无法转换的字符串(如空字符、非数值串)转为 NaN
            # 3. 用 0.0 填充转换后产生的 NaN (针对新客户或脏数据)
            X[self.col] = pd.to_numeric(
                X[self.col].astype(str).str.strip(), 
                errors="coerce"
            ).fillna(0.0)

            # 处理可能产生的无穷大值，确保数据健壮性
            X[self.col] = X[self.col].replace([np.inf, -np.inf], 0.0)

        return X

import pandas as pd
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class IdColumnDropper(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="IdColumnDropper",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="删除指定的客户ID或高基数标识列，防止模型过拟合噪声及特征空间膨胀。",
        conditions=["数据含客户 ID 等标识列", "数据含高基数唯一标识"],
        dependencies=[],
        params={"id_cols": ["customer_id"]},
        parent="IdColumnDropper@1"
    )

    def __init__(self, id_cols=["customer_id"]):
        self.id_cols = id_cols

    def fit(self, X, y=None):
        # 记录 fit 状态，本算子为无状态转换，但符合 sklearn 规范
        self.id_cols_ = list(self.id_cols)
        return self

    def transform(self, X):
        # 复制数据，保证输入不变性
        X = X.copy()

        # 识别出存在于列中的目标列
        cols_to_drop = [c for c in self.id_cols_ if c in X.columns]

        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)

        return X


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_imputer", TotalChargesCleaner()),  # 任务数据包含字符串类型的数值列，必须先进行预处理以转化为数值型供后续模型使用。
        ("1_imputer", MissingValueImputer()),  # 处理剩余数值特征和类别特征的缺失值，保证数据完整性。
        ("2_feature", IdColumnDropper()),  # 客户ID无预测价值，需剔除避免模型记忆化。
        ("3_encoder", OneHotCategoricalEncoder(max_categories=20)),  # 对类别特征进行One-Hot编码，满足模型输入要求。
        ("4_model", LightGBMModel(n_estimators=500, learning_rate=0.05, num_leaves=31, class_weight='balanced', random_state=42)),  # LightGBM在表格数据且不平衡的情况下表现优异，通过设置class_weight=balanced优先保证召回率达到0.75的目标。
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
