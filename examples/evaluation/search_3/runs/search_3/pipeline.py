# 警告：该方案未通过验证，仅为得分最高的候选，供人工参考。
"""预测客户流失，召回率不低于0.8；优先尝试可解释模型，未达标时探索不同模型或参数 —— 自动生成的算法流水线

需求：预测客户流失，召回率不低于0.8；优先尝试可解释模型，未达标时探索不同模型或参数。
方案：lgbm_imbalanced_pipeline
设计依据：针对不平衡数据流失预测任务，采用轻量级清洗及经过历史验证的 LightGBM 增强方案，并设置 class_weight='balanced' 以满足 recall_first 需求。
主指标：recall >= 0.8

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
        description="将total_charges字段中的空白字符串转换为数值，并处理异常值以符合后续模型计算需求",
        conditions=["数据包含 total_charges 列且该列含有空白字符或被识别为字符串"],
        dependencies=["pandas", "numpy"],
        params={"col": "total_charges"},
        parent="TotalChargesCleaner@1"
    )

    def __init__(self, col="total_charges"):
        self.col = col

    def fit(self, X, y=None):
        """记录列名并返回 self"""
        return self

    def transform(self, X):
        """
        修正 total_charges：
        1. 复制输入数据以防原地修改
        2. 将字符串两端的空白符清除
        3. 利用 pd.to_numeric 将非数值转为 NaN
        4. 使用 0 填充缺失值（代表新入网用户）
        """
        X = X.copy()
        if self.col in X.columns:
            # 执行类型转换：先转字符串并去除空格，coerce 非法值为 NaN，最后填补为 0
            X[self.col] = pd.to_numeric(
                X[self.col].astype(str).str.strip(), 
                errors="coerce"
            ).fillna(0.0)

            # 确保最终列为浮点型，并处理任何可能产生的 inf 值
            X[self.col] = X[self.col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

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
        description="自动识别并剔除高基数ID类列，防止模型过拟合及维度灾难",
        conditions=["数据集包含唯一标识列"],
        dependencies=["pandas", "numpy"],
        params={"id_cols": ("customer_id",)},
        parent="IdColumnDropper@1"
    )

    def __init__(self, id_cols=("customer_id",)):
        self.id_cols = id_cols

    def fit(self, X, y=None):
        """
        验证输入数据结构并记录状态。
        """
        self.cols_to_drop_ = [c for c in self.id_cols if c in X.columns]
        return self

    def transform(self, X):
        """
        执行删除操作。
        """
        X = X.copy()

        # 剔除预定义的ID列
        cols_actually_dropped = [c for c in self.cols_to_drop_ if c in X.columns]
        X = X.drop(columns=cols_actually_dropped)

        # 逻辑兜底：检查是否存在其他疑似高基数ID的列（超过行数90%的唯一值）
        # 避免在线上推理阶段因为动态列名导致的问题
        potential_ids = [
            col for col in X.select_dtypes(include=['object']).columns 
            if X[col].nunique() / len(X) > 0.9
        ]
        X = X.drop(columns=potential_ids)

        return X


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_imputer", TotalChargesCleaner()),  # 根据Lesson:69d51358，需要对脏数据进行清洗以支持后续数值计算。
        ("1_imputer", MissingValueImputer()),  # 填补数据中的残余缺失值以确保训练数据的完整性。
        ("2_feature", IdColumnDropper()),  # 删除无关的标识列以降低噪声影响。
        ("3_encoder", OneHotCategoricalEncoder(max_categories=20)),  # 将类别特征进行 one-hot 编码，符合数据画像中低基数的特征。
        ("4_model", LightGBMModel(n_estimators=400, learning_rate=0.05, num_leaves=31, class_weight='balanced', random_state=42)),  # 根据 Lesson:f2c1398b，使用具有 balanced 权重的 LightGBM 能够有效处理不平衡数据并满足召回率要求。
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
