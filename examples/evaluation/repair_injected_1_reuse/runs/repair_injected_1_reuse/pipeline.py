"""预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户，召回率不低于0.75 —— 自动生成的算法流水线

需求：预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户，召回率不低于0.75
方案：interpretable_logreg_pipeline
设计依据：采用线性模型，通过显式处理标识列、缺失值及标准化，满足可解释性且利用平衡权重提升召回率。
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


import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class IdColumnDropper(TransformerMixin, Operator):
    """
    删除客户ID及其他高基数标识列，防止后续编码时出现维度爆炸，并确保输出数据结构稳定。
    """
    meta = OperatorMeta(
        name="IdColumnDropper",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="删除指定的标识列，防止高基数特征导致的维度爆炸问题。",
        conditions=["数据包含标识列", "需要进行特征编码"],
        dependencies=[],
        params={"id_cols": ("customer_id",)},
        parent="IdColumnDropper@1"
    )

    def __init__(self, id_cols=("customer_id",)):
        self.id_cols = id_cols

    def fit(self, X, y=None):
        self.id_cols_ = list(self.id_cols)
        return self

    def transform(self, X):
        X = X.copy()
        # 识别当前 DataFrame 中存在的标识列
        cols_to_drop = [c for c in self.id_cols_ if c in X.columns]

        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)

        # 确保算子输出的健壮性：处理可能的 NaN/Inf
        # 尽管此算子仅做删除，但考虑到数据流完整性，在此处进行显式兜底校验
        X = X.replace([np.inf, -np.inf], np.nan)

        return X

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
        return self

    def transform(self, X):
        X = X.copy()

        # 处理 total_charges 列：转换为数值，强制处理非数值为 0，并确保结果无缺失或无穷大
        if self.col in X.columns:
            # 强制转换为数值类型，无法转换的转为 NaN
            clean_series = pd.to_numeric(X[self.col].astype(str).str.strip(), errors='coerce')
            # 填充 NaN 和无穷值为 0
            X[self.col] = clean_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 处理 tenure_col 列：生成分桶特征
        if self.tenure_col in X.columns:
            bins = [0, 6, 12, 24, 48, 60, 1000]
            labels = ['0-6m', '7-12m', '13-24m', '25-48m', '49-60m', '60m+']

            tenure_vals = pd.to_numeric(X[self.tenure_col], errors='coerce').fillna(0)
            X['tenure_binned'] = pd.cut(
                tenure_vals, 
                bins=bins, 
                labels=labels, 
                include_lowest=True
            ).astype(str)

            # 确保分桶结果没有空值
            X['tenure_binned'] = X['tenure_binned'].fillna('unknown')

        return X


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_feature", IdColumnDropper()),  # 根据 Lesson:88299a38，删除标识列避免维度过高。
        ("1_imputer", TotalChargesCleaner(col='total_charges', tenure_col='tenure')),  # 利用经过验证的算子处理总费用的缺失和类型转换问题。
        ("2_imputer", MissingValueImputer()),  # 对其他列进行通用缺失值填补。
        ("3_encoder", OneHotCategoricalEncoder(max_categories=20)),  # 对类别特征进行 One-Hot 编码。
        ("4_scaler", StandardFeatureScaler()),  # 根据 Lesson:74e2da10，逻辑回归必须先标准化。
        ("5_model", LogisticRegressionModel(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)),  # 逻辑回归具备高可解释性，且使用 balanced 权重应对样本不平衡。
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
