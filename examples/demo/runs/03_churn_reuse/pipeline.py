"""预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户。优先复用已沉淀的清洗 —— 自动生成的算法流水线

需求：预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户。优先复用已沉淀的清洗算子，至少一个方案采用通用缺失值填补、类别编码、标准化和平衡类权重逻辑回归。
方案：logistic_regression_pipeline
设计依据：该方案基于逻辑回归，强调模型的可解释性与稳健的特征预处理，通过剔除ID、清洗数值字符串以及标准化，符合业务场景要求。
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


import numpy as np
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
        description="自动识别并剔除数据中的标识类(ID)特征，防止后续编码导致维度灾难。",
        conditions=["数据包含标识列"],
        dependencies=[],
        params={"id_columns": None}
    )

    def __init__(self, id_columns=None):
        self.id_columns = id_columns

    def fit(self, X, y=None):
        # 如果未指定id_columns，通过启发式规则自动识别：
        # 若某列唯一值数量等于行数，或列名包含id且唯一值极高，则视为ID列
        if self.id_columns is None:
            n_rows = len(X)
            self.id_columns_ = [
                col for col in X.columns 
                if X[col].nunique() == n_rows or ('id' in col.lower() and X[col].nunique() > n_rows * 0.9)
            ]
        else:
            self.id_columns_ = self.id_columns
        return self

    def transform(self, X):
        X = X.copy()
        # 移除标识列，过滤掉不存在于X中的列名以防报错
        cols_to_drop = [c for c in self.id_columns_ if c in X.columns]
        X = X.drop(columns=cols_to_drop)

        # 确保算子输出无无穷值(虽通常不产生，但按规范清理)
        # 注意：此处仅处理数值列，类别列不受影响
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)

        return X

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind, Modality

class TotalChargesCleaner(TransformerMixin, Operator):
    """
    清理包含空白符或脏字符的total_charges列并转换为浮点型，缺失值默认为0。
    """
    meta = OperatorMeta(
        name="TotalChargesCleaner",
        kind=OperatorKind.IMPUTER,
        version=2,
        origin="synthesized",
        description="清理包含空白符的total_charges列并转换为浮点型",
        conditions=["数据存在字符串格式的数值列", "存在包含空白字符的缺失值"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"col": "total_charges"},
        parent="TotalChargesCleaner@1"
    )

    def __init__(self, col="total_charges"):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.col in X.columns:
            # 根据Lesson:69d51358经验：先strip，再转数值（强制转换空值），最后填充缺失值
            X[self.col] = pd.to_numeric(
                X[self.col].astype(str).str.strip(), 
                errors="coerce"
            ).fillna(0.0)

            # 确保转换后没有无穷值，保持特征稳定性
            X[self.col] = X[self.col].replace([np.inf, -np.inf], 0.0)

        return X


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_feature", IdColumnDropper()),  # 根据Lesson:88299a38，建模前需剔除标识列。
        ("1_imputer", TotalChargesCleaner(col='total_charges')),  # 根据Lesson:69d51358，需要对string_numeric列进行清洗以避免类型转换失败。
        ("2_imputer", MissingValueImputer()),  # 处理剩余列的缺失值。
        ("3_encoder", OneHotCategoricalEncoder(max_categories=20)),  # 对低基数类别特征进行One-Hot编码。
        ("4_scaler", StandardFeatureScaler()),  # 线性模型需要标准化输入以确保收敛与解释性。
        ("5_model", LogisticRegressionModel(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)),  # 逻辑回归具备良好的可解释性，且通过class_weight解决样本不平衡问题。
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
