"""预测30天内可能流失的客户，需可解释，召回优先。比较三个方案，至少一个方案使用清 —— 自动生成的算法流水线

需求：预测30天内可能流失的客户，需可解释，召回优先。比较三个方案，至少一个方案使用清洗、通用缺失值填补、类别编码、标准化和平衡类权重的逻辑回归；可增加在网时长分桶，引用检索到的历史实现。
方案：logistic_regression_pipeline
设计依据：这是满足任务要求的基准方案，使用逻辑回归，包含必要的预处理和样本权重配置。
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

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class TenureBucketizer(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="TenureBucketizer",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="将数值型在网时长特征划分为业务导向的区间，增强线性模型的解释性。",
        conditions=["输入必须包含数值型tenure列"],
        dependencies=["numpy", "pandas", "sklearn"],
        params={"col": "tenure"},
        parent="TenureBucketizer@1"
    )

    def __init__(self, col="tenure"):
        self.col = col

    def fit(self, X, y=None):
        """记录算子状态，本算子为无状态转换，仅返回self。"""
        self.bins_ = [-1, 6, 12, 24, 48, np.inf]
        self.labels_ = ["0-6", "6-12", "12-24", "24-48", "48+"]
        return self

    def transform(self, X):
        """
        将tenure列分桶，处理缺失值及非数值情况。
        遵循数据清洗规范：先处理潜在脏数据，确保输出类型一致。
        """
        X = X.copy()

        # 处理脏数据：先确保转为数值，无法转换的(含空值)处理为-1（即处于0-6区间内）
        raw_tenure = pd.to_numeric(X[self.col].astype(str).str.strip(), errors='coerce')
        raw_tenure = raw_tenure.fillna(-1)

        # 分桶操作
        bucket_col = f"{self.col}_bucket"
        X[bucket_col] = pd.cut(
            raw_tenure, 
            bins=self.bins_, 
            labels=self.labels_
        ).astype(str)

        # 确保分桶后的缺失值被处理（pd.cut可能在NaN输入下产生nan字符串）
        X[bucket_col] = X[bucket_col].replace("nan", "0-6")

        return X


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_imputer", TotalChargesCleaner()),  # 解决数值列被读取为字符串且包含空格导致的问题，引用 Lesson:69d51358。
        ("1_imputer", MissingValueImputer()),  # 处理其他列中的缺失值。
        ("2_feature", TenureBucketizer()),  # 对在网时长进行分桶处理，提高可解释性并提升线性模型表现。
        ("3_encoder", OneHotCategoricalEncoder(max_categories=20)),  # 类别特征 One-Hot 编码。
        ("4_scaler", StandardFeatureScaler()),  # 逻辑回归必须进行标准化，引用 Lesson:74e2da10。
        ("5_model", LogisticRegressionModel(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)),  # 逻辑回归模型，通过 balanced 权重应对样本不平衡，提升召回率。
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
