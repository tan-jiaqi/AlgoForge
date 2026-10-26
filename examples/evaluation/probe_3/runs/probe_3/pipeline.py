# 警告：该方案未通过验证，仅为得分最高的候选，供人工参考。
"""预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75；增加累计费 —— 自动生成的算法流水线

需求：预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75；增加累计费用与月费的比值特征，正确处理缺失和零分母。
方案：random_forest_robust_pipeline
设计依据：使用随机森林，通过特征工程增强模型，利用森林天然的非线性捕捉能力提升召回表现。
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

class ServiceCountFeature(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="ServiceCountFeature",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="统计客户订购的增值服务数量并计算累计费用与月费的比值，以量化客户粘性",
        conditions=["数据包含增值服务列", "数据包含月费与总费用列"],
        dependencies=["numpy", "pandas", "sklearn"],
        params={
            "service_cols": ["online_security", "tech_support", "streaming_tv"],
            "monthly_col": "monthly_charges",
            "total_col": "total_charges"
        },
        parent="ServiceCountFeature@1"
    )

    def __init__(self, service_cols=["online_security", "tech_support", "streaming_tv"], monthly_col="monthly_charges", total_col="total_charges"):
        self.service_cols = service_cols
        self.monthly_col = monthly_col
        self.total_col = total_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # 1. 统计增值服务数量 (假设“是”为订购)
        # 将非“是”的值视作未订购，处理缺失值风险
        X["n_services"] = 0
        for col in self.service_cols:
            if col in X.columns:
                X["n_services"] += (X[col].astype(str).str.strip() == "是").astype(int)

        # 2. 处理数值列：清理空白字符并转为 float
        # 针对 total_charges 可能存在的 ' ' 空格字符串问题
        for col in [self.monthly_col, self.total_col]:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col].astype(str).str.strip(), errors='coerce').fillna(0.0)

        # 3. 计算累计费用/月费比值 (隐含在网时长特征)
        # 规避零分母：若 monthly_charges 为 0 则比值设为 0
        denominator = X[self.monthly_col].replace(0, np.nan)
        X["charge_ratio"] = (X[self.total_col] / denominator).fillna(0.0)

        # 4. 处理最终可能产生的无穷值或缺失值
        X["charge_ratio"] = X["charge_ratio"].replace([np.inf, -np.inf], 0.0).fillna(0.0)

        return X


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_imputer", MissingValueImputer()),  # 通用填补策略。
        ("1_feature", ServiceCountFeature()),  # 利用业务逻辑提取粘性特征提升模型对复杂流失行为的建模能力。
        ("2_encoder", OneHotCategoricalEncoder(max_categories=20)),  # 类别特征编码。
        ("3_model", RandomForestModel(n_estimators=300, max_depth=12, class_weight='balanced', random_state=42)),  # 随机森林鲁棒性强，设置深度防过拟合，并引入服务数量特征。
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
