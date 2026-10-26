# AlgoForge 验证报告

| 项目 | 内容 |
| --- | --- |
| 需求 | 预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75 |
| 数据 | data/churn.csv |
| 运行 ID | kg_no_lessons_3 |
| LLM | gemini-3.5-flash |
| 状态 | ❌ 未通过 |
| 最优方案 | lgbm_churn_prediction_pipeline |
| 总耗时 | 18.9 s |

共验证 1 个候选方案，0 个通过；重新规划 0 次，修复 0 轮。最优方案 `lgbm_churn_prediction_pipeline` 的 recall = 0.603（要求 ≥ 0.75）。

## 1. 需求理解（TaskSpec）

| 字段 | 值 |
| --- | --- |
| 任务类型 | binary_classification |
| 数据模态 | tabular |
| 目标列 | churn |
| 评价指标 | recall ≥ 0.75（主） |
| 业务约束 | interpretable、recall_first |
| 数据画像标签 | has_missing_values、numeric_features、has_categorical_features、low_cardinality、string_typed_numeric、id_columns、imbalanced、time_tenure、mixed_features |

## 2. 知识检索（设计依据）

匹配场景：telecom_churn。每类步骤从知识图谱检索候选算子，得分 = 0.4 × 文本相关度 + 0.3 × 条件匹配 + 0.3 × 历史验证分。

| 步骤 | 候选算子 | 可直接复用 | 得分 | 文本/条件/历史 | 证据 |
| --- | --- | --- | --- | --- | --- |
| imputer | MissingValueImputer@1 | 是 | 0.484 | 0.60 / 0.09 / 0.72 | 匹配条件：has_missing_values；历史验证 3 次，平均 recall 0.717 |
| imputer | TotalChargesCleaner@1 | 否（历史实现，可参考生成） | 0.482 | 1.00 / 0.27 / 0.00 | 匹配条件：has_missing_values、numeric_features、string_typed_numeric |
| feature | TenureBucketizer@1 | 否（历史实现，可参考生成） | 0.708 | 0.96 / 0.27 / 0.80 | 匹配条件：interpretable、numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | ChargeRatioFeatureGenerator@1 | 否（历史实现，可参考生成） | 0.620 | 0.81 / 0.18 / 0.80 | 匹配条件：numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | ServiceCountFeature@1 | 否（历史实现，可参考生成） | 0.427 | 1.00 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| feature | IdColumnDropper@1 | 否（历史实现，可参考生成） | 0.373 | 0.86 / 0.09 / 0.00 | 匹配条件：id_columns |
| encoder | OneHotCategoricalEncoder@1 | 是 | 0.669 | 1.00 / 0.18 / 0.72 | 匹配条件：has_categorical_features、low_cardinality；历史验证 3 次，平均 recall 0.717 |
| encoder | KFoldTargetEncoder@1 | 否（历史实现，可参考生成） | 0.402 | 0.94 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| scaler | StandardFeatureScaler@1 | 是 | 0.640 | 1.00 / 0.09 / 0.71 | 匹配条件：numeric_features；历史验证 1 次，平均 recall 0.710 |
| model | LogisticRegressionModel@1 | 是 | 0.640 | 1.00 / 0.09 / 0.71 | 匹配条件：interpretable；历史验证 1 次，平均 recall 0.710 |
| model | BalancedLightGBMClassifier@1 | 否（历史实现，可参考生成） | 0.604 | 0.86 / 0.09 / 0.78 | 匹配条件：imbalanced；历史验证 1 次，平均 recall 0.780 |
| model | RandomForestModel@1 | 是 | 0.549 | 0.81 / 0.09 / 0.66 | 匹配条件：mixed_features；历史验证 1 次，平均 recall 0.660 |
| model | LightGBMModel@1 | 是 | 0.535 | 0.67 / 0.09 / 0.80 | 匹配条件：imbalanced；历史验证 1 次，平均 recall 0.800 |

### 知识引用检查

检索并传入上下文不等于采纳。以下引用由模型声明，程序仅核对可见范围与结构化适用对象；不证明语义落实或指标收益。

| 方案 | 步骤 | 模型引用 | 范围检查 | 语义落实 |
| --- | --- | --- | --- | --- |
| lgbm_churn_prediction_pipeline | 1 | Operator:TotalChargesCleaner@1 | 通过 | 未验证 |
| lgbm_churn_prediction_pipeline | 2 | Operator:MissingValueImputer@1 | 通过 | 未验证 |
| lgbm_churn_prediction_pipeline | 3 | Operator:IdColumnDropper@1 | 通过 | 未验证 |
| lgbm_churn_prediction_pipeline | 4 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| lgbm_churn_prediction_pipeline | 5 | Operator:LightGBMModel@1、ValidationRun:f6e4284c | 通过 | 未验证 |

## 3. 候选方案比较

排序规则：先看是否通过全部检查，再比主指标，再比次要指标之和，最后训练耗时短者优先。指标为多随机种子 × 分层 K 折交叉验证的均值 ± 种子间标准差。

| 排名 | 方案 | 结果 | recall | roc_auc | precision | f1 | recall_at_20 | 单次训练 (s) | 峰值内存 (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ★ lgbm_churn_prediction_pipeline | 未通过 | 0.603 ± 0.007 | 0.780 ± 0.003 | 0.494 ± 0.002 | 0.543 ± 0.002 | 0.418 ± 0.009 | 0.349 | 263 |

```
recall（| 为阈值 0.75）
lgbm_churn_prediction_pipeline       ■■■■■■■■■■■■···|···· 0.603
```

### 各方案的结构与诊断

- `lgbm_churn_prediction_pipeline`：生成:TotalChargesCleaner → MissingValueImputer → 生成:IdColumnDropper → OneHotCategoricalEncoder → LightGBMModel。诊断：`M_THRESHOLD` recall = 0.6030 低于阈值 0.75

## 4. 最优方案：lgbm_churn_prediction_pipeline

本方案针对流失预测场景，采用高质量的预处理手段（TotalChargesCleaner）确保数据可用性，利用IdColumnDropper避免过拟合，并选择历史效果最好的LightGBM模型，通过class_weight均衡策略确保高召回率。

| # | 步骤 | 算子 | 参数 | 选择理由 |
| --- | --- | --- | --- | --- |
| 1 | imputer | 生成 TotalChargesCleaner | - | 任务数据包含字符串类型的数值列，必须先进行预处理以转化为数值型供后续模型使用。 |
| 2 | imputer | MissingValueImputer@1 | - | 处理剩余数值特征和类别特征的缺失值，保证数据完整性。 |
| 3 | feature | 生成 IdColumnDropper | - | 客户ID无预测价值，需剔除避免模型记忆化。 |
| 4 | encoder | OneHotCategoricalEncoder@1 | - | 对类别特征进行One-Hot编码，满足模型输入要求。 |
| 5 | model | LightGBMModel@1 | class_weight=balanced, learning_rate=0.05, n_estimators=500, num_leaves=31, random_state=42 | LightGBM在表格数据且不平衡的情况下表现优异，通过设置class_weight=balanced优先保证召回率达到0.75的目标。 |

## 5. 能力生成与修复

| 生成算子 | 类型 | 用途 | 参考的历史实现 | 修复轮数 | 注册到能力库 |
| --- | --- | --- | --- | --- | --- |
| TotalChargesCleaner@2 | imputer | 将包含非数值字符的total_charges列清理并转换为浮点型，处理数据脏值问题 | TotalChargesCleaner@1 | 0 | 否 |
| IdColumnDropper@2 | feature | 删除客户ID列，防止模型过拟合噪声 | IdColumnDropper@1 | 0 | 否 |

## 6. 验证机制

| 级别 | 检查内容 | 本次配置 |
| --- | --- | --- |
| 1 静态检查 | 语法、import 白名单、禁止危险调用、接口与 meta 一致性 | 仅针对生成的算子 |
| 2 沙箱执行 | 独立子进程，限制超时与内存，清除含 KEY/TOKEN 的环境变量 | 超时 180 s，内存 8192 MB |
| 3 契约检查 | 不修改输入、行数一致、不引入 NaN/Inf、概率合法、能处理未见类别与缺失值 | 冒烟样本 300 行 |
| 4 指标检查 | 阈值、种子间稳定性、疑似泄漏、耗时与内存 | 5 折 × 种子 [42, 7, 2024]；std 上限 0.03 |

最优方案资源消耗：单次训练 0.349 s，预测 55.5 ms / 千行，峰值内存 263 MB。

## 7. 知识沉淀

- 回写验证记录 ValidationRun：1 条
- 注册到能力库的新算子：无
- 新提炼的经验：无
- 本次提供的经验：0 条（累计检索命中，不代表采纳）

## 附录 A. 生成的算法代码（pipeline.py）

```python
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

```

## 附录 B. 运行记录

各阶段耗时：PLAN 5.3s，BUILD 5.7s，VERIFY 7.9s，ROUTE 0.0s，CURATE 0.0s

```
[PLAN] 场景 telecom_churn；画像标签 ['has_missing_values', 'numeric_features', 'has_categorical_features', 'low_cardinality', 'string_typed_numeric', 'id_columns', 'imbalanced', 'time_tenure', 'mixed_features']
[PLAN] TaskSpec：binary_classification，目标 churn，主指标 recall >= 0.75
[PLAN] 候选方案：['lgbm_churn_prediction_pipeline(*TotalChargesCleaner → MissingValueImputer → *IdColumnDropper → OneHotCategoricalEncoder → LightGBMModel)']
[PLAN] 完成，用时 5.3s → BUILD
[BUILD] 生成算子 TotalChargesCleaner@2（参考 TotalChargesCleaner@1）：将包含非数值字符的total_charges列清理并转换为浮点型，处理数据脏值问题
[BUILD] 生成算子 IdColumnDropper@2（参考 IdColumnDropper@1）：删除客户ID列，防止模型过拟合噪声
[BUILD] 完成，用时 5.7s → VERIFY
[VERIFY] lgbm_churn_prediction_pipeline：未通过，recall=0.6030；M_THRESHOLD recall = 0.6030 低于阈值 0.75
[VERIFY] 完成，用时 7.9s → ROUTE
[ROUTE] 没有方案通过验证
[ROUTE] 完成，用时 0.0s → CURATE
[CURATE] 写入 KG：1 条 ValidationRun；注册新算子 []；新增经验 0 条
[CURATE] 完成，用时 0.0s → FAILED
```

### LLM 调用来源

| 阶段 | 请求哈希 | 来源 | 本次耗时(s) | 录制耗时(s) | tokens |
| --- | --- | --- | --- | --- | --- |
| plan | 1d72b68fe551d421 | remote | 5.110 | - | 9231 |
| synthesize | 1cd2b3253803b10e | remote | 2.960 | - | 2137 |
| synthesize | cbdcec8627ab88dd | remote | 2.773 | - | 1945 |
