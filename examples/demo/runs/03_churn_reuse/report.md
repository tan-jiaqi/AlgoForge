# AlgoForge 验证报告

| 项目 | 内容 |
| --- | --- |
| 需求 | 预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户。优先复用已沉淀的清洗算子，至少一个方案采用通用缺失值填补、类别编码、标准化和平衡类权重逻辑回归。 |
| 数据 | data/churn.csv |
| 运行 ID | 03_churn_reuse |
| LLM | gemini-3.5-flash |
| 状态 | ✅ 成功 |
| 最优方案 | logistic_regression_pipeline |
| 总耗时 | 11.9 s |

共验证 2 个候选方案，1 个通过；重新规划 0 次，修复 0 轮。最优方案 `logistic_regression_pipeline` 的 recall = 0.777（要求 ≥ 0.75）。

## 1. 需求理解（TaskSpec）

| 字段 | 值 |
| --- | --- |
| 任务类型 | binary_classification |
| 数据模态 | tabular |
| 目标列 | churn |
| 评价指标 | recall ≥ 0.75（主）；roc_auc ≥ 0.8 |
| 业务约束 | interpretable、recall_first |
| 数据画像标签 | has_missing_values、numeric_features、has_categorical_features、low_cardinality、string_typed_numeric、id_columns、imbalanced、time_tenure、mixed_features |

## 2. 知识检索（设计依据）

匹配场景：telecom_churn。每类步骤从知识图谱检索候选算子，得分 = 0.4 × 文本相关度 + 0.3 × 条件匹配 + 0.3 × 历史验证分。

| 步骤 | 候选算子 | 可直接复用 | 得分 | 文本/条件/历史 | 证据 |
| --- | --- | --- | --- | --- | --- |
| imputer | MissingValueImputer@1 | 是 | 0.647 | 1.00 / 0.09 / 0.73 | 匹配条件：has_missing_values；历史验证 4 次，平均 recall 0.732 |
| imputer | TotalChargesCleaner@2 | 是 | 0.501 | 0.67 / 0.00 / 0.78 | 历史验证 1 次，平均 recall 0.778 |
| imputer | TotalChargesCleaner@1 | 否（历史实现，可参考生成） | 0.437 | 0.89 / 0.27 / 0.00 | 匹配条件：has_missing_values、numeric_features、string_typed_numeric |
| feature | ChargeRatioFeatureGenerator@1 | 否（历史实现，可参考生成） | 0.651 | 0.89 / 0.18 / 0.80 | 匹配条件：numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | TenureBucketizer@1 | 否（历史实现，可参考生成） | 0.638 | 0.79 / 0.27 / 0.80 | 匹配条件：interpretable、numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | TenureBucketizer@2 | 是 | 0.519 | 0.71 / 0.00 / 0.78 | 历史验证 1 次，平均 recall 0.778 |
| encoder | OneHotCategoricalEncoder@1 | 是 | 0.617 | 0.86 / 0.18 / 0.73 | 匹配条件：has_categorical_features、low_cardinality；历史验证 4 次，平均 recall 0.732 |
| encoder | KFoldTargetEncoder@1 | 否（历史实现，可参考生成） | 0.427 | 1.00 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| scaler | StandardFeatureScaler@1 | 是 | 0.650 | 1.00 / 0.09 / 0.74 | 匹配条件：numeric_features；历史验证 2 次，平均 recall 0.744 |
| model | BalancedLightGBMClassifier@1 | 否（历史实现，可参考生成） | 0.661 | 1.00 / 0.09 / 0.78 | 匹配条件：imbalanced；历史验证 1 次，平均 recall 0.780 |
| model | LogisticRegressionModel@1 | 是 | 0.600 | 0.87 / 0.09 / 0.74 | 匹配条件：interpretable；历史验证 2 次，平均 recall 0.744 |
| model | LightGBMModel@1 | 是 | 0.535 | 0.67 / 0.09 / 0.80 | 匹配条件：imbalanced；历史验证 1 次，平均 recall 0.800 |

### 参考的历史经验

- **逻辑回归标准化与可解释性**：必须先进行特征标准化（如 StandardFeatureScaler），在 0.1~10 之间搜索正则化强度 C，可直接输出系数给业务方。
- **数值列含空白字符导致类型转换失败**（E_RUNTIME）：先 str.strip()，再 pd.to_numeric(errors='coerce')，缺失值填补为 0
- **样本不平衡导致默认模型召回率偏低**（M_THRESHOLD）：启用平衡类权重（如 class_weight='balanced' 或 BalancedLightGBMClassifier）以提升召回率
- **数值特征清洗需显式处理空白字符与零除异常**（E_RUNTIME）：在数据清洗阶段，应显式将空/空白字符串替换为 np.nan，并使用 pd.to_numeric(errors='coerce') 转换；在数值运算阶段，应使用 np.errstate 上下文管理器捕获溢出/除零异常，并统一使用 replace/fillna 对 inf 和 NaN 进行健壮性填充。
- **未剔除客户 ID 导致 One-Hot 编码后维度暴涨与内存溢出**（E_TIMEOUT）：建模前剔除 ID 等标识列，并在编码器中设置最大类别数

### 知识引用检查

检索并传入上下文不等于采纳。以下引用由模型声明，程序仅核对可见范围与结构化适用对象；不证明语义落实或指标收益。

| 方案 | 步骤 | 模型引用 | 范围检查 | 语义落实 |
| --- | --- | --- | --- | --- |
| logistic_regression_pipeline | 1 | Lesson:88299a38 | 通过 | 未验证 |
| logistic_regression_pipeline | 2 | Operator:TotalChargesCleaner@2、Lesson:69d51358 | 通过 | 未验证 |
| logistic_regression_pipeline | 3 | Operator:MissingValueImputer@1 | 通过 | 未验证 |
| logistic_regression_pipeline | 4 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| logistic_regression_pipeline | 5 | Operator:StandardFeatureScaler@1、Lesson:74e2da10 | 通过 | 未验证 |
| logistic_regression_pipeline | 6 | Operator:LogisticRegressionModel@1、Lesson:9e9e77d6 | 通过 | 未验证 |
| lightgbm_pipeline | 1 | Lesson:88299a38 | 通过 | 未验证 |
| lightgbm_pipeline | 2 | Operator:TotalChargesCleaner@2 | 通过 | 未验证 |
| lightgbm_pipeline | 3 | Operator:TenureBucketizer@2 | 通过 | 未验证 |
| lightgbm_pipeline | 4 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| lightgbm_pipeline | 5 | Operator:LightGBMModel@1、Lesson:9e9e77d6 | 通过 | 未验证 |

## 3. 候选方案比较

排序规则：先看是否通过全部检查，再比主指标，再比次要指标之和，最后训练耗时短者优先。指标为多随机种子 × 分层 K 折交叉验证的均值 ± 种子间标准差。

| 排名 | 方案 | 结果 | recall | roc_auc | precision | f1 | recall_at_20 | 单次训练 (s) | 峰值内存 (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ★ logistic_regression_pipeline | 通过 | 0.777 ± 0.002 | 0.814 ± 0.000 | 0.483 ± 0.003 | 0.596 ± 0.003 | 0.457 ± 0.002 | 0.064 | 254 |
| 2 | lightgbm_pipeline | 未通过 | 0.648 ± 0.011 | 0.785 ± 0.002 | 0.491 ± 0.004 | 0.558 ± 0.006 | 0.427 ± 0.002 | 0.391 | 271 |

```
recall（| 为阈值 0.75）
logistic_regression_pipeline         ■■■■■■■■■■■■■■■|···· 0.777
lightgbm_pipeline                    ■■■■■■■■■■■■■··|···· 0.648
```

### 各方案的结构与诊断

- `logistic_regression_pipeline`：生成:IdColumnDropper → TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel。诊断：无
- `lightgbm_pipeline`：生成:IdColumnDropper → TotalChargesCleaner → TenureBucketizer → OneHotCategoricalEncoder → LightGBMModel。诊断：`M_THRESHOLD` recall = 0.6476 低于阈值 0.75；`M_THRESHOLD` roc_auc = 0.7847 低于阈值 0.8

## 4. 最优方案：logistic_regression_pipeline

该方案基于逻辑回归，强调模型的可解释性与稳健的特征预处理，通过剔除ID、清洗数值字符串以及标准化，符合业务场景要求。

| # | 步骤 | 算子 | 参数 | 选择理由 |
| --- | --- | --- | --- | --- |
| 1 | feature | 生成 IdColumnDropper | - | 根据Lesson:88299a38，建模前需剔除标识列。 |
| 2 | imputer | TotalChargesCleaner@2 | col=total_charges | 根据Lesson:69d51358，需要对string_numeric列进行清洗以避免类型转换失败。 |
| 3 | imputer | MissingValueImputer@1 | - | 处理剩余列的缺失值。 |
| 4 | encoder | OneHotCategoricalEncoder@1 | max_categories=20 | 对低基数类别特征进行One-Hot编码。 |
| 5 | scaler | StandardFeatureScaler@1 | - | 线性模型需要标准化输入以确保收敛与解释性。 |
| 6 | model | LogisticRegressionModel@1 | class_weight=balanced, C=1.0 | 逻辑回归具备良好的可解释性，且通过class_weight解决样本不平衡问题。 |

## 5. 能力生成与修复

| 生成算子 | 类型 | 用途 | 参考的历史实现 | 修复轮数 | 注册到能力库 |
| --- | --- | --- | --- | --- | --- |
| IdColumnDropper@2 | feature | 移除ID类特征，避免One-Hot编码导致维度灾难 | - | 0 | 是 |

## 6. 验证机制

| 级别 | 检查内容 | 本次配置 |
| --- | --- | --- |
| 1 静态检查 | 语法、import 白名单、禁止危险调用、接口与 meta 一致性 | 仅针对生成的算子 |
| 2 沙箱执行 | 独立子进程，限制超时与内存，清除含 KEY/TOKEN 的环境变量 | 超时 180 s，内存 8192 MB |
| 3 契约检查 | 不修改输入、行数一致、不引入 NaN/Inf、概率合法、能处理未见类别与缺失值 | 冒烟样本 300 行 |
| 4 指标检查 | 阈值、种子间稳定性、疑似泄漏、耗时与内存 | 5 折 × 种子 [42, 7, 2024]；std 上限 0.03 |

最优方案资源消耗：单次训练 0.064 s，预测 29.5 ms / 千行，峰值内存 254 MB。

## 7. 知识沉淀

- 回写验证记录 ValidationRun：2 条
- 注册到能力库的新算子：IdColumnDropper@2
- 新提炼的经验：无
- 本次提供的经验：6 条（累计检索命中，不代表采纳）

## 附录 A. 生成的算法代码（pipeline.py）

```python
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

```

## 附录 B. 运行记录

各阶段耗时：PLAN 0.2s，BUILD 0.0s，VERIFY 11.7s，ROUTE 0.0s，CURATE 0.0s

```
[PLAN] 场景 telecom_churn；画像标签 ['has_missing_values', 'numeric_features', 'has_categorical_features', 'low_cardinality', 'string_typed_numeric', 'id_columns', 'imbalanced', 'time_tenure', 'mixed_features']
[PLAN] TaskSpec：binary_classification，目标 churn，主指标 recall >= 0.75
[PLAN] 候选方案：['logistic_regression_pipeline(*IdColumnDropper → TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel)', 'lightgbm_pipeline(*IdColumnDropper → TotalChargesCleaner → TenureBucketizer → OneHotCategoricalEncoder → LightGBMModel)']
[PLAN] 完成，用时 0.2s → BUILD
[BUILD] 生成算子 IdColumnDropper@2：移除ID类特征，避免One-Hot编码导致维度灾难
[BUILD] 完成，用时 0.0s → VERIFY
[VERIFY] logistic_regression_pipeline：通过，recall=0.7768
[VERIFY] lightgbm_pipeline：未通过，recall=0.6476；M_THRESHOLD recall = 0.6476 低于阈值 0.75；M_THRESHOLD roc_auc = 0.7847 低于阈值 0.8
[VERIFY] 完成，用时 11.7s → ROUTE
[ROUTE] 完成，用时 0.0s → CURATE
[CURATE] 写入 KG：2 条 ValidationRun；注册新算子 ['IdColumnDropper@2']；新增经验 0 条
[CURATE] 完成，用时 0.0s → DONE
```

### LLM 调用来源

| 阶段 | 请求哈希 | 来源 | 本次耗时(s) | 录制耗时(s) | tokens |
| --- | --- | --- | --- | --- | --- |
| plan_spec | 6d6b50b3b2cf16a1 | replay | 0.000 | 1.803 | 1993 |
| plan | c0c5913c5ef145c6 | replay | 0.000 | 4.491 | 12302 |
| synthesize | eb30cbd5a5c451f7 | replay | 0.000 | 2.766 | 2497 |
