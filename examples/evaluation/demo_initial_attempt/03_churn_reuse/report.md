# AlgoForge 验证报告

| 项目 | 内容 |
| --- | --- |
| 需求 | 预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户 |
| 数据 | data/churn.csv |
| 运行 ID | 03_churn_reuse |
| LLM | gemini-3.5-flash |
| 状态 | ✅ 成功 |
| 最优方案 | interpretable_logreg_pipeline |
| 总耗时 | 43.2 s |

共验证 2 个候选方案，1 个通过；重新规划 0 次，修复 0 轮。最优方案 `interpretable_logreg_pipeline` 的 recall = 0.777（要求 ≥ 0.75）。

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
| imputer | MissingValueImputer@1 | 是 | 0.500 | 0.64 / 0.09 / 0.72 | 匹配条件：has_missing_values；历史验证 3 次，平均 recall 0.717 |
| imputer | TotalChargesCleaner@1 | 否（历史实现，可参考生成） | 0.482 | 1.00 / 0.27 / 0.00 | 匹配条件：has_missing_values、numeric_features、string_typed_numeric |
| feature | TenureBucketizer@1 | 否（历史实现，可参考生成） | 0.644 | 0.81 / 0.27 / 0.80 | 匹配条件：interpretable、numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | ChargeRatioFeatureGenerator@1 | 否（历史实现，可参考生成） | 0.603 | 0.77 / 0.18 / 0.80 | 匹配条件：numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | ServiceCountFeature@1 | 否（历史实现，可参考生成） | 0.427 | 1.00 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| encoder | OneHotCategoricalEncoder@1 | 是 | 0.640 | 0.93 / 0.18 / 0.72 | 匹配条件：has_categorical_features、low_cardinality；历史验证 3 次，平均 recall 0.717 |
| encoder | KFoldTargetEncoder@1 | 否（历史实现，可参考生成） | 0.427 | 1.00 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| scaler | StandardFeatureScaler@1 | 是 | 0.640 | 1.00 / 0.09 / 0.71 | 匹配条件：numeric_features；历史验证 1 次，平均 recall 0.710 |
| model | BalancedLightGBMClassifier@1 | 否（历史实现，可参考生成） | 0.661 | 1.00 / 0.09 / 0.78 | 匹配条件：imbalanced；历史验证 1 次，平均 recall 0.780 |
| model | LogisticRegressionModel@1 | 是 | 0.628 | 0.97 / 0.09 / 0.71 | 匹配条件：interpretable；历史验证 1 次，平均 recall 0.710 |
| model | RandomForestModel@1 | 是 | 0.602 | 0.94 / 0.09 / 0.66 | 匹配条件：mixed_features；历史验证 1 次，平均 recall 0.660 |

### 参考的历史经验

- **数值列含空白字符导致类型转换失败**（E_RUNTIME）：先 str.strip()，再 pd.to_numeric(errors='coerce')，缺失值填补为 0
- **样本不平衡导致默认模型召回率偏低**（M_THRESHOLD）：启用平衡类权重（如 class_weight='balanced' 或 BalancedLightGBMClassifier）以提升召回率
- **逻辑回归标准化与可解释性**：必须先进行特征标准化（如 StandardFeatureScaler），在 0.1~10 之间搜索正则化强度 C，可直接输出系数给业务方。
- **未剔除客户 ID 导致 One-Hot 编码后维度暴涨与内存溢出**（E_TIMEOUT）：建模前剔除 ID 等标识列，并在编码器中设置最大类别数
- **移除对 pandas 特定 API 的依赖以增强输入类型兼容性**（E_RUNTIME）：在数据预处理阶段，优先进行类型检查，通过转换为 NumPy 数组或使用通用的数值清洗逻辑，避免直接调用特定数据结构（如 DataFrame）独有的 API，从而确保对密集矩阵、稀疏矩阵及 DataFrame 的广泛兼容性。

### 知识引用检查

检索并传入上下文不等于采纳。以下引用由模型声明，程序仅核对可见范围与结构化适用对象；不证明语义落实或指标收益。

| 方案 | 步骤 | 模型引用 | 范围检查 | 语义落实 |
| --- | --- | --- | --- | --- |
| random_forest_pipeline | 1 | Operator:TotalChargesCleaner@1、Lesson:69d51358 | 通过 | 未验证 |
| random_forest_pipeline | 2 | Operator:MissingValueImputer@1 | 通过 | 未验证 |
| random_forest_pipeline | 3 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| random_forest_pipeline | 4 | Operator:RandomForestModel@1、Lesson:9e9e77d6 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 1 | Operator:TotalChargesCleaner@1、Lesson:69d51358 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 2 | Operator:MissingValueImputer@1 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 3 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 4 | Operator:StandardFeatureScaler@1、Lesson:74e2da10 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 5 | Operator:LogisticRegressionModel@1、Lesson:9e9e77d6 | 通过 | 未验证 |

## 3. 候选方案比较

排序规则：先看是否通过全部检查，再比主指标，再比次要指标之和，最后训练耗时短者优先。指标为多随机种子 × 分层 K 折交叉验证的均值 ± 种子间标准差。

| 排名 | 方案 | 结果 | recall | roc_auc | precision | f1 | recall_at_20 | 单次训练 (s) | 峰值内存 (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ★ interpretable_logreg_pipeline | 通过 | 0.777 ± 0.002 | 0.814 ± 0.001 | 0.486 ± 0.001 | 0.598 ± 0.002 | 0.457 ± 0.001 | 0.087 | 257 |
| 2 | random_forest_pipeline | 未通过 | 0.594 ± 0.005 | 0.787 ± 0.002 | 0.518 ± 0.001 | 0.553 ± 0.003 | 0.429 ± 0.005 | 1.054 | 359 |

```
recall（| 为阈值 0.75）
interpretable_logreg_pipeline        ■■■■■■■■■■■■■■■|···· 0.777
random_forest_pipeline               ■■■■■■■■■■■■···|···· 0.594
```

### 各方案的结构与诊断

- `interpretable_logreg_pipeline`：生成:TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel。诊断：无
- `random_forest_pipeline`：生成:TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → RandomForestModel。诊断：`M_THRESHOLD` recall = 0.5939 低于阈值 0.75；`M_THRESHOLD` roc_auc = 0.7871 低于阈值 0.8

## 4. 最优方案：interpretable_logreg_pipeline

本方案侧重于模型的可解释性，使用逻辑回归满足业务对原因分析的需求，并配套标准化算子。

| # | 步骤 | 算子 | 参数 | 选择理由 |
| --- | --- | --- | --- | --- |
| 1 | imputer | 生成 TotalChargesCleaner | - | 前置清洗逻辑，保证数据类型正确。 |
| 2 | imputer | MissingValueImputer@1 | - | 填充常规缺失值。 |
| 3 | encoder | OneHotCategoricalEncoder@1 | - | 对类别特征编码，为线性模型准备输入。 |
| 4 | scaler | StandardFeatureScaler@1 | - | 根据 Lesson:74e2da10，逻辑回归必须在标准化特征后训练，以保证系数可解释性。 |
| 5 | model | LogisticRegressionModel@1 | class_weight=balanced, C=0.1, max_iter=1000 | 逻辑回归提供高可解释性，通过正则化强度调整和样本权重平衡，兼顾模型稳定性和召回要求。 |

## 5. 能力生成与修复

| 生成算子 | 类型 | 用途 | 参考的历史实现 | 修复轮数 | 注册到能力库 |
| --- | --- | --- | --- | --- | --- |
| TotalChargesCleaner@2 | imputer | 将字符串类型的数值列转换为浮点数，处理空白字符并填补缺失值 | TotalChargesCleaner@1 | 0 | 是 |

## 6. 验证机制

| 级别 | 检查内容 | 本次配置 |
| --- | --- | --- |
| 1 静态检查 | 语法、import 白名单、禁止危险调用、接口与 meta 一致性 | 仅针对生成的算子 |
| 2 沙箱执行 | 独立子进程，限制超时与内存，清除含 KEY/TOKEN 的环境变量 | 超时 180 s，内存 8192 MB |
| 3 契约检查 | 不修改输入、行数一致、不引入 NaN/Inf、概率合法、能处理未见类别与缺失值 | 冒烟样本 300 行 |
| 4 指标检查 | 阈值、种子间稳定性、疑似泄漏、耗时与内存 | 5 折 × 种子 [42, 7, 2024]；std 上限 0.03 |

最优方案资源消耗：单次训练 0.087 s，预测 43.4 ms / 千行，峰值内存 257 MB。

## 7. 知识沉淀

- 回写验证记录 ValidationRun：2 条
- 注册到能力库的新算子：TotalChargesCleaner@2
- 新提炼的经验：无
- 本次提供的经验：6 条（累计检索命中，不代表采纳）

## 附录 A. 生成的算法代码（pipeline.py）

```python
"""预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户 —— 自动生成的算法流水线

需求：预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户
方案：interpretable_logreg_pipeline
设计依据：本方案侧重于模型的可解释性，使用逻辑回归满足业务对原因分析的需求，并配套标准化算子。
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

class TotalChargesCleaner(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="TotalChargesCleaner",
        kind=OperatorKind.IMPUTER,
        version=2,
        origin="synthesized",
        description="将字符串类型的数值列转换为浮点数，处理空白字符并使用0填补缺失值",
        conditions=["数据包含字符串类型数值列", "存在因新用户产生的空白字符缺失值"],
        dependencies=["pandas", "numpy"],
        params={"col": "total_charges"},
        parent="TotalChargesCleaner@1"
    )

    def __init__(self, col="total_charges"):
        self.col = col

    def fit(self, X, y=None):
        # 无需学习状态，记录已完成 fit 的标记
        self.fitted_ = True
        return self

    def transform(self, X):
        # 确保输入为 DataFrame 以使用 pandas 处理逻辑
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_out = X.copy()

        if self.col in X_out.columns:
            # 1. 强制转换为字符串以处理潜在的混合类型
            # 2. strip() 去除首尾空格，解决 ' ' 导致的转换失败
            # 3. pd.to_numeric(errors='coerce') 将无法转换的值转为 NaN
            # 4. fillna(0.0) 填充缺失值（针对新入网客户无消费记录的场景）
            processed_col = pd.to_numeric(
                X_out[self.col].astype(str).str.strip(), 
                errors="coerce"
            ).fillna(0.0)

            # 确保结果没有无穷值
            X_out[self.col] = processed_col.replace([np.inf, -np.inf], 0.0)

        return X_out


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_imputer", TotalChargesCleaner()),  # 前置清洗逻辑，保证数据类型正确。
        ("1_imputer", MissingValueImputer()),  # 填充常规缺失值。
        ("2_encoder", OneHotCategoricalEncoder(max_categories=20)),  # 对类别特征编码，为线性模型准备输入。
        ("3_scaler", StandardFeatureScaler()),  # 根据 Lesson:74e2da10，逻辑回归必须在标准化特征后训练，以保证系数可解释性。
        ("4_model", LogisticRegressionModel(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)),  # 逻辑回归提供高可解释性，通过正则化强度调整和样本权重平衡，兼顾模型稳定性和召回要求。
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

各阶段耗时：PLAN 12.5s，BUILD 3.1s，VERIFY 27.6s，ROUTE 0.0s，CURATE 0.0s

```
[PLAN] 场景 telecom_churn；画像标签 ['has_missing_values', 'numeric_features', 'has_categorical_features', 'low_cardinality', 'string_typed_numeric', 'id_columns', 'imbalanced', 'time_tenure', 'mixed_features']
[PLAN] TaskSpec：binary_classification，目标 churn，主指标 recall >= 0.75
[PLAN] 候选方案：['random_forest_pipeline(*TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → RandomForestModel)', 'interpretable_logreg_pipeline(*TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel)']
[PLAN] 完成，用时 12.5s → BUILD
[BUILD] 生成算子 TotalChargesCleaner@2（参考 TotalChargesCleaner@1）：将字符串类型的数值列转换为浮点数，处理空白字符并填补缺失值
[BUILD] 完成，用时 3.1s → VERIFY
[VERIFY] random_forest_pipeline：未通过，recall=0.5939；M_THRESHOLD recall = 0.5939 低于阈值 0.75；M_THRESHOLD roc_auc = 0.7871 低于阈值 0.8
[VERIFY] interpretable_logreg_pipeline：通过，recall=0.7773
[VERIFY] 完成，用时 27.6s → ROUTE
[ROUTE] 完成，用时 0.0s → CURATE
[CURATE] 写入 KG：2 条 ValidationRun；注册新算子 ['TotalChargesCleaner@2']；新增经验 0 条
[CURATE] 完成，用时 0.0s → DONE
```

### LLM 调用来源

| 阶段 | 请求哈希 | 来源 | 本次耗时(s) | 录制耗时(s) | tokens |
| --- | --- | --- | --- | --- | --- |
| plan_spec | 1f94067009b6443c | replay | 0.000 | - | 未提供 |
| plan | f299630329848e9c | remote | 6.446 | - | 10681 |
| plan | fd740dcc56d43c75 | remote | 5.849 | - | 12075 |
| synthesize | ace14bde0d115e71 | remote | 3.096 | - | 2573 |
