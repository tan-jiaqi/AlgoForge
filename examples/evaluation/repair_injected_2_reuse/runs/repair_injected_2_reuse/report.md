# AlgoForge 验证报告

| 项目 | 内容 |
| --- | --- |
| 需求 | 预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户，召回率不低于0.75 |
| 数据 | data/churn.csv |
| 运行 ID | repair_injected_2_reuse |
| LLM | gemini-3.5-flash |
| 状态 | ✅ 成功 |
| 最优方案 | interpretable_logreg_pipeline |
| 总耗时 | 43.9 s |

共验证 3 个候选方案，1 个通过；重新规划 0 次，修复 0 轮。最优方案 `interpretable_logreg_pipeline` 的 recall = 0.778（要求 ≥ 0.75）。

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
| imputer | MissingValueImputer@1 | 是 | 0.510 | 0.66 / 0.09 / 0.73 | 匹配条件：has_missing_values；历史验证 4 次，平均 recall 0.732 |
| imputer | TotalChargesCleaner@1 | 否（历史实现，可参考生成） | 0.482 | 1.00 / 0.27 / 0.00 | 匹配条件：has_missing_values、numeric_features、string_typed_numeric |
| imputer | TotalChargesCleaner@2 | 是 | 0.455 | 0.55 / 0.00 / 0.78 | 历史验证 1 次，平均 recall 0.778 |
| feature | TenureBucketizer@1 | 否（历史实现，可参考生成） | 0.646 | 0.81 / 0.27 / 0.80 | 匹配条件：interpretable、numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | ChargeRatioFeatureGenerator@1 | 否（历史实现，可参考生成） | 0.615 | 0.80 / 0.18 / 0.80 | 匹配条件：numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | ServiceCountFeature@1 | 否（历史实现，可参考生成） | 0.427 | 1.00 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| feature | IdColumnDropper@1 | 否（历史实现，可参考生成） | 0.350 | 0.81 / 0.09 / 0.00 | 匹配条件：id_columns |
| encoder | OneHotCategoricalEncoder@1 | 是 | 0.645 | 0.93 / 0.18 / 0.73 | 匹配条件：has_categorical_features、low_cardinality；历史验证 4 次，平均 recall 0.732 |
| encoder | KFoldTargetEncoder@1 | 否（历史实现，可参考生成） | 0.427 | 1.00 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| scaler | StandardFeatureScaler@1 | 是 | 0.650 | 1.00 / 0.09 / 0.74 | 匹配条件：numeric_features；历史验证 2 次，平均 recall 0.744 |
| model | BalancedLightGBMClassifier@1 | 否（历史实现，可参考生成） | 0.661 | 1.00 / 0.09 / 0.78 | 匹配条件：imbalanced；历史验证 1 次，平均 recall 0.780 |
| model | LogisticRegressionModel@1 | 是 | 0.637 | 0.97 / 0.09 / 0.74 | 匹配条件：interpretable；历史验证 2 次，平均 recall 0.744 |
| model | RandomForestModel@1 | 是 | 0.596 | 0.93 / 0.09 / 0.66 | 匹配条件：mixed_features；历史验证 1 次，平均 recall 0.660 |
| model | LightGBMModel@1 | 是 | 0.560 | 0.73 / 0.09 / 0.80 | 匹配条件：imbalanced；历史验证 1 次，平均 recall 0.800 |

### 参考的历史经验

- **样本不平衡导致默认模型召回率偏低**（M_THRESHOLD）：启用平衡类权重（如 class_weight='balanced' 或 BalancedLightGBMClassifier）以提升召回率
- **数值列含空白字符导致类型转换失败**（E_RUNTIME）：先 str.strip()，再 pd.to_numeric(errors='coerce')，缺失值填补为 0
- **逻辑回归标准化与可解释性**：必须先进行特征标准化（如 StandardFeatureScaler），在 0.1~10 之间搜索正则化强度 C，可直接输出系数给业务方。
- **未剔除客户 ID 导致 One-Hot 编码后维度暴涨与内存溢出**（E_TIMEOUT）：建模前剔除 ID 等标识列，并在编码器中设置最大类别数
- **LightGBM 调参及不平衡处理**：设置 num_leaves 15~63, learning_rate 0.03~0.1, n_estimators 200~600；不平衡时设置 class_weight='balanced'；树模型无需标准化。

### 知识引用检查

每步引用的知识由 Planner 声明，程序检查引用是否在本轮检索结果中、是否适用于该步骤。

| 方案 | 步骤 | 模型引用 | 范围检查 |
| --- | --- | --- | --- |
| （被拒绝的规划） | - | 无 | 未通过：lgbm_feature_engineering_pipeline: 第 2 步：证据 Operator:ChargeRatioFeatureGenerator@1 不适用于 None；lgbm_feature_engineering_pipeline: 第 2 步：证据 Operator:ServiceCountFeature@1 不适用于 None；random_forest_robust_pipeline: 第 2 步：证据 Operator:TenureBucketizer@1 不适用于 None |
| interpretable_logreg_pipeline | 1 | Operator:TotalChargesCleaner@2 | 通过 |
| interpretable_logreg_pipeline | 2 | Operator:MissingValueImputer@1 | 通过 |
| interpretable_logreg_pipeline | 3 | Operator:OneHotCategoricalEncoder@1 | 通过 |
| interpretable_logreg_pipeline | 4 | Operator:StandardFeatureScaler@1、Lesson:74e2da10 | 通过 |
| interpretable_logreg_pipeline | 5 | Operator:LogisticRegressionModel@1、Lesson:9e9e77d6 | 通过 |
| lgbm_optimized_pipeline | 1 | Operator:TotalChargesCleaner@2 | 通过 |
| lgbm_optimized_pipeline | 2 | Operator:OneHotCategoricalEncoder@1 | 通过 |
| lgbm_optimized_pipeline | 3 | Operator:LightGBMModel@1、Lesson:f2c1398b | 通过 |
| balanced_rf_pipeline | 1 | Operator:MissingValueImputer@1 | 通过 |
| balanced_rf_pipeline | 2 | Operator:OneHotCategoricalEncoder@1 | 通过 |
| balanced_rf_pipeline | 3 | Operator:RandomForestModel@1、Lesson:9e9e77d6 | 通过 |

## 3. 候选方案比较

排序规则：先看是否通过全部检查，再比主指标，再比次要指标之和，最后训练耗时短者优先。指标为多随机种子 × 分层 K 折交叉验证的均值 ± 种子间标准差。

| 排名 | 方案 | 结果 | recall | roc_auc | precision | f1 | recall_at_20 | 单次训练 (s) | 峰值内存 (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ★ interpretable_logreg_pipeline | 通过 | 0.778 ± 0.001 | 0.813 ± 0.001 | 0.481 ± 0.001 | 0.595 ± 0.001 | 0.454 ± 0.002 | 0.096 | 259 |
| 2 | lgbm_optimized_pipeline | 未通过 | 0.627 ± 0.007 | 0.781 ± 0.002 | 0.493 ± 0.003 | 0.552 ± 0.005 | 0.414 ± 0.004 | 0.301 | 259 |
| 3 | balanced_rf_pipeline | 未通过 | 0.582 ± 0.007 | 0.781 ± 0.003 | 0.509 ± 0.006 | 0.543 ± 0.007 | 0.423 ± 0.003 | 1.035 | 352 |

```
recall（| 为阈值 0.75）
interpretable_logreg_pipeline        ■■■■■■■■■■■■■■■|···· 0.778
lgbm_optimized_pipeline              ■■■■■■■■■■■■■··|···· 0.627
balanced_rf_pipeline                 ■■■■■■■■■■■■···|···· 0.582
```

### 各方案的结构与诊断

- `interpretable_logreg_pipeline`：TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel。诊断：无
- `lgbm_optimized_pipeline`：TotalChargesCleaner → OneHotCategoricalEncoder → LightGBMModel。诊断：`M_THRESHOLD` recall = 0.6273 低于阈值 0.75
- `balanced_rf_pipeline`：MissingValueImputer → OneHotCategoricalEncoder → RandomForestModel。诊断：`M_THRESHOLD` recall = 0.5824 低于阈值 0.75

## 4. 最优方案：interpretable_logreg_pipeline

利用逻辑回归确保解释性，并遵循标准化与样本平衡的原则优化召回率。

| # | 步骤 | 算子 | 参数 | 选择理由 |
| --- | --- | --- | --- | --- |
| 1 | imputer | TotalChargesCleaner@2 | - | 处理总费用列的格式及缺失值，这是逻辑回归的先决步骤。 |
| 2 | imputer | MissingValueImputer@1 | - | 补全剩余的缺失值。 |
| 3 | encoder | OneHotCategoricalEncoder@1 | - | 对类别特征进行编码以适应线性模型。 |
| 4 | scaler | StandardFeatureScaler@1 | - | 逻辑回归必须进行标准化以确保模型可解释性。 |
| 5 | model | LogisticRegressionModel@1 | - | 使用逻辑回归提供高解释性，并配置 balanced 权重优化召回率。 |

## 5. 能力生成与修复

本次运行所有步骤均复用能力库中已有算子，未生成新算子。

## 6. 验证机制

| 级别 | 检查内容 | 本次配置 |
| --- | --- | --- |
| 1 静态检查 | 语法、import 白名单、禁止危险调用、接口与 meta 一致性 | 仅针对生成的算子 |
| 2 沙箱执行 | 独立子进程，限制超时与内存，清除含 KEY/TOKEN 的环境变量 | 超时 180 s，内存 8192 MB |
| 3 契约检查 | 不修改输入、行数一致、不引入 NaN/Inf、概率合法、能处理未见类别与缺失值 | 冒烟样本 300 行 |
| 4 指标检查 | 阈值、种子间稳定性、疑似泄漏、耗时与内存 | 5 折 × 种子 [42, 7, 2024]；std 上限 0.03 |

最优方案资源消耗：单次训练 0.096 s，预测 45.5 ms / 千行，峰值内存 259 MB。

## 7. 知识沉淀

- 回写验证记录 ValidationRun：3 条
- 注册到能力库的新算子：无
- 新提炼的经验：无
- 本次检索使用的经验：5 条（命中次数已累计）

## 附录 A. 生成的算法代码（pipeline.py）

```python
"""预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户，召回率不低于0.75 —— 自动生成的算法流水线

需求：预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户，召回率不低于0.75
方案：interpretable_logreg_pipeline
设计依据：利用逻辑回归确保解释性，并遵循标准化与样本平衡的原则优化召回率。
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
        self.mean_total_charges_ = 0.0
        return self

    def transform(self, X):
        X = X.copy()

        # 处理 total_charges 列
        if self.col in X.columns:
            # 转换为数值，将无法转换的转为 NaN，然后填充为 0
            X[self.col] = pd.to_numeric(X[self.col].astype(str).str.strip(), errors="coerce")
            X[self.col] = X[self.col].fillna(0.0)
            # 处理无穷大
            X[self.col] = X[self.col].replace([np.inf, -np.inf], 0.0)

        # 处理 tenure 分桶
        if self.tenure_col in X.columns:
            # 确保 tenure 是数值，缺失值处理为 0
            tenure_series = pd.to_numeric(X[self.tenure_col], errors="coerce").fillna(0.0)
            bins = [0, 6, 12, 24, 48, 60, 1000]
            labels = ["0-6m", "7-12m", "13-24m", "25-48m", "49-60m", "60m+"]
            X["tenure_binned"] = pd.cut(
                tenure_series, 
                bins=bins, 
                labels=labels, 
                include_lowest=True
            ).astype(str)
            # 填充处理可能出现的 NaN
            X["tenure_binned"] = X["tenure_binned"].fillna("unknown")

        return X


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_imputer", TotalChargesCleaner(col='total_charges', tenure_col='tenure')),  # 处理总费用列的格式及缺失值，这是逻辑回归的先决步骤。
        ("1_imputer", MissingValueImputer()),  # 补全剩余的缺失值。
        ("2_encoder", OneHotCategoricalEncoder(max_categories=20)),  # 对类别特征进行编码以适应线性模型。
        ("3_scaler", StandardFeatureScaler()),  # 逻辑回归必须进行标准化以确保模型可解释性。
        ("4_model", LogisticRegressionModel(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)),  # 使用逻辑回归提供高解释性，并配置 balanced 权重优化召回率。
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

各阶段耗时：PLAN 10.2s，BUILD 0.0s，VERIFY 33.6s，ROUTE 0.0s，CURATE 0.0s

```
[PLAN] 场景 telecom_churn；画像标签 ['has_missing_values', 'numeric_features', 'has_categorical_features', 'low_cardinality', 'string_typed_numeric', 'id_columns', 'imbalanced', 'time_tenure', 'mixed_features']
[PLAN] TaskSpec：binary_classification，目标 churn，主指标 recall >= 0.75
[PLAN] 候选方案：['interpretable_logreg_pipeline(TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel)', 'lgbm_optimized_pipeline(TotalChargesCleaner → OneHotCategoricalEncoder → LightGBMModel)', 'balanced_rf_pipeline(MissingValueImputer → OneHotCategoricalEncoder → RandomForestModel)']
[PLAN] 完成，用时 10.2s → BUILD
[BUILD] 完成，用时 0.0s → VERIFY
[VERIFY] interpretable_logreg_pipeline：通过，recall=0.7785
[VERIFY] lgbm_optimized_pipeline：未通过，recall=0.6273；M_THRESHOLD recall = 0.6273 低于阈值 0.75
[VERIFY] balanced_rf_pipeline：未通过，recall=0.5824；M_THRESHOLD recall = 0.5824 低于阈值 0.75
[VERIFY] 完成，用时 33.6s → ROUTE
[ROUTE] 完成，用时 0.0s → CURATE
[CURATE] 写入 KG：3 条 ValidationRun；注册新算子 []；新增经验 0 条
[CURATE] 完成，用时 0.0s → DONE
```

### LLM 调用来源

| 阶段 | 请求哈希 | 来源 | 本次耗时(s) | 录制耗时(s) | tokens |
| --- | --- | --- | --- | --- | --- |
| plan | fb4bf4616f7173f9 | remote | 6.373 | - | 12345 |
| plan | 916c39c69b4a0813 | remote | 3.686 | - | 13735 |
