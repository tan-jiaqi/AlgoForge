# AlgoForge 验证报告

| 项目 | 内容 |
| --- | --- |
| 需求 | 预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户，召回率不低于0.75 |
| 数据 | data/churn.csv |
| 运行 ID | repair_injected_1_reuse |
| LLM | gemini-3.5-flash |
| 状态 | ✅ 成功 |
| 最优方案 | interpretable_logreg_pipeline |
| 总耗时 | 63.6 s |

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
- **数值清理算子必须在内部显式处理 NaN 和 Inf**（E_NAN）：在数据清理或转换逻辑的末尾，使用 .replace([np.inf, -np.inf], np.nan).fillna(value) 显式清理掉所有非法数值，确保输出数据的有效性。

### 知识引用检查

检索并传入上下文不等于采纳。以下引用由模型声明，程序仅核对可见范围与结构化适用对象；不证明语义落实或指标收益。

| 方案 | 步骤 | 模型引用 | 范围检查 | 语义落实 |
| --- | --- | --- | --- | --- |
| （被拒绝的规划） | - | 无 | 未通过：robust_lgbm_pipeline: 数据含类别列，模型之前缺少 encoder 步骤（如 OneHotCategoricalEncoder）；feature_engineered_rf_pipeline: 数据含类别列，模型之前缺少 encoder 步骤（如 OneHotCategoricalEncoder） | 未验证 |
| interpretable_logreg_pipeline | 1 | Operator:IdColumnDropper@1 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 2 | Operator:TotalChargesCleaner@2 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 3 | Operator:MissingValueImputer@1 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 4 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 5 | Operator:StandardFeatureScaler@1 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 6 | Operator:LogisticRegressionModel@1 | 通过 | 未验证 |
| robust_lgbm_pipeline | 1 | Operator:IdColumnDropper@1 | 通过 | 未验证 |
| robust_lgbm_pipeline | 2 | Operator:TotalChargesCleaner@2 | 通过 | 未验证 |
| robust_lgbm_pipeline | 3 | Operator:MissingValueImputer@1 | 通过 | 未验证 |
| robust_lgbm_pipeline | 4 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| robust_lgbm_pipeline | 5 | Operator:LightGBMModel@1 | 通过 | 未验证 |
| feature_engineered_rf_pipeline | 1 | Operator:IdColumnDropper@1 | 通过 | 未验证 |
| feature_engineered_rf_pipeline | 2 | Operator:ChargeRatioFeatureGenerator@1 | 通过 | 未验证 |
| feature_engineered_rf_pipeline | 3 | Operator:MissingValueImputer@1 | 通过 | 未验证 |
| feature_engineered_rf_pipeline | 4 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| feature_engineered_rf_pipeline | 5 | Operator:RandomForestModel@1 | 通过 | 未验证 |

## 3. 候选方案比较

排序规则：先看是否通过全部检查，再比主指标，再比次要指标之和，最后训练耗时短者优先。指标为多随机种子 × 分层 K 折交叉验证的均值 ± 种子间标准差。

| 排名 | 方案 | 结果 | recall | roc_auc | precision | f1 | recall_at_20 | 单次训练 (s) | 峰值内存 (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ★ interpretable_logreg_pipeline | 通过 | 0.778 ± 0.001 | 0.813 ± 0.001 | 0.481 ± 0.002 | 0.595 ± 0.001 | 0.456 ± 0.002 | 0.084 | 251 |
| 2 | feature_engineered_rf_pipeline | 未通过 | 0.719 ± 0.001 | 0.803 ± 0.001 | 0.492 ± 0.003 | 0.584 ± 0.002 | 0.451 ± 0.004 | 1.013 | 278 |
| 3 | robust_lgbm_pipeline | 未通过 | 0.657 ± 0.009 | 0.786 ± 0.003 | 0.491 ± 0.001 | 0.561 ± 0.004 | 0.423 ± 0.007 | 0.352 | 266 |

```
recall（| 为阈值 0.75）
interpretable_logreg_pipeline        ■■■■■■■■■■■■■■■|···· 0.778
feature_engineered_rf_pipeline       ■■■■■■■■■■■■■■·|···· 0.719
robust_lgbm_pipeline                 ■■■■■■■■■■■■■··|···· 0.657
```

### 各方案的结构与诊断

- `interpretable_logreg_pipeline`：生成:IdColumnDropper → TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel。诊断：无
- `feature_engineered_rf_pipeline`：生成:IdColumnDropper → 生成:ChargeRatioFeatureGenerator → MissingValueImputer → OneHotCategoricalEncoder → RandomForestModel。诊断：`M_THRESHOLD` recall = 0.7191 低于阈值 0.75
- `robust_lgbm_pipeline`：生成:IdColumnDropper → TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → LightGBMModel。诊断：`M_THRESHOLD` recall = 0.6569 低于阈值 0.75

## 4. 最优方案：interpretable_logreg_pipeline

采用线性模型，通过显式处理标识列、缺失值及标准化，满足可解释性且利用平衡权重提升召回率。

| # | 步骤 | 算子 | 参数 | 选择理由 |
| --- | --- | --- | --- | --- |
| 1 | feature | 生成 IdColumnDropper | - | 根据 Lesson:88299a38，删除标识列避免维度过高。 |
| 2 | imputer | TotalChargesCleaner@2 | col=total_charges, tenure_col=tenure | 利用经过验证的算子处理总费用的缺失和类型转换问题。 |
| 3 | imputer | MissingValueImputer@1 | - | 对其他列进行通用缺失值填补。 |
| 4 | encoder | OneHotCategoricalEncoder@1 | max_categories=20 | 对类别特征进行 One-Hot 编码。 |
| 5 | scaler | StandardFeatureScaler@1 | - | 根据 Lesson:74e2da10，逻辑回归必须先标准化。 |
| 6 | model | LogisticRegressionModel@1 | C=1.0, class_weight=balanced | 逻辑回归具备高可解释性，且使用 balanced 权重应对样本不平衡。 |

## 5. 能力生成与修复

| 生成算子 | 类型 | 用途 | 参考的历史实现 | 修复轮数 | 注册到能力库 |
| --- | --- | --- | --- | --- | --- |
| IdColumnDropper@2 | feature | 删除客户ID列以防止编码维度爆炸 | IdColumnDropper@1 | 0 | 是 |
| ChargeRatioFeatureGenerator@2 | feature | 生成消费比值特征以捕捉行为特征 | ChargeRatioFeatureGenerator@1 | 0 | 否 |

## 6. 验证机制

| 级别 | 检查内容 | 本次配置 |
| --- | --- | --- |
| 1 静态检查 | 语法、import 白名单、禁止危险调用、接口与 meta 一致性 | 仅针对生成的算子 |
| 2 沙箱执行 | 独立子进程，限制超时与内存，清除含 KEY/TOKEN 的环境变量 | 超时 180 s，内存 8192 MB |
| 3 契约检查 | 不修改输入、行数一致、不引入 NaN/Inf、概率合法、能处理未见类别与缺失值 | 冒烟样本 300 行 |
| 4 指标检查 | 阈值、种子间稳定性、疑似泄漏、耗时与内存 | 5 折 × 种子 [42, 7, 2024]；std 上限 0.03 |

最优方案资源消耗：单次训练 0.084 s，预测 42.0 ms / 千行，峰值内存 251 MB。

## 7. 知识沉淀

- 回写验证记录 ValidationRun：3 条
- 注册到能力库的新算子：IdColumnDropper@2
- 新提炼的经验：无
- 本次提供的经验：5 条（累计检索命中，不代表采纳）

## 附录 A. 生成的算法代码（pipeline.py）

```python
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

```

## 附录 B. 运行记录

各阶段耗时：PLAN 13.3s，BUILD 15.3s，VERIFY 34.9s，ROUTE 0.0s，CURATE 0.0s

```
[PLAN] 场景 telecom_churn；画像标签 ['has_missing_values', 'numeric_features', 'has_categorical_features', 'low_cardinality', 'string_typed_numeric', 'id_columns', 'imbalanced', 'time_tenure', 'mixed_features']
[PLAN] TaskSpec：binary_classification，目标 churn，主指标 recall >= 0.75
[PLAN] 候选方案：['interpretable_logreg_pipeline(*IdColumnDropper → TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel)', 'robust_lgbm_pipeline(*IdColumnDropper → TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → LightGBMModel)', 'feature_engineered_rf_pipeline(*IdColumnDropper → *ChargeRatioFeatureGenerator → MissingValueImputer → OneHotCategoricalEncoder → RandomForestModel)']
[PLAN] 完成，用时 13.3s → BUILD
[BUILD] 生成算子 IdColumnDropper@2（参考 IdColumnDropper@1）：删除客户ID列以防止编码维度爆炸
[BUILD] 生成算子 ChargeRatioFeatureGenerator@2（参考 ChargeRatioFeatureGenerator@1）：生成消费比值特征以捕捉行为特征
[BUILD] 完成，用时 15.3s → VERIFY
[VERIFY] interpretable_logreg_pipeline：通过，recall=0.7780
[VERIFY] robust_lgbm_pipeline：未通过，recall=0.6569；M_THRESHOLD recall = 0.6569 低于阈值 0.75
[VERIFY] feature_engineered_rf_pipeline：未通过，recall=0.7191；M_THRESHOLD recall = 0.7191 低于阈值 0.75
[VERIFY] 完成，用时 34.9s → ROUTE
[ROUTE] 完成，用时 0.0s → CURATE
[CURATE] 写入 KG：3 条 ValidationRun；注册新算子 ['IdColumnDropper@2']；新增经验 0 条
[CURATE] 完成，用时 0.0s → DONE
```

### LLM 调用来源

| 阶段 | 请求哈希 | 来源 | 本次耗时(s) | 录制耗时(s) | tokens |
| --- | --- | --- | --- | --- | --- |
| plan | 3a5d9160aab50c0f | remote | 5.634 | - | 12524 |
| plan | 1265cec049fe2338 | remote | 7.520 | - | 14673 |
| synthesize | 571e4464773ac72e | remote | 9.291 | - | 2431 |
| synthesize | 9332f73277f00f92 | remote | 6.035 | - | 2817 |
