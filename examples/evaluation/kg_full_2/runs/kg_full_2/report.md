# AlgoForge 验证报告

| 项目 | 内容 |
| --- | --- |
| 需求 | 预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75 |
| 数据 | data/churn.csv |
| 运行 ID | kg_full_2 |
| LLM | gemini-3.5-flash |
| 状态 | ❌ 未通过 |
| 最优方案 | lightgbm_churn_prediction |
| 总耗时 | 20.8 s |

共验证 1 个候选方案，0 个通过；重新规划 0 次，修复 0 轮。最优方案 `lightgbm_churn_prediction` 的 recall = 0.603（要求 ≥ 0.75）。

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

### 参考的历史经验

- **样本不平衡导致默认模型召回率偏低**（M_THRESHOLD）：启用平衡类权重（如 class_weight='balanced' 或 BalancedLightGBMClassifier）以提升召回率
- **随机森林防过拟合与召回优化**：设置 n_estimators 300 左右，限制 max_depth 在 8~16 防止过拟合；召回优先时需配合调整分类阈值。
- **逻辑回归标准化与可解释性**：必须先进行特征标准化（如 StandardFeatureScaler），在 0.1~10 之间搜索正则化强度 C，可直接输出系数给业务方。
- **数值列含空白字符导致类型转换失败**（E_RUNTIME）：先 str.strip()，再 pd.to_numeric(errors='coerce')，缺失值填补为 0
- **未剔除客户 ID 导致 One-Hot 编码后维度暴涨与内存溢出**（E_TIMEOUT）：建模前剔除 ID 等标识列，并在编码器中设置最大类别数

### 知识引用检查

检索并传入上下文不等于采纳。以下引用由模型声明，程序仅核对可见范围与结构化适用对象；不证明语义落实或指标收益。

| 方案 | 步骤 | 模型引用 | 范围检查 | 语义落实 |
| --- | --- | --- | --- | --- |
| lightgbm_churn_prediction | 1 | Operator:TotalChargesCleaner@1 | 通过 | 未验证 |
| lightgbm_churn_prediction | 2 | Operator:MissingValueImputer@1 | 通过 | 未验证 |
| lightgbm_churn_prediction | 3 | Operator:IdColumnDropper@1 | 通过 | 未验证 |
| lightgbm_churn_prediction | 4 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| lightgbm_churn_prediction | 5 | Operator:LightGBMModel@1、ValidationRun:f6e4284c | 通过 | 未验证 |

## 3. 候选方案比较

排序规则：先看是否通过全部检查，再比主指标，再比次要指标之和，最后训练耗时短者优先。指标为多随机种子 × 分层 K 折交叉验证的均值 ± 种子间标准差。

| 排名 | 方案 | 结果 | recall | roc_auc | precision | f1 | recall_at_20 | 单次训练 (s) | 峰值内存 (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ★ lightgbm_churn_prediction | 未通过 | 0.603 ± 0.007 | 0.780 ± 0.003 | 0.494 ± 0.002 | 0.543 ± 0.002 | 0.418 ± 0.009 | 0.430 | 267 |

```
recall（| 为阈值 0.75）
lightgbm_churn_prediction            ■■■■■■■■■■■■···|···· 0.603
```

### 各方案的结构与诊断

- `lightgbm_churn_prediction`：生成:TotalChargesCleaner → MissingValueImputer → 生成:IdColumnDropper → OneHotCategoricalEncoder → LightGBMModel。诊断：`M_THRESHOLD` recall = 0.6030 低于阈值 0.75

## 4. 最优方案：lightgbm_churn_prediction

该方案针对电信流失场景，优先解决脏数据和ID列导致的内存及转换问题，并采用经过验证的LightGBM模型以达到召回率阈值。

| # | 步骤 | 算子 | 参数 | 选择理由 |
| --- | --- | --- | --- | --- |
| 1 | imputer | 生成 TotalChargesCleaner | - | 根据 Lesson:69d51358，需先处理 total_charges 列的脏数据，防止类型转换错误。 |
| 2 | imputer | MissingValueImputer@1 | - | 处理 monthly_charges 等其他列的缺失值。 |
| 3 | feature | 生成 IdColumnDropper | - | 根据 Lesson:88299a38，建模前必须剔除 ID 列。 |
| 4 | encoder | OneHotCategoricalEncoder@1 | max_categories=10 | 对类别特征进行编码，限制类别数量以控制特征维度。 |
| 5 | model | LightGBMModel@1 | class_weight=balanced, n_estimators=500, learning_rate=0.05 | 使用 LightGBM 并配合 class_weight='balanced' 优化召回率，满足召回优先的业务目标，参考 Lesson:9e9e77d6。 |

## 5. 能力生成与修复

| 生成算子 | 类型 | 用途 | 参考的历史实现 | 修复轮数 | 注册到能力库 |
| --- | --- | --- | --- | --- | --- |
| TotalChargesCleaner@2 | imputer | 将字符串类型的数值列转换为浮点型，处理空白值并填充缺失值 | TotalChargesCleaner@1 | 0 | 否 |
| IdColumnDropper@2 | feature | 剔除客户 ID 列以防止维度爆炸和过拟合 | IdColumnDropper@1 | 0 | 否 |

## 6. 验证机制

| 级别 | 检查内容 | 本次配置 |
| --- | --- | --- |
| 1 静态检查 | 语法、import 白名单、禁止危险调用、接口与 meta 一致性 | 仅针对生成的算子 |
| 2 沙箱执行 | 独立子进程，限制超时与内存，清除含 KEY/TOKEN 的环境变量 | 超时 180 s，内存 8192 MB |
| 3 契约检查 | 不修改输入、行数一致、不引入 NaN/Inf、概率合法、能处理未见类别与缺失值 | 冒烟样本 300 行 |
| 4 指标检查 | 阈值、种子间稳定性、疑似泄漏、耗时与内存 | 5 折 × 种子 [42, 7, 2024]；std 上限 0.03 |

最优方案资源消耗：单次训练 0.430 s，预测 53.1 ms / 千行，峰值内存 267 MB。

## 7. 知识沉淀

- 回写验证记录 ValidationRun：1 条
- 注册到能力库的新算子：无
- 新提炼的经验：无
- 本次提供的经验：6 条（累计检索命中，不代表采纳）

## 附录 A. 生成的算法代码（pipeline.py）

```python
# 警告：该方案未通过验证，仅为得分最高的候选，供人工参考。
"""预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75 —— 自动生成的算法流水线

需求：预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75
方案：lightgbm_churn_prediction
设计依据：该方案针对电信流失场景，优先解决脏数据和ID列导致的内存及转换问题，并采用经过验证的LightGBM模型以达到召回率阈值。
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
        description="将字符串类型的 total_charges 列转换为浮点型，处理空白值并填充缺失值",
        conditions=["包含 total_charges 字符串列"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"col": "total_charges"},
        parent="TotalChargesCleaner@1"
    )

    def __init__(self, col="total_charges"):
        self.col = col

    def fit(self, X, y=None):
        """记录列名并返回 self，该算子无状态学习。"""
        self.col_ = self.col
        return self

    def transform(self, X):
        """
        处理逻辑：
        1. 复制输入数据防止污染原数据
        2. 将指定列转为字符串并去除首尾空白
        3. 使用 pd.to_numeric 将无法解析的值转为 NaN
        4. 使用 0.0 填充转换后的 NaN（针对新客户逻辑）
        5. 处理可能存在的无穷值并填充为 0.0
        """
        X = X.copy()
        if self.col_ in X.columns:
            # 1. 清洗及类型转换
            clean_series = pd.to_numeric(
                X[self.col_].astype(str).str.strip(), 
                errors="coerce"
            )

            # 2. 填充缺失值
            clean_series = clean_series.fillna(0.0)

            # 3. 处理可能产生的无穷值
            clean_series = clean_series.replace([np.inf, -np.inf], 0.0)

            X[self.col_] = clean_series

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
        description="剔除客户 ID 等高基数标识列，防止维度爆炸和过拟合",
        conditions=["数据包含高基数标识列"],
        dependencies=[],
        params={"id_cols": ["customer_id"]},
        parent="IdColumnDropper@1"
    )

    def __init__(self, id_cols=["customer_id"]):
        self.id_cols = id_cols

    def fit(self, X, y=None):
        """记录输入列以供校验，此处仅返回self"""
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):
        """
        剔除指定的 ID 列。
        根据踩坑经验：必须先 copy 原始数据，防止原地修改影响下游；
        处理时需确保即使列不存在也不报错。
        """
        X = X.copy()
        cols_to_drop = [c for c in self.id_cols if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
        return X


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_imputer", TotalChargesCleaner()),  # 根据 Lesson:69d51358，需先处理 total_charges 列的脏数据，防止类型转换错误。
        ("1_imputer", MissingValueImputer()),  # 处理 monthly_charges 等其他列的缺失值。
        ("2_feature", IdColumnDropper()),  # 根据 Lesson:88299a38，建模前必须剔除 ID 列。
        ("3_encoder", OneHotCategoricalEncoder(max_categories=10)),  # 对类别特征进行编码，限制类别数量以控制特征维度。
        ("4_model", LightGBMModel(n_estimators=500, learning_rate=0.05, num_leaves=31, class_weight='balanced', random_state=42)),  # 使用 LightGBM 并配合 class_weight='balanced' 优化召回率，满足召回优先的业务目标，参考 Lesson:9e9e77d6。
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

各阶段耗时：PLAN 6.5s，BUILD 5.2s，VERIFY 9.2s，ROUTE 0.0s，CURATE 0.0s

```
[PLAN] 场景 telecom_churn；画像标签 ['has_missing_values', 'numeric_features', 'has_categorical_features', 'low_cardinality', 'string_typed_numeric', 'id_columns', 'imbalanced', 'time_tenure', 'mixed_features']
[PLAN] TaskSpec：binary_classification，目标 churn，主指标 recall >= 0.75
[PLAN] 候选方案：['lightgbm_churn_prediction(*TotalChargesCleaner → MissingValueImputer → *IdColumnDropper → OneHotCategoricalEncoder → LightGBMModel)']
[PLAN] 完成，用时 6.5s → BUILD
[BUILD] 生成算子 TotalChargesCleaner@2（参考 TotalChargesCleaner@1）：将字符串类型的数值列转换为浮点型，处理空白值并填充缺失值
[BUILD] 生成算子 IdColumnDropper@2（参考 IdColumnDropper@1）：剔除客户 ID 列以防止维度爆炸和过拟合
[BUILD] 完成，用时 5.2s → VERIFY
[VERIFY] lightgbm_churn_prediction：未通过，recall=0.6030；M_THRESHOLD recall = 0.6030 低于阈值 0.75
[VERIFY] 完成，用时 9.2s → ROUTE
[ROUTE] 没有方案通过验证
[ROUTE] 完成，用时 0.0s → CURATE
[CURATE] 写入 KG：1 条 ValidationRun；注册新算子 []；新增经验 0 条
[CURATE] 完成，用时 0.0s → FAILED
```

### LLM 调用来源

| 阶段 | 请求哈希 | 来源 | 本次耗时(s) | 录制耗时(s) | tokens |
| --- | --- | --- | --- | --- | --- |
| plan | 969ce11249be4302 | remote | 6.297 | - | 10434 |
| synthesize | 9e338c307c6d6160 | remote | 2.725 | - | 2568 |
| synthesize | 8cad6f33bbf0452d | remote | 2.427 | - | 2311 |
