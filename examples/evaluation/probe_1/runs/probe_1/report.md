# AlgoForge 验证报告

| 项目 | 内容 |
| --- | --- |
| 需求 | 预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75；将 total_charges 的空白字符串转换并填补为合理数值。 |
| 数据 | data/churn.csv |
| 运行 ID | probe_1 |
| LLM | gemini-3.5-flash |
| 状态 | ❌ 未通过 |
| 最优方案 | random_forest_pipeline |
| 总耗时 | 106.6 s |

共验证 3 个候选方案，0 个通过；重新规划 0 次，修复 1 轮。最优方案 `random_forest_pipeline` 的 recall = 0.677（要求 ≥ 0.75）。

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
| imputer | MissingValueImputer@1 | 是 | 0.491 | 0.62 / 0.09 / 0.72 | 匹配条件：has_missing_values；历史验证 3 次，平均 recall 0.717 |
| imputer | TotalChargesCleaner@1 | 否（历史实现，可参考生成） | 0.482 | 1.00 / 0.27 / 0.00 | 匹配条件：has_missing_values、numeric_features、string_typed_numeric |
| feature | TenureBucketizer@1 | 否（历史实现，可参考生成） | 0.722 | 1.00 / 0.27 / 0.80 | 匹配条件：interpretable、numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | ChargeRatioFeatureGenerator@1 | 否（历史实现，可参考生成） | 0.617 | 0.81 / 0.18 / 0.80 | 匹配条件：numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | ServiceCountFeature@1 | 否（历史实现，可参考生成） | 0.416 | 0.97 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| feature | IdColumnDropper@1 | 否（历史实现，可参考生成） | 0.385 | 0.89 / 0.09 / 0.00 | 匹配条件：id_columns |
| encoder | OneHotCategoricalEncoder@1 | 是 | 0.613 | 0.86 / 0.18 / 0.72 | 匹配条件：has_categorical_features、low_cardinality；历史验证 3 次，平均 recall 0.717 |
| encoder | KFoldTargetEncoder@1 | 否（历史实现，可参考生成） | 0.427 | 1.00 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| scaler | StandardFeatureScaler@1 | 是 | 0.640 | 1.00 / 0.09 / 0.71 | 匹配条件：numeric_features；历史验证 1 次，平均 recall 0.710 |
| model | LogisticRegressionModel@1 | 是 | 0.640 | 1.00 / 0.09 / 0.71 | 匹配条件：interpretable；历史验证 1 次，平均 recall 0.710 |
| model | BalancedLightGBMClassifier@1 | 否（历史实现，可参考生成） | 0.601 | 0.85 / 0.09 / 0.78 | 匹配条件：imbalanced；历史验证 1 次，平均 recall 0.780 |
| model | RandomForestModel@1 | 是 | 0.556 | 0.83 / 0.09 / 0.66 | 匹配条件：mixed_features；历史验证 1 次，平均 recall 0.660 |
| model | LightGBMModel@1 | 是 | 0.546 | 0.70 / 0.09 / 0.80 | 匹配条件：imbalanced；历史验证 1 次，平均 recall 0.800 |

### 参考的历史经验

- **数值列含空白字符导致类型转换失败**（E_RUNTIME）：先 str.strip()，再 pd.to_numeric(errors='coerce')，缺失值填补为 0
- **样本不平衡导致默认模型召回率偏低**（M_THRESHOLD）：启用平衡类权重（如 class_weight='balanced' 或 BalancedLightGBMClassifier）以提升召回率
- **随机森林防过拟合与召回优化**：设置 n_estimators 300 左右，限制 max_depth 在 8~16 防止过拟合；召回优先时需配合调整分类阈值。
- **逻辑回归标准化与可解释性**：必须先进行特征标准化（如 StandardFeatureScaler），在 0.1~10 之间搜索正则化强度 C，可直接输出系数给业务方。
- **未剔除客户 ID 导致 One-Hot 编码后维度暴涨与内存溢出**（E_TIMEOUT）：建模前剔除 ID 等标识列，并在编码器中设置最大类别数

### 知识引用检查

检索并传入上下文不等于采纳。以下引用由模型声明，程序仅核对可见范围与结构化适用对象；不证明语义落实或指标收益。

| 方案 | 步骤 | 模型引用 | 范围检查 | 语义落实 |
| --- | --- | --- | --- | --- |
| lgbm_pipeline | 1 | Operator:TotalChargesCleaner@1、Lesson:69d51358 | 通过 | 未验证 |
| lgbm_pipeline | 2 | Operator:MissingValueImputer@1 | 通过 | 未验证 |
| lgbm_pipeline | 3 | Operator:IdColumnDropper@1、Lesson:88299a38 | 通过 | 未验证 |
| lgbm_pipeline | 4 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| lgbm_pipeline | 5 | Operator:LightGBMModel@1、Lesson:9e9e77d6 | 通过 | 未验证 |
| logistic_interpretable_pipeline | 1 | Operator:TotalChargesCleaner@1、Lesson:69d51358 | 通过 | 未验证 |
| logistic_interpretable_pipeline | 2 | Operator:TenureBucketizer@1 | 通过 | 未验证 |
| logistic_interpretable_pipeline | 3 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| logistic_interpretable_pipeline | 4 | Operator:StandardFeatureScaler@1、Lesson:74e2da10 | 通过 | 未验证 |
| logistic_interpretable_pipeline | 5 | Operator:LogisticRegressionModel@1、Lesson:74e2da10、Lesson:9e9e77d6 | 通过 | 未验证 |
| random_forest_pipeline | 1 | Operator:TotalChargesCleaner@1 | 通过 | 未验证 |
| random_forest_pipeline | 2 | Operator:ChargeRatioFeatureGenerator@1 | 通过 | 未验证 |
| random_forest_pipeline | 3 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| random_forest_pipeline | 4 | Operator:RandomForestModel@1、Lesson:57718714 | 通过 | 未验证 |

## 3. 候选方案比较

排序规则：先看是否通过全部检查，再比主指标，再比次要指标之和，最后训练耗时短者优先。指标为多随机种子 × 分层 K 折交叉验证的均值 ± 种子间标准差。

| 排名 | 方案 | 结果 | recall | roc_auc | precision | f1 | recall_at_20 | 单次训练 (s) | 峰值内存 (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ★ random_forest_pipeline | 未通过 | 0.677 ± 0.006 | 0.801 ± 0.002 | 0.501 ± 0.004 | 0.576 ± 0.005 | 0.450 ± 0.002 | 0.901 | 282 |
| 2 | lgbm_pipeline | 未通过 | 0.654 ± 0.009 | 0.786 ± 0.004 | 0.491 ± 0.001 | 0.561 ± 0.004 | 0.426 ± 0.008 | 0.308 | 267 |
| 3 | logistic_interpretable_pipeline | 未通过 | - | - | - | - | - | - | 233 |

```
recall（| 为阈值 0.75）
random_forest_pipeline               ■■■■■■■■■■■■■■·|···· 0.677
lgbm_pipeline                        ■■■■■■■■■■■■■··|···· 0.654
logistic_interpretable_pipeline      ···············|···· -
```

### 各方案的结构与诊断

- `random_forest_pipeline`：生成:TotalChargesCleaner → 生成:ChargeRatioFeatureGenerator → OneHotCategoricalEncoder → RandomForestModel。诊断：`M_THRESHOLD` recall = 0.6775 低于阈值 0.75
- `lgbm_pipeline`：生成:TotalChargesCleaner → MissingValueImputer → 生成:IdColumnDropper → OneHotCategoricalEncoder → LightGBMModel。诊断：`M_THRESHOLD` recall = 0.6544 低于阈值 0.75
- `logistic_interpretable_pipeline`：生成:TotalChargesCleaner → 生成:TenureBucketizer → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel。诊断：`E_RUNTIME` ValueError: Input X contains NaN.
LogisticRegression does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to pr (at _assert_all_finite_element_wise:186)

## 4. 最优方案：random_forest_pipeline

利用随机森林处理复杂的非线性关系，引入消费比值特征以捕捉流失信号。

| # | 步骤 | 算子 | 参数 | 选择理由 |
| --- | --- | --- | --- | --- |
| 1 | imputer | 生成 TotalChargesCleaner | - | 数据清洗第一步。 |
| 2 | feature | 生成 ChargeRatioFeatureGenerator | - | 通过领域知识构造特征提升模型感知能力。 |
| 3 | encoder | OneHotCategoricalEncoder@1 | - | 编码。 |
| 4 | model | RandomForestModel@1 | n_estimators=400, max_depth=12, class_weight=balanced | 利用随机森林非线性拟合能力，通过调整深度控制过拟合。 |

## 5. 能力生成与修复

| 生成算子 | 类型 | 用途 | 参考的历史实现 | 修复轮数 | 注册到能力库 |
| --- | --- | --- | --- | --- | --- |
| TotalChargesCleaner@2 | imputer | 将字符串类型的total_charges转换为数值，处理空白符并填补缺失值 | TotalChargesCleaner@1 | 0 | 否 |
| IdColumnDropper@2 | feature | 删除客户ID等标识列以避免编码后维度爆炸 | IdColumnDropper@1 | 1 | 否 |
| TenureBucketizer@2 | feature | 对tenure进行分箱以提升线性模型可解释性 | TenureBucketizer@1 | 0 | 否 |
| ChargeRatioFeatureGenerator@2 | feature | 生成消费趋势特征 | ChargeRatioFeatureGenerator@1 | 0 | 否 |

### 修复 IdColumnDropper@2 第 1 轮：✅ 已修复

- 诊断：`E_INTERFACE`（static）meta.params 与 __init__ 默认值不一致：["id_cols: meta=['customer_id'], __init__=('customer_id',)"]
- 提示：实例化时 meta.params 会覆盖 __init__ 默认值，两处必须一致
- 参考经验：0 条

```diff
--- 修复前
+++ 修复后
@@ -19,4 +19,4 @@
 
-    def __init__(self, id_cols=("customer_id",)):
-        self.id_cols = list(id_cols)
+    def __init__(self, id_cols=["customer_id"]):
+        self.id_cols = id_cols
 
@@ -33,3 +33,3 @@
         
-        # 2. 处理 total_charges 等因读取产生的脏数据（符合 Lessons 经验）
+        # 2. 处理 total_charges 等因读取产生的脏数据
         # 将空白字符串转换为数字并填补缺失值，防止后续下游编码器崩溃
```

## 6. 验证机制

| 级别 | 检查内容 | 本次配置 |
| --- | --- | --- |
| 1 静态检查 | 语法、import 白名单、禁止危险调用、接口与 meta 一致性 | 仅针对生成的算子 |
| 2 沙箱执行 | 独立子进程，限制超时与内存，清除含 KEY/TOKEN 的环境变量 | 超时 180 s，内存 8192 MB |
| 3 契约检查 | 不修改输入、行数一致、不引入 NaN/Inf、概率合法、能处理未见类别与缺失值 | 冒烟样本 300 行 |
| 4 指标检查 | 阈值、种子间稳定性、疑似泄漏、耗时与内存 | 5 折 × 种子 [42, 7, 2024]；std 上限 0.03 |

最优方案资源消耗：单次训练 0.901 s，预测 315.5 ms / 千行，峰值内存 282 MB。

修复前的 1 次验证结果保留在 state.json 的 history 中。

## 7. 知识沉淀

- 回写验证记录 ValidationRun：3 条
- 注册到能力库的新算子：无
- 新提炼的经验：确保算子元数据定义与构造函数默认值类型一致（E_INTERFACE）
- 本次提供的经验：7 条（累计检索命中，不代表采纳）

## 附录 A. 生成的算法代码（pipeline.py）

```python
# 警告：该方案未通过验证，仅为得分最高的候选，供人工参考。
"""预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75；将 tot —— 自动生成的算法流水线

需求：预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75；将 total_charges 的空白字符串转换并填补为合理数值。
方案：random_forest_pipeline
设计依据：利用随机森林处理复杂的非线性关系，引入消费比值特征以捕捉流失信号。
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
from factory.operators.tabular import OneHotCategoricalEncoder


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
        description="将total_charges列中的脏字符串（含空白符）转换为数值，并用0填补缺失值",
        conditions=["数据包含字符串类型的数值列", "存在因空白符导致的转换错误"],
        dependencies=["pandas", "numpy"],
        params={"col": "total_charges"},
        parent="TotalChargesCleaner@1"
    )

    def __init__(self, col="total_charges"):
        self.col = col

    def fit(self, X, y=None):
        """记录列名，该算子无需学习统计量"""
        self.col_ = self.col
        return self

    def transform(self, X):
        """
        执行转换：
        1. 复制输入以保证不可变性
        2. 处理字符串中的空白符并强制转换为浮点数
        3. 处理转换产生的缺失值（赋值为0）
        """
        X = X.copy()

        if self.col_ in X.columns:
            # 依据 Lessons 修复：先 strip，再 coerce，最后用 0 填充
            # 确保转换过程不抛出异常
            X[self.col_] = pd.to_numeric(
                X[self.col_].astype(str).str.strip(), 
                errors="coerce"
            ).fillna(0.0)

            # 处理可能存在的无穷值，确保输出的数值稳定性
            X[self.col_] = X[self.col_].replace([np.inf, -np.inf], 0.0)

        return X

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class ChargeRatioFeatureGenerator(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="ChargeRatioFeatureGenerator",
        kind=OperatorKind.FEATURE,
        version=2,
        origin="synthesized",
        description="生成消费比值特征，通过处理total_charges脏数据并计算总费与月费比例",
        conditions=["数据中包含total_charges, monthly_charges, tenure字段"],
        dependencies=["pandas", "numpy", "sklearn"],
        params={"monthly": "monthly_charges", "total": "total_charges", "tenure": "tenure"},
        parent="ChargeRatioFeatureGenerator@1"
    )

    def __init__(self, monthly="monthly_charges", total="total_charges", tenure="tenure"):
        self.monthly = monthly
        self.total = total
        self.tenure = tenure

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # 处理 total_charges 脏数据 (Lesson:69d51358)
        # 将空白字符串和非法字符转为数值，缺失值填补为 0
        if self.total in X.columns:
            X[self.total] = pd.to_numeric(X[self.total].astype(str).str.strip(), errors='coerce').fillna(0.0)

        # 确保 monthly_charges 和 tenure 为数值类型
        X[self.monthly] = pd.to_numeric(X[self.monthly], errors='coerce').fillna(0.0)
        X[self.tenure] = pd.to_numeric(X[self.tenure], errors='coerce').fillna(0.0)

        # 计算特征: avg_monthly = total / tenure (tenure=0时为0)
        # 避免除以0，tenure为0时比值记为1
        tenure_safe = X[self.tenure].replace(0, np.nan)
        avg_calc = X[self.total] / tenure_safe

        # 计算比值: (avg_calc / monthly)
        # 处理无穷值和缺失值
        ratio = avg_calc / X[self.monthly]
        X["charge_ratio"] = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        return X


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_imputer", TotalChargesCleaner()),  # 数据清洗第一步。
        ("1_feature", ChargeRatioFeatureGenerator()),  # 通过领域知识构造特征提升模型感知能力。
        ("2_encoder", OneHotCategoricalEncoder(max_categories=20)),  # 编码。
        ("3_model", RandomForestModel(n_estimators=400, max_depth=12, class_weight='balanced', random_state=42)),  # 利用随机森林非线性拟合能力，通过调整深度控制过拟合。
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

各阶段耗时：PLAN 22.2s，BUILD 48.8s，VERIFY 30.5s，ROUTE 0.0s，REPAIR 2.8s，CURATE 2.2s

```
[PLAN] 场景 telecom_churn；画像标签 ['has_missing_values', 'numeric_features', 'has_categorical_features', 'low_cardinality', 'string_typed_numeric', 'id_columns', 'imbalanced', 'time_tenure', 'mixed_features']
[PLAN] TaskSpec：binary_classification，目标 churn，主指标 recall >= 0.75
[PLAN] 候选方案：['lgbm_pipeline(*TotalChargesCleaner → MissingValueImputer → *IdColumnDropper → OneHotCategoricalEncoder → LightGBMModel)', 'logistic_interpretable_pipeline(*TotalChargesCleaner → *TenureBucketizer → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel)', 'random_forest_pipeline(*TotalChargesCleaner → *ChargeRatioFeatureGenerator → OneHotCategoricalEncoder → RandomForestModel)']
[PLAN] 完成，用时 22.2s → BUILD
[BUILD] 生成算子 TotalChargesCleaner@2（参考 TotalChargesCleaner@1）：将字符串类型的total_charges转换为数值，处理空白符并填补缺失值
[BUILD] 生成算子 IdColumnDropper@2（参考 IdColumnDropper@1）：删除客户ID等标识列以避免编码后维度爆炸
[BUILD] 生成算子 TenureBucketizer@2（参考 TenureBucketizer@1）：对tenure进行分箱以提升线性模型可解释性
[BUILD] 生成算子 ChargeRatioFeatureGenerator@2（参考 ChargeRatioFeatureGenerator@1）：生成消费趋势特征
[BUILD] 完成，用时 48.8s → VERIFY
[VERIFY] lgbm_pipeline：未通过，recall=nan；E_INTERFACE meta.params 与 __init__ 默认值不一致：["id_cols: meta=['customer_id'], __init__=('customer_id',)"]
[VERIFY] logistic_interpretable_pipeline：未通过，recall=nan；E_RUNTIME ValueError: Input X contains NaN.
LogisticRegression does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to pr (at _assert_all_finite_element_wise:186)
[VERIFY] random_forest_pipeline：未通过，recall=0.6775；M_THRESHOLD recall = 0.6775 低于阈值 0.75
[VERIFY] 完成，用时 23.3s → ROUTE
[ROUTE] 完成，用时 0.0s → REPAIR
[REPAIR] 修复 IdColumnDropper@2 第 1 轮：E_INTERFACE meta.params 与 __init__ 默认值不一致：["id_cols: meta=['customer_id'], __init__=('custom
[REPAIR] 完成，用时 2.8s → VERIFY
[VERIFY] lgbm_pipeline：未通过，recall=0.6544；M_THRESHOLD recall = 0.6544 低于阈值 0.75
[VERIFY] 完成，用时 7.3s → ROUTE
[ROUTE] 没有方案通过验证
[ROUTE] 完成，用时 0.0s → CURATE
[CURATE] 写入 KG：3 条 ValidationRun；注册新算子 []；新增经验 1 条
[CURATE] 完成，用时 2.2s → FAILED
```

### LLM 调用来源

| 阶段 | 请求哈希 | 来源 | 本次耗时(s) | 录制耗时(s) | tokens |
| --- | --- | --- | --- | --- | --- |
| plan | 50b3ded985c73e1a | remote | 22.079 | - | 11709 |
| synthesize | 14165a56ca9696fd | remote | 2.707 | - | 2542 |
| synthesize | 9841e20786f21535 | remote | 3.014 | - | 2460 |
| synthesize | 3ea0eb19929c75bb | remote | 40.252 | - | 2636 |
| synthesize | 4207dd75ff0369e9 | remote | 2.833 | - | 2682 |
| repair | ff562e78c2f145c8 | remote | 2.763 | - | 2064 |
| lesson | 4c6c97268e37bf60 | remote | 2.214 | - | 1223 |
