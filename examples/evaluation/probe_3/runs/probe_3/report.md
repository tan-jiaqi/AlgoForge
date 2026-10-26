# AlgoForge 验证报告

| 项目 | 内容 |
| --- | --- |
| 需求 | 预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75；增加累计费用与月费的比值特征，正确处理缺失和零分母。 |
| 数据 | data/churn.csv |
| 运行 ID | probe_3 |
| LLM | gemini-3.5-flash |
| 状态 | ❌ 未通过 |
| 最优方案 | random_forest_robust_pipeline |
| 总耗时 | 104.8 s |

共验证 3 个候选方案，0 个通过；重新规划 0 次，修复 1 轮。最优方案 `random_forest_robust_pipeline` 的 recall = 0.679（要求 ≥ 0.75）。

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
| imputer | MissingValueImputer@1 | 是 | 0.533 | 0.73 / 0.09 / 0.72 | 匹配条件：has_missing_values；历史验证 3 次，平均 recall 0.717 |
| imputer | TotalChargesCleaner@1 | 否（历史实现，可参考生成） | 0.482 | 1.00 / 0.27 / 0.00 | 匹配条件：has_missing_values、numeric_features、string_typed_numeric |
| feature | ChargeRatioFeatureGenerator@1 | 否（历史实现，可参考生成） | 0.695 | 1.00 / 0.18 / 0.80 | 匹配条件：numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | TenureBucketizer@1 | 否（历史实现，可参考生成） | 0.664 | 0.86 / 0.27 / 0.80 | 匹配条件：interpretable、numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | ServiceCountFeature@1 | 否（历史实现，可参考生成） | 0.392 | 0.91 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| feature | IdColumnDropper@1 | 否（历史实现，可参考生成） | 0.349 | 0.81 / 0.09 / 0.00 | 匹配条件：id_columns |
| encoder | OneHotCategoricalEncoder@1 | 是 | 0.590 | 0.80 / 0.18 / 0.72 | 匹配条件：has_categorical_features、low_cardinality；历史验证 3 次，平均 recall 0.717 |
| encoder | KFoldTargetEncoder@1 | 否（历史实现，可参考生成） | 0.427 | 1.00 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| scaler | StandardFeatureScaler@1 | 是 | 0.640 | 1.00 / 0.09 / 0.71 | 匹配条件：numeric_features；历史验证 1 次，平均 recall 0.710 |
| model | LogisticRegressionModel@1 | 是 | 0.640 | 1.00 / 0.09 / 0.71 | 匹配条件：interpretable；历史验证 1 次，平均 recall 0.710 |
| model | BalancedLightGBMClassifier@1 | 否（历史实现，可参考生成） | 0.634 | 0.93 / 0.09 / 0.78 | 匹配条件：imbalanced；历史验证 1 次，平均 recall 0.780 |
| model | RandomForestModel@1 | 是 | 0.568 | 0.86 / 0.09 / 0.66 | 匹配条件：mixed_features；历史验证 1 次，平均 recall 0.660 |
| model | LightGBMModel@1 | 是 | 0.544 | 0.69 / 0.09 / 0.80 | 匹配条件：imbalanced；历史验证 1 次，平均 recall 0.800 |

### 参考的历史经验

- **数值列含空白字符导致类型转换失败**（E_RUNTIME）：先 str.strip()，再 pd.to_numeric(errors='coerce')，缺失值填补为 0
- **样本不平衡导致默认模型召回率偏低**（M_THRESHOLD）：启用平衡类权重（如 class_weight='balanced' 或 BalancedLightGBMClassifier）以提升召回率
- **逻辑回归标准化与可解释性**：必须先进行特征标准化（如 StandardFeatureScaler），在 0.1~10 之间搜索正则化强度 C，可直接输出系数给业务方。
- **随机森林防过拟合与召回优化**：设置 n_estimators 300 左右，限制 max_depth 在 8~16 防止过拟合；召回优先时需配合调整分类阈值。
- **未剔除客户 ID 导致 One-Hot 编码后维度暴涨与内存溢出**（E_TIMEOUT）：建模前剔除 ID 等标识列，并在编码器中设置最大类别数

### 知识引用检查

检索并传入上下文不等于采纳。以下引用由模型声明，程序仅核对可见范围与结构化适用对象；不证明语义落实或指标收益。

| 方案 | 步骤 | 模型引用 | 范围检查 | 语义落实 |
| --- | --- | --- | --- | --- |
| lgbm_recall_priority_pipeline | 1 | Operator:TotalChargesCleaner@1、Lesson:69d51358 | 通过 | 未验证 |
| lgbm_recall_priority_pipeline | 2 | Operator:ChargeRatioFeatureGenerator@1 | 通过 | 未验证 |
| lgbm_recall_priority_pipeline | 3 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| lgbm_recall_priority_pipeline | 4 | Operator:LightGBMModel@1、Lesson:9e9e77d6 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 1 | Operator:TotalChargesCleaner@1 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 2 | Operator:IdColumnDropper@1、Lesson:88299a38 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 3 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 4 | Operator:StandardFeatureScaler@1、Lesson:74e2da10 | 通过 | 未验证 |
| interpretable_logreg_pipeline | 5 | Operator:LogisticRegressionModel@1、Lesson:74e2da10 | 通过 | 未验证 |
| random_forest_robust_pipeline | 1 | Operator:MissingValueImputer@1 | 通过 | 未验证 |
| random_forest_robust_pipeline | 2 | Operator:ServiceCountFeature@1 | 通过 | 未验证 |
| random_forest_robust_pipeline | 3 | Operator:OneHotCategoricalEncoder@1 | 通过 | 未验证 |
| random_forest_robust_pipeline | 4 | Operator:RandomForestModel@1、Lesson:57718714 | 通过 | 未验证 |

## 3. 候选方案比较

排序规则：先看是否通过全部检查，再比主指标，再比次要指标之和，最后训练耗时短者优先。指标为多随机种子 × 分层 K 折交叉验证的均值 ± 种子间标准差。

| 排名 | 方案 | 结果 | recall | roc_auc | precision | f1 | recall_at_20 | 单次训练 (s) | 峰值内存 (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ★ random_forest_robust_pipeline | 未通过 | 0.679 ± 0.002 | 0.800 ± 0.001 | 0.500 ± 0.004 | 0.576 ± 0.003 | 0.450 ± 0.002 | 0.683 | 284 |
| 2 | lgbm_recall_priority_pipeline | 未通过 | 0.591 ± 0.006 | 0.781 ± 0.002 | 0.503 ± 0.003 | 0.543 ± 0.004 | 0.415 ± 0.004 | 0.345 | 262 |
| 3 | interpretable_logreg_pipeline | 未通过 | - | - | - | - | - | - | 233 |

```
recall（| 为阈值 0.75）
random_forest_robust_pipeline        ■■■■■■■■■■■■■■·|···· 0.679
lgbm_recall_priority_pipeline        ■■■■■■■■■■■■···|···· 0.591
interpretable_logreg_pipeline        ···············|···· -
```

### 各方案的结构与诊断

- `random_forest_robust_pipeline`：MissingValueImputer → 生成:ServiceCountFeature → OneHotCategoricalEncoder → RandomForestModel。诊断：`M_THRESHOLD` recall = 0.6789 低于阈值 0.75
- `lgbm_recall_priority_pipeline`：生成:TotalChargesCleaner → 生成:ChargeRatioFeatureGenerator → OneHotCategoricalEncoder → LightGBMModel。诊断：`M_THRESHOLD` recall = 0.5912 低于阈值 0.75
- `interpretable_logreg_pipeline`：生成:TotalChargesCleaner → 生成:IdColumnDropper → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel。诊断：`E_RUNTIME` 未见类别/缺失值健壮性检查失败：ValueError: Input X contains NaN.
LogisticRegression does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to pr (at _assert_all_finite_element_wise:186)

## 4. 最优方案：random_forest_robust_pipeline

使用随机森林，通过特征工程增强模型，利用森林天然的非线性捕捉能力提升召回表现。

| # | 步骤 | 算子 | 参数 | 选择理由 |
| --- | --- | --- | --- | --- |
| 1 | imputer | MissingValueImputer@1 | - | 通用填补策略。 |
| 2 | feature | 生成 ServiceCountFeature | - | 利用业务逻辑提取粘性特征提升模型对复杂流失行为的建模能力。 |
| 3 | encoder | OneHotCategoricalEncoder@1 | - | 类别特征编码。 |
| 4 | model | RandomForestModel@1 | class_weight=balanced, n_estimators=300, max_depth=12 | 随机森林鲁棒性强，设置深度防过拟合，并引入服务数量特征。 |

## 5. 能力生成与修复

| 生成算子 | 类型 | 用途 | 参考的历史实现 | 修复轮数 | 注册到能力库 |
| --- | --- | --- | --- | --- | --- |
| TotalChargesCleaner@2 | imputer | 将字符串类型的数值列转换为浮点型，处理非数值字符并填补缺失值 | TotalChargesCleaner@1 | 0 | 否 |
| ChargeRatioFeatureGenerator@2 | feature | 构造月费与累计费用的比值特征，需处理零分母 | ChargeRatioFeatureGenerator@1 | 0 | 否 |
| IdColumnDropper@2 | feature | 剔除ID列避免维度爆炸 | IdColumnDropper@1 | 0 | 否 |
| ServiceCountFeature@2 | feature | 统计业务粘性特征 | ServiceCountFeature@1 | 1 | 否 |

### 修复 ServiceCountFeature@2 第 1 轮：✅ 已修复

- 诊断：`E_INTERFACE`（static）meta.params 与 __init__ 默认值不一致：["service_cols: meta=['online_security', 'tech_support', 'streaming_tv'], __init__=('online_security', 'tech_support', 'streaming_tv')"]
- 提示：实例化时 meta.params 会覆盖 __init__ 默认值，两处必须一致
- 参考经验：0 条

```diff
--- 修复前
+++ 修复后
@@ -15,3 +15,7 @@
         dependencies=["numpy", "pandas", "sklearn"],
-        params={"service_cols": ["online_security", "tech_support", "streaming_tv"], "monthly_col": "monthly_charges", "total_col": "total_charges"},
+        params={
+            "service_cols": ["online_security", "tech_support", "streaming_tv"],
+            "monthly_col": "monthly_charges",
+            "total_col": "total_charges"
+        },
         parent="ServiceCountFeature@1"
@@ -19,3 +23,3 @@
 
-    def __init__(self, service_cols=("online_security", "tech_support", "streaming_tv"), monthly_col="monthly_charges", total_col="total_charges"):
+    def __init__(self, service_cols=["online_security", "tech_support", "streaming_tv"], monthly_col="monthly_charges", total_col="total_charges"):
         self.service_cols = service_cols
```

## 6. 验证机制

| 级别 | 检查内容 | 本次配置 |
| --- | --- | --- |
| 1 静态检查 | 语法、import 白名单、禁止危险调用、接口与 meta 一致性 | 仅针对生成的算子 |
| 2 沙箱执行 | 独立子进程，限制超时与内存，清除含 KEY/TOKEN 的环境变量 | 超时 180 s，内存 8192 MB |
| 3 契约检查 | 不修改输入、行数一致、不引入 NaN/Inf、概率合法、能处理未见类别与缺失值 | 冒烟样本 300 行 |
| 4 指标检查 | 阈值、种子间稳定性、疑似泄漏、耗时与内存 | 5 折 × 种子 [42, 7, 2024]；std 上限 0.03 |

最优方案资源消耗：单次训练 0.683 s，预测 227.8 ms / 千行，峰值内存 284 MB。

修复前的 1 次验证结果保留在 state.json 的 history 中。

## 7. 知识沉淀

- 回写验证记录 ValidationRun：3 条
- 注册到能力库的新算子：无
- 新提炼的经验：确保算子元数据 params 与 __init__ 默认参数类型及结构严格对齐（E_INTERFACE）
- 本次提供的经验：7 条（累计检索命中，不代表采纳）

## 附录 A. 生成的算法代码（pipeline.py）

```python
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

```

## 附录 B. 运行记录

各阶段耗时：PLAN 6.9s，BUILD 25.3s，VERIFY 26.4s，ROUTE 0.0s，REPAIR 43.2s，CURATE 2.9s

```
[PLAN] 场景 telecom_churn；画像标签 ['has_missing_values', 'numeric_features', 'has_categorical_features', 'low_cardinality', 'string_typed_numeric', 'id_columns', 'imbalanced', 'time_tenure', 'mixed_features']
[PLAN] TaskSpec：binary_classification，目标 churn，主指标 recall >= 0.75
[PLAN] 候选方案：['lgbm_recall_priority_pipeline(*TotalChargesCleaner → *ChargeRatioFeatureGenerator → OneHotCategoricalEncoder → LightGBMModel)', 'interpretable_logreg_pipeline(*TotalChargesCleaner → *IdColumnDropper → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel)', 'random_forest_robust_pipeline(MissingValueImputer → *ServiceCountFeature → OneHotCategoricalEncoder → RandomForestModel)']
[PLAN] 完成，用时 6.9s → BUILD
[BUILD] 生成算子 TotalChargesCleaner@2（参考 TotalChargesCleaner@1）：将字符串类型的数值列转换为浮点型，处理非数值字符并填补缺失值
[BUILD] 生成算子 ChargeRatioFeatureGenerator@2（参考 ChargeRatioFeatureGenerator@1）：构造月费与累计费用的比值特征，需处理零分母
[BUILD] 生成算子 IdColumnDropper@2（参考 IdColumnDropper@1）：剔除ID列避免维度爆炸
[BUILD] 生成算子 ServiceCountFeature@2（参考 ServiceCountFeature@1）：统计业务粘性特征
[BUILD] 完成，用时 25.3s → VERIFY
[VERIFY] lgbm_recall_priority_pipeline：未通过，recall=0.5912；M_THRESHOLD recall = 0.5912 低于阈值 0.75
[VERIFY] interpretable_logreg_pipeline：未通过，recall=nan；E_RUNTIME 未见类别/缺失值健壮性检查失败：ValueError: Input X contains NaN.
LogisticRegression does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to pr (at _assert_all_finite_element_wise:186)
[VERIFY] random_forest_robust_pipeline：未通过，recall=nan；E_INTERFACE meta.params 与 __init__ 默认值不一致：["service_cols: meta=['online_security', 'tech_support', 'streaming_tv'], __init__=('online_security', 'tech_support', 'streaming_tv')"]
[VERIFY] 完成，用时 10.0s → ROUTE
[ROUTE] 完成，用时 0.0s → REPAIR
[REPAIR] 修复 ServiceCountFeature@2 第 1 轮：E_INTERFACE meta.params 与 __init__ 默认值不一致：["service_cols: meta=['online_security', 'tech_sup
[REPAIR] 完成，用时 43.2s → VERIFY
[VERIFY] random_forest_robust_pipeline：未通过，recall=0.6789；M_THRESHOLD recall = 0.6789 低于阈值 0.75
[VERIFY] 完成，用时 16.4s → ROUTE
[ROUTE] 没有方案通过验证
[ROUTE] 完成，用时 0.0s → CURATE
[CURATE] 写入 KG：3 条 ValidationRun；注册新算子 []；新增经验 1 条
[CURATE] 完成，用时 2.9s → FAILED
```

### LLM 调用来源

| 阶段 | 请求哈希 | 来源 | 本次耗时(s) | 录制耗时(s) | tokens |
| --- | --- | --- | --- | --- | --- |
| plan | 3b462e4ab009b785 | remote | 6.765 | - | 11581 |
| synthesize | 88aa61c7e0d3b886 | remote | 15.918 | - | 2637 |
| synthesize | 50c7ae1dd281dd15 | remote | 2.855 | - | 2726 |
| synthesize | 4be0277406bc5045 | remote | 2.705 | - | 2399 |
| synthesize | 6f5d3eb47b53aeca | remote | 3.815 | - | 2686 |
| repair | 5bdb50280dbc24cf | remote | 43.159 | - | 2505 |
| lesson | f6fd0b636997c5c8 | remote | 2.899 | - | 1364 |
