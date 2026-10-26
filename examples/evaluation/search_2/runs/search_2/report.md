# AlgoForge 验证报告

| 项目 | 内容 |
| --- | --- |
| 需求 | 预测客户流失，召回率不低于0.8；优先尝试可解释模型，未达标时探索不同模型或参数。 |
| 数据 | data/churn.csv |
| 运行 ID | search_2 |
| LLM | gemini-3.5-flash |
| 状态 | ✅ 成功 |
| 最优方案 | interpretable_logreg_pipeline |
| 总耗时 | 45.2 s |

共验证 3 个候选方案，1 个通过；重新规划 1 次，修复 0 轮。最优方案 `interpretable_logreg_pipeline` 的 recall = 0.848（要求 ≥ 0.8）。

## 1. 需求理解（TaskSpec）

| 字段 | 值 |
| --- | --- |
| 任务类型 | binary_classification |
| 数据模态 | tabular |
| 目标列 | churn |
| 评价指标 | recall ≥ 0.8（主） |
| 业务约束 | interpretable、recall_first |
| 数据画像标签 | has_missing_values、numeric_features、has_categorical_features、low_cardinality、string_typed_numeric、id_columns、imbalanced、time_tenure、mixed_features |

## 2. 知识检索（设计依据）

匹配场景：telecom_churn。每类步骤从知识图谱检索候选算子，得分 = 0.4 × 文本相关度 + 0.3 × 条件匹配 + 0.3 × 历史验证分。

| 步骤 | 候选算子 | 可直接复用 | 得分 | 文本/条件/历史 | 证据 |
| --- | --- | --- | --- | --- | --- |
| imputer | MissingValueImputer@1 | 是 | 0.492 | 0.62 / 0.09 / 0.72 | 匹配条件：has_missing_values；历史验证 3 次，平均 recall 0.717 |
| imputer | TotalChargesCleaner@1 | 否（历史实现，可参考生成） | 0.482 | 1.00 / 0.27 / 0.00 | 匹配条件：has_missing_values、numeric_features、string_typed_numeric |
| feature | TenureBucketizer@1 | 否（历史实现，可参考生成） | 0.722 | 1.00 / 0.27 / 0.80 | 匹配条件：interpretable、numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | ChargeRatioFeatureGenerator@1 | 否（历史实现，可参考生成） | 0.616 | 0.80 / 0.18 / 0.80 | 匹配条件：numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | IdColumnDropper@1 | 否（历史实现，可参考生成） | 0.419 | 0.98 / 0.09 / 0.00 | 匹配条件：id_columns |
| feature | ServiceCountFeature@1 | 否（历史实现，可参考生成） | 0.415 | 0.97 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| encoder | OneHotCategoricalEncoder@1 | 是 | 0.669 | 1.00 / 0.18 / 0.72 | 匹配条件：has_categorical_features、low_cardinality；历史验证 3 次，平均 recall 0.717 |
| encoder | KFoldTargetEncoder@1 | 否（历史实现，可参考生成） | 0.366 | 0.85 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| scaler | StandardFeatureScaler@1 | 是 | 0.640 | 1.00 / 0.09 / 0.71 | 匹配条件：numeric_features；历史验证 1 次，平均 recall 0.710 |
| model | BalancedLightGBMClassifier@1 | 否（历史实现，可参考生成） | 0.661 | 1.00 / 0.09 / 0.78 | 匹配条件：imbalanced；历史验证 1 次，平均 recall 0.780 |
| model | LogisticRegressionModel@1 | 是 | 0.618 | 0.94 / 0.09 / 0.71 | 匹配条件：interpretable；历史验证 1 次，平均 recall 0.710 |
| model | LightGBMModel@1 | 是 | 0.549 | 0.70 / 0.09 / 0.80 | 匹配条件：imbalanced；历史验证 1 次，平均 recall 0.800 |
| model | RandomForestModel@1 | 是 | 0.541 | 0.79 / 0.09 / 0.66 | 匹配条件：mixed_features；历史验证 1 次，平均 recall 0.660 |

### 参考的历史经验

- **样本不平衡导致默认模型召回率偏低**（M_THRESHOLD）：启用平衡类权重（如 class_weight='balanced' 或 BalancedLightGBMClassifier）以提升召回率
- **逻辑回归标准化与可解释性**：必须先进行特征标准化（如 StandardFeatureScaler），在 0.1~10 之间搜索正则化强度 C，可直接输出系数给业务方。
- **随机森林防过拟合与召回优化**：设置 n_estimators 300 左右，限制 max_depth 在 8~16 防止过拟合；召回优先时需配合调整分类阈值。
- **数值列含空白字符导致类型转换失败**（E_RUNTIME）：先 str.strip()，再 pd.to_numeric(errors='coerce')，缺失值填补为 0
- **LightGBM 调参及不平衡处理**：设置 num_leaves 15~63, learning_rate 0.03~0.1, n_estimators 200~600；不平衡时设置 class_weight='balanced'；树模型无需标准化。

### 知识引用检查

每步引用的知识由 Planner 声明，程序检查引用是否在本轮检索结果中、是否适用于该步骤。

| 方案 | 步骤 | 模型引用 | 范围检查 |
| --- | --- | --- | --- |
| lgbm_imbalanced_pipeline | 1 | Operator:TotalChargesCleaner@1、Lesson:69d51358 | 通过 |
| lgbm_imbalanced_pipeline | 2 | Operator:MissingValueImputer@1 | 通过 |
| lgbm_imbalanced_pipeline | 3 | Operator:IdColumnDropper@1 | 通过 |
| lgbm_imbalanced_pipeline | 4 | Operator:OneHotCategoricalEncoder@1 | 通过 |
| lgbm_imbalanced_pipeline | 5 | Operator:LightGBMModel@1、Lesson:f2c1398b、Lesson:9e9e77d6 | 通过 |
| （被拒绝的规划） | - | 无 | 未通过：lgbm_recall_optimized: 第 2 步：缺少 Operator:MissingValueImputer@1 证据引用；lgbm_recall_optimized: 第 4 步：缺少 Operator:OneHotCategoricalEncoder@1 证据引用；lgbm_recall_optimized: 第 5 步：缺少 Operator:LightGBMModel@1 证据引用；balanced_lgbm_with_features: 第 5 步：未知算子 BalancedLightGBMClassifier@1；balanced_lgbm_with_features: |
| lgbm_recall_optimized | 1 | Operator:TotalChargesCleaner@1 | 通过 |
| lgbm_recall_optimized | 2 | Operator:MissingValueImputer@1 | 通过 |
| lgbm_recall_optimized | 3 | Operator:ChargeRatioFeatureGenerator@1 | 通过 |
| lgbm_recall_optimized | 4 | Operator:OneHotCategoricalEncoder@1 | 通过 |
| lgbm_recall_optimized | 5 | Operator:LightGBMModel@1 | 通过 |
| rf_recall_optimized | 1 | Operator:MissingValueImputer@1 | 通过 |
| rf_recall_optimized | 2 | Operator:ServiceCountFeature@1 | 通过 |
| rf_recall_optimized | 3 | Operator:OneHotCategoricalEncoder@1 | 通过 |
| rf_recall_optimized | 4 | Operator:RandomForestModel@1 | 通过 |
| interpretable_logreg_pipeline | 1 | Operator:MissingValueImputer@1 | 通过 |
| interpretable_logreg_pipeline | 2 | Operator:OneHotCategoricalEncoder@1 | 通过 |
| interpretable_logreg_pipeline | 3 | Operator:StandardFeatureScaler@1 | 通过 |
| interpretable_logreg_pipeline | 4 | Operator:LogisticRegressionModel@1 | 通过 |
| lgbm_basic_balanced | 1 | Operator:MissingValueImputer@1 | 通过 |
| lgbm_basic_balanced | 2 | Operator:TenureBucketizer@1 | 通过 |
| lgbm_basic_balanced | 3 | Operator:OneHotCategoricalEncoder@1 | 通过 |
| lgbm_basic_balanced | 4 | Operator:LightGBMModel@1 | 通过 |

## 3. 候选方案比较

排序规则：先看是否通过全部检查，再比主指标，再比次要指标之和，最后训练耗时短者优先。指标为多随机种子 × 分层 K 折交叉验证的均值 ± 种子间标准差。

| 排名 | 方案 | 结果 | recall | roc_auc | precision | f1 | recall_at_20 | 单次训练 (s) | 峰值内存 (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ★ interpretable_logreg_pipeline | 通过 | 0.848 ± 0.003 | 0.814 ± 0.001 | 0.443 ± 0.001 | 0.582 ± 0.001 | 0.457 ± 0.002 | 0.097 | 257 |
| 2 | lgbm_basic_balanced | 未通过 | 0.639 ± 0.010 | 0.785 ± 0.002 | 0.493 ± 0.002 | 0.556 ± 0.005 | 0.420 ± 0.000 | 0.368 | 266 |
| 3 | lgbm_imbalanced_pipeline | 未通过 | 0.624 ± 0.007 | 0.783 ± 0.003 | 0.492 ± 0.002 | 0.550 ± 0.003 | 0.420 ± 0.006 | 0.333 | 261 |

```
recall（| 为阈值 0.8）
interpretable_logreg_pipeline        ■■■■■■■■■■■■■■■■|··· 0.848
lgbm_basic_balanced                  ■■■■■■■■■■■■■···|··· 0.639
lgbm_imbalanced_pipeline             ■■■■■■■■■■■■····|··· 0.624
```

### 各方案的结构与诊断

- `interpretable_logreg_pipeline`：MissingValueImputer → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel。诊断：无
- `lgbm_basic_balanced`：MissingValueImputer → 生成:TenureBucketizer → OneHotCategoricalEncoder → LightGBMModel。诊断：`M_THRESHOLD` recall = 0.6391 低于阈值 0.8
- `lgbm_imbalanced_pipeline`：生成:TotalChargesCleaner → MissingValueImputer → 生成:IdColumnDropper → OneHotCategoricalEncoder → LightGBMModel。诊断：`M_THRESHOLD` recall = 0.6243 低于阈值 0.8

### 方案搜索（Beam Search）

指标不达标时，以已验证结果排名前 k 的方案为父方案，由 Planner 扩展 k × m 个子方案；与已验证方案结构相同的直接去重，其余按先验（知识图谱中该场景的历史验证记录 + 本次运行的观测，平滑后按步骤加权，模型步骤权重更高）估计 recall，只保留前 k 个进入验证。

第 1 轮：父方案 `lgbm_imbalanced_pipeline`，beam 宽度 2。

| 子方案 | 结构 | 先验 recall | 实际 recall | 结果 |
| --- | --- | --- | --- | --- |
| lgbm_recall_optimized | 生成:TotalChargesCleaner → MissingValueImputer → 生成:ChargeRatioFeatureGenerator → OneHotCategoricalEncoder → LightGBMModel | 0.738 | - | 先验第 3，超出 beam 宽度 2，剪枝 |
| rf_recall_optimized | MissingValueImputer → 生成:ServiceCountFeature → OneHotCategoricalEncoder → RandomForestModel | 0.737 | - | 先验第 4，超出 beam 宽度 2，剪枝 |
| ✔ interpretable_logreg_pipeline | MissingValueImputer → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel | 0.742 | 0.848 | 先验第 2，进入验证 |
| ✔ lgbm_basic_balanced | MissingValueImputer → 生成:TenureBucketizer → OneHotCategoricalEncoder → LightGBMModel | 0.742 | 0.639 | 先验第 1，进入验证 |

| 子方案 | 步骤 | 先验证据来源 | 数值 |
| --- | --- | --- | --- |
| lgbm_recall_optimized | *TotalChargesCleaner | 本次运行观测 | 0.6243 |
| lgbm_recall_optimized | MissingValueImputer@1 | ValidationRun:e11ba947 | 0.71 |
| lgbm_recall_optimized | MissingValueImputer@1 | ValidationRun:e0d43fa3 | 0.66 |
| lgbm_recall_optimized | MissingValueImputer@1 | ValidationRun:a7f163bc | 0.78 |
| lgbm_recall_optimized | MissingValueImputer@1 | 本次运行观测 | 0.6243 |
| lgbm_recall_optimized | *ChargeRatioFeatureGenerator | ValidationRun:f6e4284c | 0.8 |
| lgbm_recall_optimized | OneHotCategoricalEncoder@1 | ValidationRun:e11ba947 | 0.71 |
| lgbm_recall_optimized | OneHotCategoricalEncoder@1 | ValidationRun:e0d43fa3 | 0.66 |
| lgbm_recall_optimized | OneHotCategoricalEncoder@1 | ValidationRun:a7f163bc | 0.78 |
| lgbm_recall_optimized | OneHotCategoricalEncoder@1 | 本次运行观测 | 0.6243 |
| lgbm_recall_optimized | LightGBMModel@1 | ValidationRun:f6e4284c | 0.8 |
| lgbm_recall_optimized | LightGBMModel@1 | 本次运行观测 | 0.6243 |
| rf_recall_optimized | MissingValueImputer@1 | ValidationRun:e11ba947 | 0.71 |
| rf_recall_optimized | MissingValueImputer@1 | ValidationRun:e0d43fa3 | 0.66 |
| rf_recall_optimized | MissingValueImputer@1 | ValidationRun:a7f163bc | 0.78 |
| rf_recall_optimized | MissingValueImputer@1 | 本次运行观测 | 0.6243 |
| rf_recall_optimized | OneHotCategoricalEncoder@1 | ValidationRun:e11ba947 | 0.71 |
| rf_recall_optimized | OneHotCategoricalEncoder@1 | ValidationRun:e0d43fa3 | 0.66 |
| rf_recall_optimized | OneHotCategoricalEncoder@1 | ValidationRun:a7f163bc | 0.78 |
| rf_recall_optimized | OneHotCategoricalEncoder@1 | 本次运行观测 | 0.6243 |
| rf_recall_optimized | RandomForestModel@1 | ValidationRun:e0d43fa3 | 0.66 |
| interpretable_logreg_pipeline | MissingValueImputer@1 | ValidationRun:e11ba947 | 0.71 |
| interpretable_logreg_pipeline | MissingValueImputer@1 | ValidationRun:e0d43fa3 | 0.66 |
| interpretable_logreg_pipeline | MissingValueImputer@1 | ValidationRun:a7f163bc | 0.78 |
| interpretable_logreg_pipeline | MissingValueImputer@1 | 本次运行观测 | 0.6243 |
| interpretable_logreg_pipeline | OneHotCategoricalEncoder@1 | ValidationRun:e11ba947 | 0.71 |
| interpretable_logreg_pipeline | OneHotCategoricalEncoder@1 | ValidationRun:e0d43fa3 | 0.66 |
| interpretable_logreg_pipeline | OneHotCategoricalEncoder@1 | ValidationRun:a7f163bc | 0.78 |
| interpretable_logreg_pipeline | OneHotCategoricalEncoder@1 | 本次运行观测 | 0.6243 |
| interpretable_logreg_pipeline | StandardFeatureScaler@1 | ValidationRun:e11ba947 | 0.71 |
| interpretable_logreg_pipeline | LogisticRegressionModel@1 | ValidationRun:e11ba947 | 0.71 |
| lgbm_basic_balanced | MissingValueImputer@1 | ValidationRun:e11ba947 | 0.71 |
| lgbm_basic_balanced | MissingValueImputer@1 | ValidationRun:e0d43fa3 | 0.66 |
| lgbm_basic_balanced | MissingValueImputer@1 | ValidationRun:a7f163bc | 0.78 |
| lgbm_basic_balanced | MissingValueImputer@1 | 本次运行观测 | 0.6243 |
| lgbm_basic_balanced | *TenureBucketizer | ValidationRun:f6e4284c | 0.8 |
| lgbm_basic_balanced | OneHotCategoricalEncoder@1 | ValidationRun:e11ba947 | 0.71 |
| lgbm_basic_balanced | OneHotCategoricalEncoder@1 | ValidationRun:e0d43fa3 | 0.66 |
| lgbm_basic_balanced | OneHotCategoricalEncoder@1 | ValidationRun:a7f163bc | 0.78 |
| lgbm_basic_balanced | OneHotCategoricalEncoder@1 | 本次运行观测 | 0.6243 |
| lgbm_basic_balanced | LightGBMModel@1 | ValidationRun:f6e4284c | 0.8 |
| lgbm_basic_balanced | LightGBMModel@1 | 本次运行观测 | 0.6243 |

## 4. 最优方案：interpretable_logreg_pipeline

针对可解释性约束，使用逻辑回归，并通过加大正类权重系数（class_weight）强行拉升召回率。

| # | 步骤 | 算子 | 参数 | 选择理由 |
| --- | --- | --- | --- | --- |
| 1 | imputer | MissingValueImputer@1 | - | - |
| 2 | encoder | OneHotCategoricalEncoder@1 | - | - |
| 3 | scaler | StandardFeatureScaler@1 | - | - |
| 4 | model | LogisticRegressionModel@1 | C=0.5, class_weight={'0': 1, '1': 4}, max_iter=1000, random_state=42 | - |

## 5. 能力生成与修复

| 生成算子 | 类型 | 用途 | 参考的历史实现 | 修复轮数 | 注册到能力库 |
| --- | --- | --- | --- | --- | --- |
| TotalChargesCleaner@2 | imputer | 将total_charges字段中的空白字符处理为0.0并转为数值类型 | TotalChargesCleaner@1 | 0 | 否 |
| IdColumnDropper@2 | feature | 删除客户ID等无预测价值的特征 | IdColumnDropper@1 | 0 | 否 |
| TenureBucketizer@2 | feature | 对tenure进行分箱以辅助模型识别流失周期 | TenureBucketizer@1 | 0 | 否 |

## 6. 验证机制

| 级别 | 检查内容 | 本次配置 |
| --- | --- | --- |
| 1 静态检查 | 语法、import 白名单、禁止危险调用、接口与 meta 一致性 | 仅针对生成的算子 |
| 2 沙箱执行 | 独立子进程，限制超时与内存，清除含 KEY/TOKEN 的环境变量 | 超时 180 s，内存 8192 MB |
| 3 契约检查 | 不修改输入、行数一致、不引入 NaN/Inf、概率合法、能处理未见类别与缺失值 | 冒烟样本 300 行 |
| 4 指标检查 | 阈值、种子间稳定性、疑似泄漏、耗时与内存 | 5 折 × 种子 [42, 7, 2024]；std 上限 0.03 |

最优方案资源消耗：单次训练 0.097 s，预测 51.7 ms / 千行，峰值内存 257 MB。

## 7. 知识沉淀

- 回写验证记录 ValidationRun：3 条
- 注册到能力库的新算子：无
- 新提炼的经验：无
- 本次检索使用的经验：7 条（命中次数已累计）

## 附录 A. 生成的算法代码（pipeline.py）

```python
"""预测客户流失，召回率不低于0.8；优先尝试可解释模型，未达标时探索不同模型或参数 —— 自动生成的算法流水线

需求：预测客户流失，召回率不低于0.8；优先尝试可解释模型，未达标时探索不同模型或参数。
方案：interpretable_logreg_pipeline
设计依据：针对可解释性约束，使用逻辑回归，并通过加大正类权重系数（class_weight）强行拉升召回率。
主指标：recall >= 0.8

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


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_imputer", MissingValueImputer()),
        ("1_encoder", OneHotCategoricalEncoder(max_categories=20)),
        ("2_scaler", StandardFeatureScaler()),
        ("3_model", LogisticRegressionModel(C=0.5, class_weight={'0': 1, '1': 4}, max_iter=1000, random_state=42)),
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

各阶段耗时：PLAN 3.9s，BUILD 10.1s，VERIFY 19.9s，ROUTE 0.0s，REPLAN 11.3s，CURATE 0.0s

```
[PLAN] 场景 telecom_churn；画像标签 ['has_missing_values', 'numeric_features', 'has_categorical_features', 'low_cardinality', 'string_typed_numeric', 'id_columns', 'imbalanced', 'time_tenure', 'mixed_features']
[PLAN] TaskSpec：binary_classification，目标 churn，主指标 recall >= 0.8
[PLAN] 候选方案：['lgbm_imbalanced_pipeline(*TotalChargesCleaner → MissingValueImputer → *IdColumnDropper → OneHotCategoricalEncoder → LightGBMModel)']
[PLAN] 完成，用时 3.9s → BUILD
[BUILD] 生成算子 TotalChargesCleaner@2（参考 TotalChargesCleaner@1）：将total_charges字段中的空白字符处理为0.0并转为数值类型
[BUILD] 生成算子 IdColumnDropper@2（参考 IdColumnDropper@1）：删除客户ID等无预测价值的特征
[BUILD] 完成，用时 7.2s → VERIFY
[VERIFY] lgbm_imbalanced_pipeline：未通过，recall=0.6243；M_THRESHOLD recall = 0.6243 低于阈值 0.8
[VERIFY] 完成，用时 7.5s → ROUTE
[ROUTE] 完成，用时 0.0s → REPLAN
[REPLAN] 第 1 轮 Beam 搜索：父方案 ['lgbm_imbalanced_pipeline']，扩展 4 个，入选 ['lgbm_basic_balanced(MissingValueImputer → *TenureBucketizer → OneHotCategoricalEncoder → LightGBMModel)', 'interpretable_logreg_pipeline(MissingValueImputer → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel)']，剪枝/去重 ['lgbm_recall_optimized(0.738)', 'rf_recall_optimized(0.7366)']
[REPLAN] 完成，用时 11.3s → BUILD
[BUILD] 生成算子 TenureBucketizer@2（参考 TenureBucketizer@1）：对tenure进行分箱以辅助模型识别流失周期
[BUILD] 完成，用时 3.0s → VERIFY
[VERIFY] lgbm_basic_balanced：未通过，recall=0.6391；M_THRESHOLD recall = 0.6391 低于阈值 0.8
[VERIFY] interpretable_logreg_pipeline：通过，recall=0.8475
[VERIFY] 完成，用时 12.4s → ROUTE
[ROUTE] 完成，用时 0.0s → CURATE
[CURATE] 写入 KG：3 条 ValidationRun；注册新算子 []；新增经验 0 条
[CURATE] 完成，用时 0.0s → DONE
```

### LLM 调用来源

| 阶段 | 请求哈希 | 来源 | 本次耗时(s) | 录制耗时(s) | tokens |
| --- | --- | --- | --- | --- | --- |
| plan | 5939545d51d8b000 | remote | 3.692 | - | 10431 |
| synthesize | 35e6fe47eda1fdb9 | remote | 4.341 | - | 2433 |
| synthesize | 741e2165ea59e3f5 | remote | 2.815 | - | 2322 |
| plan | 1af8e95985d1d74a | remote | 6.775 | - | 12281 |
| plan | c0751cf5c8508b39 | remote | 4.463 | - | 14737 |
| synthesize | 9b9b349c410be80c | remote | 2.960 | - | 2607 |
