# AlgoForge 验证报告

| 项目 | 内容 |
| --- | --- |
| 需求 | 预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75；正确处理累计费用的空白字符串，并增加在网时长分桶特征。 |
| 数据 | data/churn.csv |
| 运行 ID | probe_2 |
| LLM | gemini-3.5-flash |
| 状态 | ✅ 成功 |
| 最优方案 | interpretable_logreg_pipeline |
| 总耗时 | 45.4 s |

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
| imputer | TotalChargesCleaner@1 | 否（历史实现，可参考生成） | 0.482 | 1.00 / 0.27 / 0.00 | 匹配条件：has_missing_values、numeric_features、string_typed_numeric |
| imputer | MissingValueImputer@1 | 是 | 0.481 | 0.60 / 0.09 / 0.72 | 匹配条件：has_missing_values；历史验证 3 次，平均 recall 0.717 |
| feature | TenureBucketizer@1 | 否（历史实现，可参考生成） | 0.722 | 1.00 / 0.27 / 0.80 | 匹配条件：interpretable、numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | ChargeRatioFeatureGenerator@1 | 否（历史实现，可参考生成） | 0.622 | 0.82 / 0.18 / 0.80 | 匹配条件：numeric_features、time_tenure；历史验证 1 次，平均 recall 0.800 |
| feature | ServiceCountFeature@1 | 否（历史实现，可参考生成） | 0.375 | 0.87 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| feature | IdColumnDropper@1 | 否（历史实现，可参考生成） | 0.344 | 0.79 / 0.09 / 0.00 | 匹配条件：id_columns |
| encoder | OneHotCategoricalEncoder@1 | 是 | 0.574 | 0.76 / 0.18 / 0.72 | 匹配条件：has_categorical_features、low_cardinality；历史验证 3 次，平均 recall 0.717 |
| encoder | KFoldTargetEncoder@1 | 否（历史实现，可参考生成） | 0.427 | 1.00 / 0.09 / 0.00 | 匹配条件：has_categorical_features |
| scaler | StandardFeatureScaler@1 | 是 | 0.640 | 1.00 / 0.09 / 0.71 | 匹配条件：numeric_features；历史验证 1 次，平均 recall 0.710 |
| model | LogisticRegressionModel@1 | 是 | 0.640 | 1.00 / 0.09 / 0.71 | 匹配条件：interpretable；历史验证 1 次，平均 recall 0.710 |
| model | BalancedLightGBMClassifier@1 | 否（历史实现，可参考生成） | 0.625 | 0.91 / 0.09 / 0.78 | 匹配条件：imbalanced；历史验证 1 次，平均 recall 0.780 |
| model | RandomForestModel@1 | 是 | 0.566 | 0.85 / 0.09 / 0.66 | 匹配条件：mixed_features；历史验证 1 次，平均 recall 0.660 |
| model | LightGBMModel@1 | 是 | 0.541 | 0.68 / 0.09 / 0.80 | 匹配条件：imbalanced；历史验证 1 次，平均 recall 0.800 |

### 参考的历史经验

- **数值列含空白字符导致类型转换失败**（E_RUNTIME）：先 str.strip()，再 pd.to_numeric(errors='coerce')，缺失值填补为 0
- **样本不平衡导致默认模型召回率偏低**（M_THRESHOLD）：启用平衡类权重（如 class_weight='balanced' 或 BalancedLightGBMClassifier）以提升召回率
- **逻辑回归标准化与可解释性**：必须先进行特征标准化（如 StandardFeatureScaler），在 0.1~10 之间搜索正则化强度 C，可直接输出系数给业务方。
- **随机森林防过拟合与召回优化**：设置 n_estimators 300 左右，限制 max_depth 在 8~16 防止过拟合；召回优先时需配合调整分类阈值。
- **未剔除客户 ID 导致 One-Hot 编码后维度暴涨与内存溢出**（E_TIMEOUT）：建模前剔除 ID 等标识列，并在编码器中设置最大类别数

### 知识引用检查

每步引用的知识由 Planner 声明，程序检查引用是否在本轮检索结果中、是否适用于该步骤。

| 方案 | 步骤 | 模型引用 | 范围检查 |
| --- | --- | --- | --- |
| （被拒绝的规划） | - | 无 | 未通过：balanced_lgbm_pipeline: synthesize 步骤 4 个，超过上限 2；rf_recall_optimized_pipeline: 数据含类别列，模型之前缺少 encoder 步骤（如 OneHotCategoricalEncoder） |
| balanced_lgbm_pipeline | 1 | Operator:TotalChargesCleaner@1 | 通过 |
| balanced_lgbm_pipeline | 2 | Operator:MissingValueImputer@1 | 通过 |
| balanced_lgbm_pipeline | 3 | Operator:OneHotCategoricalEncoder@1 | 通过 |
| balanced_lgbm_pipeline | 4 | Operator:LightGBMModel@1 | 通过 |
| interpretable_logreg_pipeline | 1 | Operator:TotalChargesCleaner@1 | 通过 |
| interpretable_logreg_pipeline | 2 | Operator:MissingValueImputer@1 | 通过 |
| interpretable_logreg_pipeline | 3 | Operator:OneHotCategoricalEncoder@1 | 通过 |
| interpretable_logreg_pipeline | 4 | Operator:StandardFeatureScaler@1 | 通过 |
| interpretable_logreg_pipeline | 5 | Operator:LogisticRegressionModel@1 | 通过 |
| rf_recall_optimized_pipeline | 1 | Operator:TotalChargesCleaner@1 | 通过 |
| rf_recall_optimized_pipeline | 2 | Operator:MissingValueImputer@1 | 通过 |
| rf_recall_optimized_pipeline | 3 | Operator:OneHotCategoricalEncoder@1 | 通过 |
| rf_recall_optimized_pipeline | 4 | Operator:RandomForestModel@1 | 通过 |

## 3. 候选方案比较

排序规则：先看是否通过全部检查，再比主指标，再比次要指标之和，最后训练耗时短者优先。指标为多随机种子 × 分层 K 折交叉验证的均值 ± 种子间标准差。

| 排名 | 方案 | 结果 | recall | roc_auc | precision | f1 | recall_at_20 | 单次训练 (s) | 峰值内存 (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ★ interpretable_logreg_pipeline | 通过 | 0.778 ± 0.001 | 0.813 ± 0.001 | 0.481 ± 0.001 | 0.595 ± 0.001 | 0.454 ± 0.002 | 0.096 | 259 |
| 2 | rf_recall_optimized_pipeline | 未通过 | 0.682 ± 0.004 | 0.799 ± 0.001 | 0.495 ± 0.004 | 0.574 ± 0.003 | 0.451 ± 0.001 | 0.636 | 284 |
| 3 | balanced_lgbm_pipeline | 未通过 | 0.600 ± 0.009 | 0.779 ± 0.003 | 0.491 ± 0.004 | 0.540 ± 0.005 | 0.416 ± 0.001 | 0.428 | 269 |

```
recall（| 为阈值 0.75）
interpretable_logreg_pipeline        ■■■■■■■■■■■■■■■|···· 0.778
rf_recall_optimized_pipeline         ■■■■■■■■■■■■■■·|···· 0.682
balanced_lgbm_pipeline               ■■■■■■■■■■■■···|···· 0.600
```

### 各方案的结构与诊断

- `interpretable_logreg_pipeline`：生成:TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel。诊断：无
- `rf_recall_optimized_pipeline`：生成:TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → RandomForestModel。诊断：`M_THRESHOLD` recall = 0.6820 低于阈值 0.75
- `balanced_lgbm_pipeline`：生成:TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → LightGBMModel。诊断：`M_THRESHOLD` recall = 0.6002 低于阈值 0.75

## 4. 最优方案：interpretable_logreg_pipeline

该方案侧重于模型的可解释性，符合业务合规约束。

| # | 步骤 | 算子 | 参数 | 选择理由 |
| --- | --- | --- | --- | --- |
| 1 | imputer | 生成 TotalChargesCleaner | - | 清理数值列，参考 Lesson:69d51358。 |
| 2 | imputer | MissingValueImputer@1 | - | 补充缺失数据。 |
| 3 | encoder | OneHotCategoricalEncoder@1 | max_categories=20 | 线性模型需要编码化类别特征。 |
| 4 | scaler | StandardFeatureScaler@1 | - | 逻辑回归必须先标准化，参考 Lesson:74e2da10。 |
| 5 | model | LogisticRegressionModel@1 | class_weight=balanced | 逻辑回归满足可解释性需求。 |

## 5. 能力生成与修复

| 生成算子 | 类型 | 用途 | 参考的历史实现 | 修复轮数 | 注册到能力库 |
| --- | --- | --- | --- | --- | --- |
| TotalChargesCleaner@2 | imputer | 处理总费用列的缺失值与数据格式问题 | TotalChargesCleaner@1 | 0 | 是 |

## 6. 验证机制

| 级别 | 检查内容 | 本次配置 |
| --- | --- | --- |
| 1 静态检查 | 语法、import 白名单、禁止危险调用、接口与 meta 一致性 | 仅针对生成的算子 |
| 2 沙箱执行 | 独立子进程，限制超时与内存，清除含 KEY/TOKEN 的环境变量 | 超时 180 s，内存 8192 MB |
| 3 契约检查 | 不修改输入、行数一致、不引入 NaN/Inf、概率合法、能处理未见类别与缺失值 | 冒烟样本 300 行 |
| 4 指标检查 | 阈值、种子间稳定性、疑似泄漏、耗时与内存 | 5 折 × 种子 [42, 7, 2024]；std 上限 0.03 |

最优方案资源消耗：单次训练 0.096 s，预测 51.0 ms / 千行，峰值内存 259 MB。

## 7. 知识沉淀

- 回写验证记录 ValidationRun：3 条
- 注册到能力库的新算子：TotalChargesCleaner@2
- 新提炼的经验：无
- 本次检索使用的经验：6 条（命中次数已累计）

## 附录 A. 生成的算法代码（pipeline.py）

```python
"""预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75；正确处理累 —— 自动生成的算法流水线

需求：预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75；正确处理累计费用的空白字符串，并增加在网时长分桶特征。
方案：interpretable_logreg_pipeline
设计依据：该方案侧重于模型的可解释性，符合业务合规约束。
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

        # 1. 清洗 total_charges 列
        # Lesson:69d51358 - 处理空白字符串导致转换失败的问题
        if self.col in X.columns:
            X[self.col] = pd.to_numeric(
                X[self.col].astype(str).str.strip(), 
                errors="coerce"
            ).fillna(0.0)

            # 确保不存在无穷值
            X[self.col] = X[self.col].replace([np.inf, -np.inf], 0.0)

        # 2. 增加在网时长分桶特征 (tenure_binned)
        # 增加可解释性特征，将持续时间分为阶段
        if self.tenure_col in X.columns:
            bins = [0, 6, 12, 24, 48, 60, 100]
            labels = ['0-6m', '7-12m', '13-24m', '25-48m', '49-60m', '60m+']
            X['tenure_binned'] = pd.cut(
                X[self.tenure_col], 
                bins=bins, 
                labels=labels, 
                include_lowest=True
            ).astype(str)

            # 处理分桶后的缺失值
            X['tenure_binned'] = X['tenure_binned'].fillna('unknown')

        return X


TARGET = 'churn'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_imputer", TotalChargesCleaner()),  # 清理数值列，参考 Lesson:69d51358。
        ("1_imputer", MissingValueImputer()),  # 补充缺失数据。
        ("2_encoder", OneHotCategoricalEncoder(max_categories=20)),  # 线性模型需要编码化类别特征。
        ("3_scaler", StandardFeatureScaler()),  # 逻辑回归必须先标准化，参考 Lesson:74e2da10。
        ("4_model", LogisticRegressionModel(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)),  # 逻辑回归满足可解释性需求。
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

各阶段耗时：PLAN 12.2s，BUILD 4.8s，VERIFY 28.3s，ROUTE 0.0s，CURATE 0.0s

```
[PLAN] 场景 telecom_churn；画像标签 ['has_missing_values', 'numeric_features', 'has_categorical_features', 'low_cardinality', 'string_typed_numeric', 'id_columns', 'imbalanced', 'time_tenure', 'mixed_features']
[PLAN] TaskSpec：binary_classification，目标 churn，主指标 recall >= 0.75
[PLAN] 候选方案：['balanced_lgbm_pipeline(*TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → LightGBMModel)', 'interpretable_logreg_pipeline(*TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → StandardFeatureScaler → LogisticRegressionModel)', 'rf_recall_optimized_pipeline(*TotalChargesCleaner → MissingValueImputer → OneHotCategoricalEncoder → RandomForestModel)']
[PLAN] 完成，用时 12.2s → BUILD
[BUILD] 生成算子 TotalChargesCleaner@2（参考 TotalChargesCleaner@1）：处理总费用列的缺失值与数据格式问题
[BUILD] 完成，用时 4.8s → VERIFY
[VERIFY] balanced_lgbm_pipeline：未通过，recall=0.6002；M_THRESHOLD recall = 0.6002 低于阈值 0.75
[VERIFY] interpretable_logreg_pipeline：通过，recall=0.7785
[VERIFY] rf_recall_optimized_pipeline：未通过，recall=0.6820；M_THRESHOLD recall = 0.6820 低于阈值 0.75
[VERIFY] 完成，用时 28.3s → ROUTE
[ROUTE] 完成，用时 0.0s → CURATE
[CURATE] 写入 KG：3 条 ValidationRun；注册新算子 ['TotalChargesCleaner@2']；新增经验 0 条
[CURATE] 完成，用时 0.0s → DONE
```

### LLM 调用来源

| 阶段 | 请求哈希 | 来源 | 本次耗时(s) | 录制耗时(s) | tokens |
| --- | --- | --- | --- | --- | --- |
| plan | eb95e205c2aa9b6f | remote | 7.168 | - | 11725 |
| plan | 6effd3e625dbc8d3 | remote | 4.897 | - | 13493 |
| synthesize | 94462cfc4b3d30fa | remote | 4.763 | - | 2729 |
