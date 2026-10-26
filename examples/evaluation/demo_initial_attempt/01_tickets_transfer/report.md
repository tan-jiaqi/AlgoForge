# AlgoForge 验证报告

| 项目 | 内容 |
| --- | --- |
| 需求 | 把客服工单自动分到网络故障、账单争议、套餐变更、销户投诉四类，宏平均 F1 不低于 0.8 |
| 数据 | data/tickets.csv |
| 运行 ID | 01_tickets_transfer |
| LLM | gemini-3.5-flash |
| 状态 | ✅ 成功 |
| 最优方案 | tfidf_logistic_regression |
| 总耗时 | 71.5 s |

共验证 2 个候选方案，2 个通过；重新规划 0 次，修复 1 轮。最优方案 `tfidf_logistic_regression` 的 f1_macro = 0.824（要求 ≥ 0.8）。

## 1. 需求理解（TaskSpec）

| 字段 | 值 |
| --- | --- |
| 任务类型 | multiclass_classification |
| 数据模态 | text |
| 目标列 | label（文本列 text） |
| 评价指标 | f1_macro ≥ 0.8（主） |
| 业务约束 | - |
| 数据画像标签 | id_columns、text_input、imbalanced |

## 2. 知识检索（设计依据）

匹配场景：新场景。每类步骤从知识图谱检索候选算子，得分 = 0.4 × 文本相关度 + 0.3 × 条件匹配 + 0.3 × 历史验证分。

| 步骤 | 候选算子 | 可直接复用 | 得分 | 文本/条件/历史 | 证据 |
| --- | --- | --- | --- | --- | --- |
| vectorizer | TfidfTextVectorizer@1 | 是 | 0.500 | 1.00 / 0.33 / 0.00 | 匹配条件：text_input |
| model | LogisticRegressionModel@1 | 是 | 0.230 | 0.00 / 0.00 / 0.77 | 历史验证 1 次，平均 f1_macro 0.765 |

### 参考的历史经验

- **样本不平衡导致默认模型召回率偏低**（M_THRESHOLD）：启用平衡类权重（如 class_weight='balanced' 或 BalancedLightGBMClassifier）以提升召回率
- **LightGBM 调参及不平衡处理**：设置 num_leaves 15~63, learning_rate 0.03~0.1, n_estimators 200~600；不平衡时设置 class_weight='balanced'；树模型无需标准化。
- **数值列含空白字符导致类型转换失败**（E_RUNTIME）：先 str.strip()，再 pd.to_numeric(errors='coerce')，缺失值填补为 0
- **未剔除客户 ID 导致 One-Hot 编码后维度暴涨与内存溢出**（E_TIMEOUT）：建模前剔除 ID 等标识列，并在编码器中设置最大类别数
- **随机森林防过拟合与召回优化**：设置 n_estimators 300 左右，限制 max_depth 在 8~16 防止过拟合；召回优先时需配合调整分类阈值。

### 知识引用检查

检索并传入上下文不等于采纳。以下引用由模型声明，程序仅核对可见范围与结构化适用对象；不证明语义落实或指标收益。

| 方案 | 步骤 | 模型引用 | 范围检查 | 语义落实 |
| --- | --- | --- | --- | --- |
| tfidf_logistic_regression | 1 | Operator:TfidfTextVectorizer@1 | 通过 | 未验证 |
| tfidf_logistic_regression | 2 | Operator:LogisticRegressionModel@1、Lesson:9e9e77d6 | 通过 | 未验证 |
| tfidf_lightgbm_classifier | 1 | Operator:TfidfTextVectorizer@1 | 通过 | 未验证 |
| tfidf_lightgbm_classifier | 2 | Lesson:f2c1398b | 通过 | 未验证 |

## 3. 候选方案比较

排序规则：先看是否通过全部检查，再比主指标，再比次要指标之和，最后训练耗时短者优先。指标为多随机种子 × 分层 K 折交叉验证的均值 ± 种子间标准差。

| 排名 | 方案 | 结果 | f1_macro | accuracy | recall_macro | precision_macro | 单次训练 (s) | 峰值内存 (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ★ tfidf_logistic_regression | 通过 | 0.824 ± 0.004 | 0.838 ± 0.004 | 0.832 ± 0.003 | 0.820 ± 0.004 | 0.118 | 241 |
| 2 | tfidf_lightgbm_classifier | 通过 | 0.818 ± 0.005 | 0.834 ± 0.005 | 0.818 ± 0.004 | 0.821 ± 0.006 | 0.546 | 265 |

```
f1_macro（| 为阈值 0.8）
tfidf_logistic_regression            ■■■■■■■■■■■■■■■■|··· 0.824
tfidf_lightgbm_classifier            ■■■■■■■■■■■■■■■■|··· 0.818
```

### 各方案的结构与诊断

- `tfidf_logistic_regression`：TfidfTextVectorizer → LogisticRegressionModel。诊断：无
- `tfidf_lightgbm_classifier`：TfidfTextVectorizer → 生成:LightGBMClassifier。诊断：无

## 4. 最优方案：tfidf_logistic_regression

通过 TF-IDF 特征提取与逻辑回归模型的组合，针对类别不平衡问题设置 balanced 权重以保障宏平均 F1 指标。

| # | 步骤 | 算子 | 参数 | 选择理由 |
| --- | --- | --- | --- | --- |
| 1 | vectorizer | TfidfTextVectorizer@1 | column=text, analyzer=char_wb, ngram_range=(1, 2), max_features=20000 | 使用 TF-IDF 处理文本，字符级 n-gram 对中文客服工单表现稳健。 |
| 2 | model | LogisticRegressionModel@1 | C=1.0, class_weight=balanced, max_iter=1000 | 逻辑回归在文本特征上是强基线，class_weight='balanced' 有效应对不平衡数据。 |

## 5. 能力生成与修复

| 生成算子 | 类型 | 用途 | 参考的历史实现 | 修复轮数 | 注册到能力库 |
| --- | --- | --- | --- | --- | --- |
| LightGBMClassifier@1 | model | 利用梯度提升树处理文本向量后的分类任务，支持不平衡样本权重调整。 | - | 1 | 否 |

### 修复 LightGBMClassifier@1 第 1 轮：✅ 已修复

- 诊断：`E_RUNTIME`（sandbox）AttributeError: 'csr_matrix' object has no attribute 'select_dtypes' (at predict_proba:61)
- 提示：-
- 参考经验：2 条

```diff
--- 修复前
+++ 修复后
@@ -14,3 +14,3 @@
         description="利用梯度提升树进行多分类，支持类别权重平衡优化，适用于文本向量化后的特征输入。",
-        conditions=["输入为数值特征矩阵", "支持类别不平衡"],
+        conditions=["输入为数值特征矩阵或稀疏矩阵", "支持类别不平衡"],
         dependencies=["lightgbm", "numpy", "pandas", "scikit-learn"],
@@ -30,6 +30,14 @@
 
+    def _ensure_numeric(self, X):
+        """处理输入数据，兼容 DataFrame 和稀疏矩阵，并处理缺失值/无穷值"""
+        if isinstance(X, pd.DataFrame):
+            # 移除所有非数值列，将非数值转为 numeric，无效值转为 NaN
+            X = X.apply(pd.to_numeric, errors='coerce')
+            # 填补缺失值和无穷值
+            X = X.fillna(0).replace([np.inf, -np.inf], 0)
+        return X
+
     def fit(self, X, y):
-        # 确保输入为 DataFrame 且处理潜在的类型问题
-        if isinstance(X, pd.DataFrame):
-            X = X.select_dtypes(include=[np.number])
+        # 训练阶段处理输入
+        X_train = self._ensure_numeric(X)
         
@@ -45,3 +53,3 @@
         
-        self.model_.fit(X, y)
+        self.model_.fit(X_train, y)
         self.classes_ = self.model_.classes_
@@ -50,15 +58,13 @@
     def predict(self, X):
-        X = X.copy()
-        # 处理可能的缺失值：LightGBM 原生支持缺失值，无需手动填充
-        # 对输入进行类型校验，确保不包含不支持的非数值类型
-        X = X.select_dtypes(include=[np.number])
+        # 预测时复制以保持接口规范
+        X_test = self._ensure_numeric(X.copy())
         
-        # 预测时返回 shape (n,)
-        return self.model_.predict(X)
+        # LightGBM 原生支持稀疏矩阵和缺失值，直接预测
+        return self.model_.predict(X_test)
 
     def predict_proba(self, X):
-        X = X.copy()
-        X = X.select_dtypes(include=[np.number])
+        # 预测时复制以保持接口规范
+        X_test = self._ensure_numeric(X.copy())
         
         # 返回 shape (n, n_classes) 的 numpy 数组
-        return self.model_.predict_proba(X)
+        return self.model_.predict_proba(X_test)
```

## 6. 验证机制

| 级别 | 检查内容 | 本次配置 |
| --- | --- | --- |
| 1 静态检查 | 语法、import 白名单、禁止危险调用、接口与 meta 一致性 | 仅针对生成的算子 |
| 2 沙箱执行 | 独立子进程，限制超时与内存，清除含 KEY/TOKEN 的环境变量 | 超时 180 s，内存 8192 MB |
| 3 契约检查 | 不修改输入、行数一致、不引入 NaN/Inf、概率合法、能处理未见类别与缺失值 | 冒烟样本 300 行 |
| 4 指标检查 | 阈值、种子间稳定性、疑似泄漏、耗时与内存 | 5 折 × 种子 [42, 7, 2024]；std 上限 0.03 |

最优方案资源消耗：单次训练 0.118 s，预测 60.1 ms / 千行，峰值内存 241 MB。

修复前的 1 次验证结果保留在 state.json 的 history 中。

## 7. 知识沉淀

- 回写验证记录 ValidationRun：2 条
- 注册到能力库的新算子：无
- 新提炼的经验：移除对 pandas 特定 API 的依赖以增强输入类型兼容性（E_RUNTIME）
- 本次提供的经验：7 条（累计检索命中，不代表采纳）

## 附录 A. 生成的算法代码（pipeline.py）

```python
"""把客服工单自动分到网络故障、账单争议、套餐变更、销户投诉四类，宏平均 F1 不低 —— 自动生成的算法流水线

需求：把客服工单自动分到网络故障、账单争议、套餐变更、销户投诉四类，宏平均 F1 不低于 0.8
方案：tfidf_logistic_regression
设计依据：通过 TF-IDF 特征提取与逻辑回归模型的组合，针对类别不平衡问题设置 balanced 权重以保障宏平均 F1 指标。
主指标：f1_macro >= 0.8

由 AlgoForge 自动生成。用法：
    python pipeline.py --data <train.csv>                      # 训练并保存 model.joblib
    python pipeline.py --predict <new.csv> --model model.joblib  # 预测
"""

import argparse

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from factory.operators.models import LogisticRegressionModel
from factory.operators.text import TfidfTextVectorizer


TARGET = 'label'


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("0_vectorizer", TfidfTextVectorizer(column='text', analyzer='char_wb', ngram_range=(1, 2), max_features=20000)),  # 使用 TF-IDF 处理文本，字符级 n-gram 对中文客服工单表现稳健。
        ("1_model", LogisticRegressionModel(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)),  # 逻辑回归在文本特征上是强基线，class_weight='balanced' 有效应对不平衡数据。
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

各阶段耗时：PLAN 4.2s，BUILD 33.8s，VERIFY 16.3s，ROUTE 0.0s，REPAIR 14.4s，CURATE 2.8s

```
[PLAN] 场景 新场景；画像标签 ['id_columns', 'text_input', 'imbalanced']
[PLAN] TaskSpec：multiclass_classification，目标 label，主指标 f1_macro >= 0.8
[PLAN] 候选方案：['tfidf_logistic_regression(TfidfTextVectorizer → LogisticRegressionModel)', 'tfidf_lightgbm_classifier(TfidfTextVectorizer → *LightGBMClassifier)']
[PLAN] 完成，用时 4.2s → BUILD
[BUILD] 生成算子 LightGBMClassifier@1：利用梯度提升树处理文本向量后的分类任务，支持不平衡样本权重调整。
[BUILD] 完成，用时 33.8s → VERIFY
[VERIFY] tfidf_logistic_regression：通过，f1_macro=0.8244
[VERIFY] tfidf_lightgbm_classifier：未通过，f1_macro=nan；E_RUNTIME AttributeError: 'csr_matrix' object has no attribute 'select_dtypes' (at predict_proba:61)
[VERIFY] 完成，用时 5.8s → ROUTE
[ROUTE] 完成，用时 0.0s → REPAIR
[REPAIR] 修复 LightGBMClassifier@1 第 1 轮：E_RUNTIME AttributeError: 'csr_matrix' object has no attribute 'select_dtypes' (at predict，参考经验：['数值列含空白字符导致类型转换失败', '预测阶段出现未见类别导致编码报错']
[REPAIR] 完成，用时 14.4s → VERIFY
[VERIFY] tfidf_lightgbm_classifier：通过，f1_macro=0.8184
[VERIFY] 完成，用时 10.5s → ROUTE
[ROUTE] 完成，用时 0.0s → CURATE
[CURATE] 写入 KG：2 条 ValidationRun；注册新算子 []；新增经验 1 条
[CURATE] 完成，用时 2.8s → DONE
```

### LLM 调用来源

| 阶段 | 请求哈希 | 来源 | 本次耗时(s) | 录制耗时(s) | tokens |
| --- | --- | --- | --- | --- | --- |
| plan_spec | 00e68fcad6e9a2d1 | replay | 0.000 | - | 未提供 |
| plan | 6fc3b554906f78ac | remote | 4.189 | - | 4869 |
| synthesize | d3b6ebcf4bf82981 | remote | 33.803 | - | 2123 |
| repair | 2fe5f52f685c21a9 | remote | 14.375 | - | 2679 |
| lesson | 92482b5b70296e45 | remote | 2.768 | - | 1986 |
