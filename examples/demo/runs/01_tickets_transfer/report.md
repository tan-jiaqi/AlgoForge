# AlgoForge 验证报告

| 项目 | 内容 |
| --- | --- |
| 需求 | 把客服工单自动分到网络故障、账单争议、套餐变更、销户投诉四类，宏平均 F1 不低于 0.8 |
| 数据 | data/tickets.csv |
| 运行 ID | 01_tickets_transfer |
| LLM | gemini-3.5-flash |
| 状态 | ✅ 成功 |
| 最优方案 | char_ngram_tfidf_lr_tuned |
| 总耗时 | 8.0 s |

共验证 2 个候选方案，1 个通过；重新规划 0 次，修复 1 轮。最优方案 `char_ngram_tfidf_lr_tuned` 的 f1_macro = 0.820（要求 ≥ 0.8）。

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

- **样本不平衡导致默认阈值下召回率偏低**（M_THRESHOLD）：设置 class_weight='balanced' 或在验证集上调整分类阈值以提升召回率
- **样本不平衡导致默认模型召回率偏低**（M_THRESHOLD）：启用平衡类权重（如 class_weight='balanced' 或 BalancedLightGBMClassifier）以提升召回率
- **LightGBM 调参及不平衡处理**：设置 num_leaves 15~63, learning_rate 0.03~0.1, n_estimators 200~600；不平衡时设置 class_weight='balanced'；树模型无需标准化。
- **数值列含空白字符导致类型转换失败**（E_RUNTIME）：先 str.strip()，再 pd.to_numeric(errors='coerce')，缺失值填补为 0
- **未剔除客户 ID 导致 One-Hot 编码后维度暴涨与内存溢出**（E_TIMEOUT）：建模前剔除 ID 等标识列，并在编码器中设置最大类别数

## 3. 候选方案比较

排序规则：先看是否通过全部检查，再比主指标，再比次要指标之和，最后训练耗时短者优先。指标为多随机种子 × 分层 K 折交叉验证的均值 ± 种子间标准差。

| 排名 | 方案 | 结果 | f1_macro | accuracy | recall_macro | precision_macro | 单次训练 (s) | 峰值内存 (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ★ char_ngram_tfidf_lr_tuned | 通过 | 0.820 ± 0.004 | 0.833 ± 0.004 | 0.829 ± 0.004 | 0.815 ± 0.004 | 0.146 | 244 |
| 2 | text_preprocess_tfidf_lr | 未通过 | 0.422 ± 0.004 | 0.426 ± 0.003 | 0.469 ± 0.003 | 0.510 ± 0.009 | 0.092 | 248 |

```
f1_macro（| 为阈值 0.8）
char_ngram_tfidf_lr_tuned            ■■■■■■■■■■■■■■■■|··· 0.820
text_preprocess_tfidf_lr             ■■■■■■■■········|··· 0.422
```

### 各方案的结构与诊断

- `char_ngram_tfidf_lr_tuned`：TfidfTextVectorizer → LogisticRegressionModel。诊断：无
- `text_preprocess_tfidf_lr`：生成:ChineseTextPreprocessor → TfidfTextVectorizer → LogisticRegressionModel。诊断：`M_THRESHOLD` f1_macro = 0.4216 低于阈值 0.8

## 4. 最优方案：char_ngram_tfidf_lr_tuned

方案二避免了显式分词，通过扩大字符级 n-gram 范围至 [1, 3] 并增加最大特征数来捕捉更细粒度的文本模式，同时微调正则化参数以优化泛化性能。

| # | 步骤 | 算子 | 参数 | 选择理由 |
| --- | --- | --- | --- | --- |
| 1 | vectorizer | TfidfTextVectorizer@1 | column=text, analyzer=char_wb, ngram_range=(1, 3), max_features=30000 | 采用字符级 n-gram（1-3元），无需分词即可捕获丰富的局部文本结构，对拼写变体和中文词组边界更稳健。 |
| 2 | model | LogisticRegressionModel@1 | class_weight=balanced, C=0.5 | 使用逻辑回归，微调正则化强度 C=0.5 以防止高维特征过拟合，并保持 class_weight='balanced'。 |

## 5. 能力生成与修复

| 生成算子 | 类型 | 用途 | 参考的历史实现 | 修复轮数 | 注册到能力库 |
| --- | --- | --- | --- | --- | --- |
| ChineseTextPreprocessor@1 | feature | 对中文文本进行分词、去除停用词和标点符号等预处理，以提升后续 TF-IDF 的特征质量 | - | 1 | 否 |

### 修复 ChineseTextPreprocessor@1 第 1 轮：✅ 已修复

- 诊断：`E_UNSAFE`（static）禁止调用 getattr()
- 提示：沙箱不允许 ['__import__', 'breakpoint', 'compile', 'delattr', 'eval', 'exec', 'exit', 'getattr', 'globals', 'input', 'locals', 'open', 'quit', 'setattr', 'vars']；访问可选属性请用 try/except AttributeError
- 参考经验：0 条

```diff
--- 修复前
+++ 修复后
@@ -66,4 +66,5 @@
         X = X.copy()
-        cols = getattr(self, "text_columns_", [])
-        if not cols:
+        try:
+            cols = self.text_columns_
+        except AttributeError:
             if self.text_columns is not None:
```

## 6. 验证机制

| 级别 | 检查内容 | 本次配置 |
| --- | --- | --- |
| 1 静态检查 | 语法、import 白名单、禁止危险调用、接口与 meta 一致性 | 仅针对生成的算子 |
| 2 沙箱执行 | 独立子进程，限制超时与内存，清除含 KEY/TOKEN 的环境变量 | 超时 180 s，内存 8192 MB |
| 3 契约检查 | 不修改输入、行数一致、不引入 NaN/Inf、概率合法、能处理未见类别与缺失值 | 冒烟样本 300 行 |
| 4 指标检查 | 阈值、种子间稳定性、疑似泄漏、耗时与内存 | 5 折 × 种子 [42, 7, 2024]；std 上限 0.03 |

最优方案资源消耗：单次训练 0.146 s，预测 72.3 ms / 千行，峰值内存 244 MB。

修复前的 1 次验证结果保留在 state.json 的 history 中。

## 7. 知识沉淀

- 回写验证记录 ValidationRun：2 条
- 注册到能力库的新算子：无
- 新提炼的经验：避免在沙箱中使用 getattr 等受限反射函数，访问可选属性应使用 try/except AttributeError（E_UNSAFE）
- 本次使用的经验：7 条（命中次数已累计）

## 附录 A. 生成的算法代码（pipeline.py）

```python
"""把客服工单自动分到网络故障、账单争议、套餐变更、销户投诉四类，宏平均 F1 不低 —— 自动生成的算法流水线

需求：把客服工单自动分到网络故障、账单争议、套餐变更、销户投诉四类，宏平均 F1 不低于 0.8
方案：char_ngram_tfidf_lr_tuned
设计依据：方案二避免了显式分词，通过扩大字符级 n-gram 范围至 [1, 3] 并增加最大特征数来捕捉更细粒度的文本模式，同时微调正则化参数以优化泛化性能。
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
        ("0_vectorizer", TfidfTextVectorizer(column='text', analyzer='char_wb', ngram_range=(1, 3), max_features=30000)),  # 采用字符级 n-gram（1-3元），无需分词即可捕获丰富的局部文本结构，对拼写变体和中文词组边界更稳健。
        ("1_model", LogisticRegressionModel(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)),  # 使用逻辑回归，微调正则化强度 C=0.5 以防止高维特征过拟合，并保持 class_weight='balanced'。
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

各阶段耗时：PLAN 0.1s，BUILD 0.0s，VERIFY 8.0s，ROUTE 0.0s，REPAIR 0.0s，CURATE 0.0s

```
[PLAN] 场景 新场景；画像标签 ['id_columns', 'text_input', 'imbalanced']
[PLAN] TaskSpec：multiclass_classification，目标 label，主指标 f1_macro >= 0.8
[PLAN] 候选方案：['text_preprocess_tfidf_lr(*ChineseTextPreprocessor → TfidfTextVectorizer → LogisticRegressionModel)', 'char_ngram_tfidf_lr_tuned(TfidfTextVectorizer → LogisticRegressionModel)']
[PLAN] 完成，用时 0.0s → BUILD
[BUILD] 生成算子 ChineseTextPreprocessor@1：对中文文本进行分词、去除停用词和标点符号等预处理，以提升后续 TF-IDF 的特征质量
[BUILD] 完成，用时 0.0s → VERIFY
[VERIFY] text_preprocess_tfidf_lr：未通过，f1_macro=nan；E_UNSAFE 禁止调用 getattr()
[VERIFY] char_ngram_tfidf_lr_tuned：通过，f1_macro=0.8200
[VERIFY] 完成，用时 4.3s → ROUTE
[ROUTE] 完成，用时 0.0s → REPAIR
[REPAIR] 修复 ChineseTextPreprocessor@1 第 1 轮：E_UNSAFE 禁止调用 getattr()
[REPAIR] 完成，用时 0.0s → VERIFY
[VERIFY] text_preprocess_tfidf_lr：未通过，f1_macro=0.4216；M_THRESHOLD f1_macro = 0.4216 低于阈值 0.8
[VERIFY] 完成，用时 3.6s → ROUTE
[ROUTE] 完成，用时 0.0s → CURATE
[CURATE] 写入 KG：2 条 ValidationRun；注册新算子 []；新增经验 1 条
[CURATE] 完成，用时 0.0s → DONE
```
