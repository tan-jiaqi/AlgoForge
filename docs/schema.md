# 能力知识图谱 Schema

能力知识图谱（Capability KG）记录"有哪些算法能力、什么时候适用、历史表现如何、踩过哪些坑"。Planner 用它决定复用（Reuse）哪个算子还是生成（Synthesize）新算子；Synthesizer 修复代码时也从中检索经验。

- 实现：`factory/kg.py`，底层为 NetworkX `MultiDiGraph`
- 存储：`kg/capability_kg.json` 为主存储；`kg/capability_kg.graphml` 供 Gephi 等工具导入，列表和字典类属性在其中序列化为 JSON 字符串
- 原始抽取结果：`kg/extraction.json`，即 LLM 输出的能力卡片和文档知识，未经建图
- 节点 id 格式为 `<类型>:<键>`，例如 `Operator:LightGBMModel@1`、`Condition:imbalanced`

## 节点类型

| 类型 | 键 | 主要属性 | 说明 |
| --- | --- | --- | --- |
| `Scenario` | 场景名 | `title` `task_type` `modality` `dataset` `target_column` `constraints` | 业务场景。约束（如 `recall_first`）只从需求文档抽取 |
| `Operator` | `名称@版本` | `kind` `modalities` `description` `inputs` `outputs` `params` `origin` `executable` `code` | 算法能力单元。`origin` 取 builtin / extracted / synthesized |
| `Metric` | 指标名 | `name` `description` | 评价指标。阈值记在 `EVALUATED_BY` 边上，因为同一指标在不同场景的阈值不同 |
| `Condition` | 条件标签 | `description` | 适用条件，使用受控词表 `CONDITION_TAGS`，允许扩展 |
| `Dependency` | 包名 | `package` | 运行依赖，由 AST 从 import 中解析得到 |
| `ValidationRun` | 内容哈希 | `metrics` `passed` `origin` `note` `pipeline` `created_at` | 一次验证：历史实验（origin=history）或本系统的验证结果 |
| `Lesson` | 标题哈希 | `title` `trigger` `fix` `error_code` `hits` | 可复用经验。`error_code` 对应验证器的诊断码，`hits` 记录被检索使用的次数 |
| `Source` | 路径 | `kind`（code/doc） `path` `title` | 知识出处，用于溯源 |

`executable` 区分两类算子：
- `true`：已注册到 `OperatorRegistry`，可以直接组装进 Pipeline 复用。
- `false`：从历史代码中抽取的参考实现，带有原始源码 `code`。需要由 Synthesizer 按算子接口改写，验证通过后注册为新版本才能使用。

## 边类型

| 关系 | 起点 → 终点 | 边属性 | 含义 |
| --- | --- | --- | --- |
| `SUITABLE_WHEN` | Operator → Condition | | 算子在该条件下适用 |
| `DEPENDS_ON` | Operator → Dependency | | 算子依赖的包 |
| `EVALUATED_BY` | Scenario → Metric | `threshold` `primary` | 场景的评价指标与达标阈值 |
| `VALIDATED_BY` | Operator → ValidationRun | 各指标值 | 算子参与了这次验证 |
| `RUN_ON` | ValidationRun → Scenario | | 验证所属场景 |
| `LEARNED_FROM` | Lesson → Source / ValidationRun | | 经验从哪里得来 |
| `APPLIES_TO` | Lesson → Operator / Condition / Scenario | | 经验适用的对象 |
| `EXTRACTED_FROM` | Operator / ValidationRun / Scenario / Metric → Source | | 抽取出处 |
| `DERIVED_FROM` | Operator → Operator | | 版本谱系：新版本指向它所派生的旧版本 |

```
                 EVALUATED_BY{threshold}
   Scenario ─────────────────────────────▶ Metric
      ▲  │ EXTRACTED_FROM
RUN_ON│  └──────────────▶ Source ◀──────────────┐
      │                     ▲                    │ LEARNED_FROM
ValidationRun ◀── VALIDATED_BY ── Operator       Lesson
                                   │  │  │         │ APPLIES_TO
                     SUITABLE_WHEN │  │  └ DEPENDS_ON ▶ Dependency
                                   ▼  └ DERIVED_FROM ▶ Operator(旧版本)
                               Condition ◀───────────┘
```

## 检索

`CapabilityKG.search_operators()` 的检索流程：

1. **结构化过滤**：按 `kind`、`modality`、`executable` 过滤。
2. **打分**：`0.4 × 文本相关度 + 0.3 × 条件重合度 + 0.3 × 历史验证分`，三项都归一化到 [0, 1]。
   - 文本相关度：对算子名、描述和条件标签做 BM25+ 打分。中文按单字加二元组切分，英文拆分 CamelCase。
   - 条件重合度：任务期望的条件中，被算子 `SUITABLE_WHEN` 覆盖的比例。
   - 历史验证分：算子在指定场景、指定指标上**通过的**验证记录的平均值。作废的实验（如数据泄漏）不计入。
3. **结果**：每条结果都带分项得分和证据（"匹配条件：…""历史验证 n 次，平均 recall …"），Planner 用它们说明选择理由。

`search_lessons()` 先按诊断码精确过滤，再按文本相关度排序。修复代码时，用验证器给出的诊断码和报错信息去检索。

## 示例（摘自 `kg/capability_kg.json`）

从历史代码抽取的算子，尚不可执行，保留了参考源码：

```json
{
  "id": "Operator:TotalChargesCleaner@1",
  "type": "Operator",
  "kind": "imputer",
  "description": "将包含空白字符串的数值列安全转换为浮点型，并将缺失及空白值填充为0.0",
  "params": {"col": "total_charges"},
  "origin": "extracted",
  "executable": false,
  "code": "def clean_total_charges(df, col=\"total_charges\"): ..."
}
```
它的边：`SUITABLE_WHEN → Condition:string_typed_numeric`，`DEPENDS_ON → Dependency:pandas`，`EXTRACTED_FROM → Source:churn_v1/cleaning.py::clean_total_charges`。

被作废的历史实验，不计入历史验证分：

```json
{
  "id": "ValidationRun:10b300d0",
  "type": "ValidationRun",
  "metrics": {"roc_auc": 0.97},
  "passed": false,
  "origin": "history",
  "note": "作废：全量数据目标编码泄漏标签，上线后 AUC 跌至 0.71",
  "pipeline": ["Operator:KFoldTargetEncoder@1", "Operator:LightGBMModel@1"]
}
```

从失败复盘中抽取的经验：

```json
{
  "id": "Lesson:0d5e3eb1",
  "type": "Lesson",
  "title": "预测阶段出现未见类别导致编码报错",
  "trigger": "线上/测试集出现未见类别，OneHotEncoder 抛出 ValueError: Found unknown categories",
  "fix": "编码器配置 handle_unknown='ignore' 或 infrequent_if_exist 忽略未知类别",
  "error_code": "E_RUNTIME"
}
```
它的边：`APPLIES_TO → Operator:OneHotCategoricalEncoder@1`、`APPLIES_TO → Condition:has_categorical_features`、`LEARNED_FROM → Source:docs/failure_cases.md`。

## 如何生成与查看

```bash
python -m factory.cli extract --llm replay      # 用录制好的 LLM 响应离线重建 KG（默认）
python -m factory.cli extract --llm record      # 调用真实 API 重新抽取并录制
python -m factory.cli kg stats
python -m factory.cli kg search "处理不平衡的可解释模型" --kind model -c imbalanced -c interpretable \
    --scenario telecom_churn --metric recall
python -m factory.cli kg lessons "could not convert string to float" --code E_RUNTIME
```
