# 能力知识图谱：从本体到存储

能力知识图谱（Capability KG）是整个系统的"长期记忆"。它要回答 Agent 在工作中真正会问的四个问题：

| Agent 的问题 | 谁在问 | 图谱里的答案 |
| --- | --- | --- |
| 这一步有没有现成的算子可以用？ | Planner | `Operator`（`executable`）+ `SUITABLE_WHEN → Condition` |
| 它在这个场景上表现怎么样？ | Planner、Beam 搜索 | `VALIDATED_BY → ValidationRun → RUN_ON → Scenario` |
| 这个需求的达标线是多少？ | Planner | `Scenario → EVALUATED_BY{threshold} → Metric` |
| 这类错误以前是怎么修好的？ | Synthesizer | `Lesson`（`error_code`）`→ APPLIES_TO → Operator / Condition` |

围绕这些问题，图谱的构建分为四步：

```
 ① 本体构建            ② 知识抽取                 ③ 知识融合                   ④ 知识存储
 定义"能力"该怎么描述 ─▶ 从代码与文档中抽出实例 ─▶ 对齐、去重、消解冲突 ─▶ 持久化、检索、回写
 (kg.py: NODE_TYPES)    (extract.py)              (kg.py: add_* / tidy)       (JSON / GraphML / HTML)
                                                         ▲                          │
                                                         └── 运行时回写（Curator）◀──┘
```

第 ③ 步不只在离线建图时发生：开启回写且运行进入 CURATE 时，新算子、验证记录和经验会通过写入接口进入已有图谱。部分去重规则在该接口复用，来源优先级和算子名称对齐等规则则分别在抽取或规划阶段执行。

---

## ① 本体构建：定义"能力"该怎么描述

### 设计思路

本体从 Agent 的决策和溯源需要反推：记录适用条件、历史表现、来源与版本关系。当前是人工定义的轻量属性图 schema，没有 OWL 本体或自动逻辑推理引擎。几个关键取舍：

1. **能力的单位是算子。** 把历史实现拆分、适配为算子（`Operator`），便于记录适用条件、依赖与历史表现，也便于逐步执行时定位部分错误。
2. **区分"知识"与"能力"。** 从历史代码抽取的函数是**知识**（`executable=false`，保留源码供参考），注册进算子库、可直接组装进 Pipeline 的才是**能力**（`executable=true`）。知识经过 Synthesizer 改写、验证通过后转化为能力，用 `DERIVED_FROM` 串成版本谱系。
3. **阈值属于"场景 × 指标"，不属于指标。** 同一个 recall，在流失场景要求 ≥ 0.75，在别的场景可能是 0.6，所以阈值放在 `EVALUATED_BY` 边上。
4. **经验与验证器共用一套诊断码。** `Lesson.error_code` 与 Verifier 的诊断码使用同一枚举。修复时按诊断码精确过滤类别，再按报错文本相关度排序取前几条；诊断码本身不能唯一确定修复经验。
5. **用受控词表约束条件标签。** `Condition` 使用词表 `CONDITION_TAGS`（19 个，如 `imbalanced`、`string_typed_numeric`）。抽取时 LLM 优先从词表中选，数据画像 `profile.py` 也产出同一套标签，检索时两边才能对得上。
6. **保留来源与版本线索。** 抽取知识通过 `Source` 记录函数或文档路径，运行经验关联验证记录，新算子可关联参考版本。内置算子不一定有 `Source`；当前也没有逐条核验生成理由或保存文档证据片段的流程。

### 节点类型（8 类）

节点 id 格式为 `<类型>:<键>`，例如 `Operator:LightGBMModel@1`、`Condition:imbalanced`。

| 类型 | 键 | 主要属性 | 回答的问题 |
| --- | --- | --- | --- |
| `Scenario` | 场景名 | `title` `task_type` `modality` `dataset` `target_column` `constraints` | 业务场景是什么、有哪些约束（如 `recall_first`） |
| `Operator` | `名称@版本` | `kind` `modalities` `description` `inputs` `outputs` `params` `origin` `executable` `code` | 有哪些能力单元；`origin` 取 builtin / extracted / synthesized |
| `Metric` | 规范化的指标名 | `name` `description` | 用什么评价 |
| `Condition` | 条件标签 | `description` | 在什么情况下适用 |
| `Dependency` | 包名 | `package` | 运行需要什么环境 |
| `ValidationRun` | 内容哈希 / 运行 id | `metrics` `passed` `origin` `note` `pipeline` `created_at` | 历史上验证过什么、结果如何；`origin` 为 history（文档中的历史实验）或 factory（本系统的验证） |
| `Lesson` | 标题哈希 | `title` `trigger` `fix` `error_code` `hits` `aliases` | 踩过什么坑、怎么修；`hits` 为被检索使用的次数 |
| `Source` | 路径 | `kind`（code / doc） `path` `title` | 知识从哪里来 |

### 边类型（9 类）

| 关系 | 起点 → 终点 | 边属性 | 含义 |
| --- | --- | --- | --- |
| `SUITABLE_WHEN` | Operator → Condition | | 算子在该条件下适用 |
| `DEPENDS_ON` | Operator → Dependency | | 算子依赖的包 |
| `EVALUATED_BY` | Scenario → Metric | `threshold` `primary` | 场景的评价指标与达标阈值 |
| `VALIDATED_BY` | Operator → ValidationRun | 各指标值 | 算子参与了这次验证 |
| `RUN_ON` | ValidationRun → Scenario | | 验证所属场景（按场景统计算子历史表现） |
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

本体定义在 `factory/kg.py`（`NODE_TYPES`、`EDGE_TYPES`、`CONDITION_TAGS`）。写入接口检查节点类型名、关系名及端点是否存在；尚未强制检查关系两端的实体类型是否符合上表。直接从 JSON 加载时也未执行完整 schema 校验。

---

## ② 知识抽取：从代码与文档中抽出实例

抽取的输入由 `data/materials/manifest.yaml` 声明：一个历史代码仓库 `churn_v1`（9 个函数）和 4 篇业务文档（需求说明、实验记录、失败复盘、调参经验），实现见 `factory/extract.py`。

### 设计思路：确定性的事交给代码，需要理解的事交给 LLM

| 来源 | 确定性部分（代码） | 理解部分（LLM） | 输出 schema |
| --- | --- | --- | --- |
| 历史代码 | `ast` 解析函数签名、docstring、源码、import 依赖 | 归纳成能力卡片：是算子 / 指标 / 工具函数？叫什么、做什么、适用条件、关键参数 | `CapabilityCard` |
| 业务文档 | — | 抽取评价指标与阈值、业务约束、历史实验（流水线 + 指标 + 是否有效）、经验教训 | `DocKnowledge` |
| 运行过程 | Verifier 的诊断、修复前后的代码 diff | Curator 把"诊断 → 修复"提炼成通用经验 | `LessonRecord` |

函数签名和源码由 AST 提取；依赖结合 import、源码中出现的别名和有限包名映射推断，不能覆盖完整依赖闭包。当前扫描公开顶层函数，暂未解析类方法或跨文件调用链。能力语义与文档经验由 LLM 归纳。

### 在抽取阶段就约束输出

- **词表约束**：prompt 中给出 `CONDITION_TAGS` 词表与诊断码说明，LLM 优先从中选择条件标签与 `error_code`；
- **名字约束**：抽取文档时，prompt 中列出已知算子名（内置 + 从代码抽取的），要求 `experiments.pipeline` 与 `lessons.applies_to` 必须使用这些名字。这一步是在源头做实体对齐，为融合减负；
- **schema 校验**：抽取的 JSON 输出经过 pydantic 校验，不合格时带错误重试；
- **失败隔离**：并发的函数归纳和文档抽取任务失败时记录到 `errors`，其他任务继续；前置的文件读取、AST 解析等异常尚不在这层隔离范围内。

### 抽取结果

原始结果保存在 `kg/extraction.json`（未经融合，便于对照）：

| 来源 | 抽取得到 |
| --- | --- |
| 9 个函数 | 7 个算子（如 `TotalChargesCleaner`、`TenureBucketizer`、`BalancedLightGBMClassifier`）、1 个指标（`RecallAtTopK`）、1 个工具函数（日志，不入图） |
| 4 篇文档 | 场景指标与约束、5 条历史实验、11 条经验 |

示例：一个历史函数被归纳为能力卡片（录制的真实 LLM 响应）

```json
{"function": "clean_total_charges", "capability_type": "operator", "name": "TotalChargesCleaner", "kind": "imputer",
 "description": "将包含空白字符串的数值列安全转换为浮点型，并将缺失及空白值填充为0.0",
 "conditions": ["string_typed_numeric", "has_missing_values", "numeric_features"], "params": {"col": "total_charges"}}
```

---

## ③ 知识融合：对齐、去重、消解冲突

多来源抽取可能带来**别名、重复和冲突**。目前采用规则融合：指标规范化、经验去重等在 `CapabilityKG.add_*` / `tidy` 中执行，需求文档优先级在 `extract.build_kg` 中执行，生成算子名称与谱系对齐在 Planner 中执行。

| 问题 | 例子 | 融合规则 |
| --- | --- | --- |
| 实体链接 | 文档里写 `LightGBMModel`、`imbalanced`，需要连到图中已有节点 | 按名字解析到已有算子（同名多版本取可执行的最新版）、条件或场景；解析不到的引用丢弃，不凭空造节点 |
| 指标同义 | 旧代码的 `RecallAtTopK(k=0.2)`、文档的 `recall@20%`、需求的 `recall_at_20` | `canonical_metric` 按参数规范化为同一个节点 `Metric:recall_at_20`；`auc` → `roc_auc` |
| 经验重复 | 失败复盘与实验记录分别抽出"样本不平衡导致召回率偏低" | **诊断码相同且标题相似度 ≥ 0.5**（词元 Jaccard）视为同一条：合并适用对象与来源，被合并的标题记入 `aliases`。11 条经验融合为 10 条 |
| 来源可信度不同 | 调参笔记里随口提到的阈值、实验记录里的临时指标 | 场景的指标阈值与业务约束**只以需求文档为准** |
| 结论被推翻 | 实验记录中 AUC 0.97 的方案后来被发现数据泄漏 | 保留该实验（它本身是有价值的教训），但记为 `passed=false` 并注明作废原因，**不计入历史验证分** |
| 抽取的算子与内置算子重名 | 历史代码与内置库都有某能力 | 内置算子优先，抽取的同名算子跳过 |
| 生成算子接入版本谱系 | LLM 生成了 `IdColumnDropper`，而图中已有 `IdColumnDropper@1` | 名字忽略大小写对齐到已有算子，版本号顺延（抽取的记为 v1，生成的从 v2 起），用 `DERIVED_FROM` 指向旧版本 |
| 运行时回写 | 每次运行产生新的验证记录与经验 | `ValidationRun`（origin=factory）与历史实验并存于同一场景下；新经验同样按"诊断码 + 标题相似度"合并 |

已有图谱可以用 `python -m factory.cli kg tidy` 按同样的规则整理（幂等，可重复执行）。

这些规则尚未覆盖通用的语义实体消歧或矛盾消解。同一 ID 的普通属性或同一关系重复写入时更新属性；不同需求文档之间的矛盾没有独立裁决机制。`extract` 会重建初始图谱；Curator 在已有图上增量回写，但新批材料与已有图的通用增量融合尚未实现。

示例：合并后的经验节点

```json
{"type": "Lesson", "title": "样本不平衡导致默认模型召回率偏低", "error_code": "M_THRESHOLD",
 "fix": "class_weight='balanced'，或在验证集上调阈值以满足最低召回",
 "aliases": ["样本不平衡导致默认阈值下召回率偏低"]}
```

---

## ④ 知识存储：持久化、检索与回写

### 存储形式

| 文件 | 用途 |
| --- | --- |
| `kg/capability_kg.json` | 主存储（`schema_version: 1`），节点与边的完整属性；运行时加载为 NetworkX `MultiDiGraph` |
| `kg/capability_kg.graphml` | 图交换格式，可供 Gephi 等工具读取；迁移到 Neo4j 还需选择导入工具并映射属性；列表和字典属性序列化为 JSON 字符串 |
| `kg/capability_kg.html` / `_full.html` | pyvis 可视化：概览（场景 / 指标 / 算子 / 经验四类主干节点）与完整视图 |
| `kg/extraction.json` | 融合前的原始抽取结果 |
| `kg/operators/<Name>_v<版本>.py` | 生成并注册的算子源码；下次启动时重新做静态检查后加载进算子库 |

选择 NetworkX + JSON 而不是 Neo4j：原型规模（百级节点）下无需部署数据库，文件可以直接进 Git、做 diff；GraphML 保留了迁移到图数据库的通路。

保存时整体写出 JSON，加载时恢复 NetworkX 图；当前没有事务、并发写入控制或 schema 版本迁移。HTML 是展示产物，JSON 是主存储。

当前基础图谱规模：14 个算子（7 个内置 + 7 个抽取）、10 条经验、5 条历史实验、19 个条件、3 个指标、13 个出处。演示 3 个任务后（`examples/demo/kg/`）：算子 14 → 17、验证记录 5 → 12、场景 1 → 2、经验 10 → 11。

### 检索：存储为谁服务

`CapabilityKG.search_operators()`：

1. **结构化过滤**：按 `kind`、`modality`、`executable` 过滤；
2. **打分**：`0.4 × 文本相关度 + 0.3 × 条件重合度 + 0.3 × 历史验证分`，三项都归一化到 [0, 1]：
   - 文本相关度：对算子名、描述和条件标签做 BM25+ 打分；中文按单字加二元组切分，英文拆分 CamelCase；
   - 条件重合度：任务期望的条件（数据画像标签 + 业务约束）中，被算子 `SUITABLE_WHEN` 覆盖的比例；
   - 历史验证分：算子在指定场景、指定指标上**通过的**验证记录的平均值，作废的实验不计入；
3. **结果带证据**：每条结果都带分项得分和证据（"匹配条件：…""历史验证 n 次，平均 recall …"），Planner 用它们写选择理由，报告里也能看到。

`search_lessons()` 先按诊断码和数据模态过滤（经验的模态取其适用算子与场景的模态；未关联这类对象的经验视为通用），再按文本相关度与适用条件是否命中加权排序；被提供给 Agent 的经验 `hits` 加一，不表示模型实际采纳。

`metric_evidence()` 为 Beam 搜索提供某算子在某场景上所有可信的指标观测，用来估计子方案的先验。

`metric_records()` 使用相同过滤规则，返回验证节点 ID、指标名、数值和通过状态。Planner 上下文的 `evidence_catalog` 和搜索记录的 `observations` 保留这些来源，便于核对具体历史记录如何进入计算。规划的 `evidence_ids` 检查引用可见性与结构化适用范围；是否改变决策另由[消融及固定候选池干预](../examples/evaluation/SUMMARY.md)观察。

### 回写：图谱如何"长大"

每次运行结束，Curator 把三类知识写回（细节见 [agents_io.md](agents_io.md)）：

- 最优且通过的方案中用到的新生成算子 → 可执行 `Operator`（有参考版本时带 `DERIVED_FROM`）+ 源码文件；
- 每个候选方案的最终验证结果（包括失败的）→ `ValidationRun`；
- 成功修复的"诊断 → 修复"→ `Lesson`（`origin=factory`）。

后续任务按条件过滤与排名检索这些知识。演示任务 03 复用了任务 02 注册的 `TotalChargesCleaner@2` 与 `TenureBucketizer@2`。

---

## 附：节点示例（摘自 `kg/capability_kg.json`）

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

## 附：命令

```bash
python -m factory.cli extract --llm replay      # 用录制好的 LLM 响应离线重建 KG（默认）
python -m factory.cli extract --llm record      # 调用真实 API 重新抽取并录制
python -m factory.cli kg stats                  # 统计节点与边
python -m factory.cli kg tidy                   # 按融合规则整理已有图谱
python -m factory.cli kg viz                    # 生成概览与完整视图 HTML（--focus <节点 id> 只看邻域）
python -m factory.cli kg search "处理不平衡的可解释模型" --kind model -c imbalanced -c interpretable \
    --scenario telecom_churn --metric recall
python -m factory.cli kg lessons "could not convert string to float" --code E_RUNTIME
```
