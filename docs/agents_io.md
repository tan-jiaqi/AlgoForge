# Agent 工作原理：职责、输入输出与协作流程

本文以客户流失预测为例，说明各组件的**职责、输入、输出和代码位置**。正文介绍流程与示例，技术附录提供字段、消息格式和算子执行接口。

正文中的对象和方案为示意。真实 LLM 录制演示见 [examples/demo/](../examples/demo/)，真实调用评测见 [examples/evaluation/](../examples/evaluation/)，规则 mock 样例见 [examples/verification/](../examples/verification/)。

阅读导航：[Planner](#2-planner把需求变成候选执行方案) · [Synthesizer](#3-synthesizer生成缺失代码并按诊断修复) · [Verifier](#4-verifier运行方案给出指标和失败原因) · [Curator](#5-curator算子注册与知识回写) · [Workflow](#6-workflow阶段调度与诊断路由) · [算子接口与执行](#算子接口与执行) · [技术附录](#8-技术附录)。

## 1. 总览：任务处理流程

假设用户输入：

> 预测客户流失，需要可解释性，召回率不低于 0.75。

同时提供一个 CSV，目标列为 `churn`，其中 `total_charges` 含空白字符串，部分其他列也有缺失值。

| 组件 | 负责什么 | 主要输出 | 是否使用 LLM |
| --- | --- | --- | --- |
| Planner | 理解需求、检索知识、提出候选执行方案 | 任务规格 `TaskSpec`、候选方案 `PipelinePlan` 列表 | 理解需求与生成方案时使用；检索不使用 |
| Synthesizer | 为缺失能力生成源码，按诊断修改本次生成的代码 | 含源码的 `GeneratedOperator` | 生成和修复时使用 |
| Verifier | 检查源码、执行方案、验证接口契约和指标 | 验证报告 `ValidationReport` | 否 |
| Curator | 注册通过的新算子、记录验证结果、提炼修复经验 | 注册记录、验证记录、经验 | 仅提炼经验时使用 |
| Workflow | 调用上述组件，决定继续修复、重新规划还是结束 | 完整运行状态、流水线脚本和报告 | 调度本身不使用 |

```text
用户需求 + CSV
    ↓
Planner：明确任务 → 检索已有能力 → 给出候选方案
    ↓
Synthesizer：为标记为 Synthesize 的步骤补齐源码
    ↓
Verifier：检查并运行每个候选方案
    ↓
Workflow：根据验证结果安排修复、重新规划或结束
    ↓
Curator：记录结果，按条件注册算子和提炼经验
    ↓
流水线脚本 + 验证报告 + state.json
```

**“使用 LLM”指该阶段由 LLM 完成，是否请求远端 API 取决于运行模式。** `replay` 读取录制响应，`mock` 使用规则响应；`live` 和 `record` 命中缓存或录制时也会直接读取，未命中才请求 API。具体运行方式见 [README 第五节](../README.md#5-环境配置和运行方法)。

## 2. Planner：把需求变成候选执行方案

Planner 包含三个阶段：**需求理解 → 知识检索 → 方案生成**。它决定方案有哪些步骤、按什么顺序执行，以及每一步是 Reuse 还是 Synthesize。

### 2.1 输入与输出

| 阶段 | 输入 | 输出 | 是否使用 LLM |
| --- | --- | --- | --- |
| ① `understand()` | 自然语言需求、数据画像、指定目标列、历史场景指标 | `TaskSpec`：结构化任务规格 | 是 |
| ② `retrieve()` | `TaskSpec`、数据画像、关联场景、候选数量设置 | `PlanningContext`：候选算子、检索依据和历史经验 | 否 |
| ③ `propose()` | `TaskSpec`、数据画像、`PlanningContext`，可选的失败反馈和父方案 | `list[PipelinePlan]`：候选执行方案列表 | 是 |

### 2.2 `understand()`：需求理解

**输入：**

- `request`：用户的自然语言需求。
- `profile`：程序读取 CSV 后得到的数据画像，包括列名、类型、缺失情况、类别分布等。
- `target`：用户指定的目标列；可选，指定后优先采用。
- `scenario`：关联的历史场景；用于从知识图谱读取已有指标要求，也可以为空。

**输出：`TaskSpec`，即后续规划和验收共同使用的任务规格。**

```python
TaskSpec(
    description="预测客户流失，需要可解释性，召回率不低于 0.75。",
    task_type="binary_classification",
    modality="tabular",
    target_column="churn",
    metrics=[{"name": "recall", "threshold": 0.75, "primary": True}],
    constraints=["interpretable", "recall_first"],
)
```

这里把“召回率不低于 0.75”变成可比较的指标门槛，把“需要可解释性”变成规划约束标签。如果用户没有给出阈值，提示词要求优先采用场景中的阈值，否则采用默认值 0.7。

返回前会检查结构和目标列等字段，并移除缺乏出处的次要指标要求。`interpretable` 目前用于指导检索和规划，没有对应的确定性可解释性验收。

**作用：明确做什么、有哪些约束、用什么指标评价。**

### 2.3 `retrieve()`：查找已有能力和经验

**输入：**

- `spec`：上一步得到的 `TaskSpec`。
- `profile`：数据画像，主要使用其中的条件标签。
- `scenario`：关联的历史场景。
- `top_k`：每类算子返回多少个候选，默认 3。
- `executable_only`：是否只返回可执行算子；默认随是否允许生成新代码而调整。

代码把需求描述、画像标签和业务约束组成查询，按算子类型调用知识图谱的 `search_operators()`，再通过 `search_lessons()` 按数据模态过滤、按文本与适用条件排序查找相关经验。**这一阶段不调用 LLM。**

**输出：`PlanningContext`，即供下一步规划使用的上下文。** 假设当前使用初始能力库，清洗算子只有历史实现，以下只展示部分字段：

```text
PlanningContext
  scenario：telecom_churn
  scenario_metrics：关联场景的指标要求
  options：
    imputer：
      - MissingValueImputer@1，executable=True
      - TotalChargesCleaner@1，executable=False
    encoder：……
    model：
      - LogisticRegressionModel@1，executable=True
        score：检索得分
        evidence：匹配条件、历史验证等依据
  lessons：相关历史经验
```

这里最关键的是 `executable`：

- `True`：当前注册表中已有可执行实现，可以 Reuse。
- `False`：图谱中有历史能力描述或参考源码，但当前没有可直接执行的注册实现，可以作为 Synthesize 的参考。

因此，**知识图谱中查得到的算子，只有 `executable=true` 的才能直接运行**。每个候选还带有得分、分项得分、证据和参数信息，供 Planner 判断并在报告中说明依据。对象中的字段名是 `options`，放进 LLM 上下文时改为 `operator_options`。

**作用：告诉 Planner 有哪些能力可选、哪些可以直接执行、选择依据是什么。**

### 2.4 `propose()`：决定步骤、顺序与 Reuse / Synthesize

**输入：**

- `spec`：任务规格。
- `profile`：数据画像。
- `ctx`：检索得到的 `PlanningContext`。
- `n_candidates`：期望的候选方案数量，方法默认值为 2。
- `feedback`：之前验证失败的反馈，初次规划时通常为空。
- `parents`：重新规划时提供的已验证父方案摘要，初次规划时通常为空。

**输出：`list[PipelinePlan]`，即多个候选执行方案。** 例如可以提出下面两个方案，均保留线性模型作为可解释性方向，比较不同特征和参数：

```text
方案 A：清洗 + 逻辑回归
  TotalChargesCleaner       (Synthesize，参考历史 @1)
      ↓
  MissingValueImputer        (Reuse)
      ↓
  OneHotCategoricalEncoder   (Reuse)
      ↓
  StandardFeatureScaler     (Reuse)
      ↓
  LogisticRegressionModel    (Reuse，C=1.0)

方案 B：增加在网时长分箱特征 + 逻辑回归
  TotalChargesCleaner       (Synthesize，与 A 共用本次生成实现)
      ↓
  MissingValueImputer        (Reuse)
      ↓
  TenureBucketizer           (Synthesize，参考历史 @1)
      ↓
  OneHotCategoricalEncoder   (Reuse)
      ↓
  StandardFeatureScaler     (Reuse)
      ↓
  LogisticRegressionModel    (Reuse，C=0.5)
```

`TotalChargesCleaner` 处理特定列的脏值，不能代替对其他列的通用缺失值填补，因此仍保留 `MissingValueImputer`。候选也可以采用不同模型，实际选择由需求、检索结果和 LLM 响应共同决定。

每一步还包含选择理由和参数。例如一个复用步骤可以表示为：

```python
Step(
    kind="model",
    action="reuse",
    operator_ref="LogisticRegressionModel@1",
    params={"C": 1.0, "class_weight": "balanced"},
    rationale="根据可解释性要求选择线性分类模型，并通过类别权重处理不平衡。",
    evidence_ids=["Operator:LogisticRegressionModel@1"],
)
```

对于 Synthesize 步骤，Planner 提供的是 `SynthSpec`：类名、用途、输入输出和参考实现，源码由 Synthesizer 随后生成。**Planner 交付的是候选规划，是否可执行、指标是否达标，还要由后续验证确定。**

内部 LLM 响应使用 `PlanSet` 包装，形如 `{"plans": [...]}`；`propose()` 校验并绑定参数后，返回其中的方案列表。每个方案当前最多包含 2 个 Synthesize 步骤。更多校验规则见[技术附录](#8-技术附录)。

`propose()` 还检查本轮候选范围：Reuse 必须来自同类型可执行候选，历史 `reference` 必须来自同类型检索结果；对应步骤必须引用 `Operator:<ref>`。其他 `evidence_ids` 必须在 `evidence_catalog` 中可见，算子和历史验证引用必须关联当前算子；Lesson 检查其结构化适用对象。不存在的引用、跨算子引用和越界候选均反馈 LLM 重试；此外，表格数据有类别列时，缺少 encoder 步骤的方案同样被拒绝。

`RunState.evidence_checks` 保存每次引用检查的结果（含被拒绝的规划及原因），`lessons_used` 记录放入上下文的经验；`llm_calls` 保存调用来源、请求哈希、耗时和可获得的 token 用量。完整真实调用证据、三次重复和知识消融见[独立评测](../examples/evaluation/README.md)。

### 2.5 代码位置

主要实现：[factory/agents/planner.py](../factory/agents/planner.py)。

- `understand()`：调用 `self.llm.structured(..., TaskSpec, ...)` 理解需求。
- `retrieve()`：调用 `self.kg.search_operators()` 和 `search_lessons()` 检索知识。
- `propose()`：调用 `self.llm.structured(..., PlanSet, ...)` 提出并校验候选方案。

调用入口：[factory/workflow.py](../factory/workflow.py) 的 `_plan()`。关键调用顺序如下，画像生成和日志代码已省略：

```python
state.spec = self.planner.understand(
    state.request, rough, target=state.target, scenario=ctx.scenario,
)
ctx.planning = self.planner.retrieve(
    state.spec, ctx.profile, scenario=ctx.scenario,
)
state.candidates = self.planner.propose(
    state.spec, ctx.profile, ctx.planning, n_candidates=self.cfg.n_candidates,
)
```

其中 `rough` 是初步画像；理解需求并确定目标列后，Workflow 会重新生成 `ctx.profile`，再用于检索和规划。

**一句话总结：Planner 通过需求理解、确定性知识检索和方案生成，把自然语言需求变成带参数、理由和复用 / 生成标记的候选执行计划。** 首次规划通常经过两个 LLM 阶段，校验重试和重新规划会增加调用，实际远端请求次数还受运行模式与缓存影响。

## 3. Synthesizer：生成缺失代码，并按诊断修复

Synthesizer 只处理 Planner 标记为 Synthesize 的步骤。Reuse 步骤由执行层直接从注册表实例化，不需要 Synthesizer 再写一遍代码。

### 3.1 输入与输出

| 阶段 | 输入 | 输出 | 是否使用 LLM |
| --- | --- | --- | --- |
| `synthesize()` 生成 | `SynthSpec`、任务规格、数据画像、版本号、相关经验；按 reference 读取参考源码 | `GeneratedOperator`：源码、规格、版本和修订号 | 是 |
| `repair()` 修复 | 上一版 `GeneratedOperator`、结构化 `Diagnostic`、同诊断码的经验 | 修复后的 `GeneratedOperator`，修订号加 1 | 是 |

### 3.2 `synthesize()`：把算子规格变成实现

在前面的案例中，Planner 要求一个 `TotalChargesCleaner`，其任务是将含空白字符串的 `total_charges` 安全转换为数值。

```text
输入
  SynthSpec：类名、类型、用途、输入、输出、reference
  TaskSpec：需求描述、目标列、约束
  DataProfile：数据画像
  参考代码：图谱中的 TotalChargesCleaner@1 源码
  经验：先处理空白字符串，再转换数值等
      ↓
输出 GeneratedOperator
  spec：原算子规格
  source：完整 Python 算子类源码
  version：2（假设当前图谱和注册表中同名最高版本为 1）
  revision：0（本次运行内的首版代码）
```

LLM 接收算子接口规范，返回 Python 代码块；程序提取源码并包装为 `GeneratedOperator`。这里使用 `llm.chat()`，不走 `LLM.structured()` 的 JSON 校验。生成后先交给 Verifier，不在生成阶段直接执行。

### 3.3 `repair()`：根据具体诊断修改出错算子

假设首版清洗代码在数值转换后留下新的 NaN。Verifier 可以返回 `E_NAN`，并定位到 `TotalChargesCleaner@2`。Workflow 随后按同一诊断码检索修复经验，将**旧源码 + 具体诊断 + 同类经验**交给 `repair()`。

```text
首版源码：数值转换后留下新的 NaN
诊断：E_NAN，出错算子 TotalChargesCleaner@2
    ↓ repair()
修复源码：补齐产生的缺失值
version 仍为 2，revision 从 0 变为 1
    ↓
重新验证所有受影响的完整候选方案
```

这就是“带诊断的定向修复”：先定位问题，再修改对应算子，并用完整方案重新验证。提示词要求只处理该问题、保持接口，但程序不强制修改一定是最小 diff。

只有代码类诊断能够定位到本次生成的算子、且未超过修复上限时，才进入该分支；默认每个算子最多修复 3 轮。`RepairRecord.fixed=true` 只表示原算子上的同一诊断码不再出现，不表示整个方案已经通过。

实际执行过的[规则 mock 修复样例](../examples/verification/README.md#定向修复与后续复用)将 `fill_value=None` 改为 `0.0`，随后通过验证并在第二次运行中复用。该样例使用召回率门槛 0.4，与本文 0.75 的教学案例不同；不能将其结果当成本文案例已经达标。

### 3.4 代码位置

主要实现：[factory/agents/synthesizer.py](../factory/agents/synthesizer.py) 的 `synthesize()`、`repair()` 和接口规范 `CONTRACT`。调用入口是 [factory/workflow.py](../factory/workflow.py) 的 `_build()`、`_repair()`。

`_build()` 按算子名称去重，同名生成算子在一次运行内可以被多个候选共享；修复后，Workflow 将涉及它的旧报告移入历史，再重新验证。

**一句话总结：Synthesizer 把“需要一个什么算子”变成源码，并根据验证器提供的具体错误修改这份源码。**

## 4. Verifier：运行方案，给出指标和失败原因

Verifier 是确定性验证器，不调用 LLM。它依据检查规则和实际执行结果生成报告，供 Workflow 决定下一步。

### 4.1 输入与输出

| 方法 | 输入 | 输出 | 是否使用 LLM |
| --- | --- | --- | --- |
| `verify()` | `PipelinePlan`、`TaskSpec`、CSV 路径、该方案使用的生成源码 | `ValidationReport`：通过状态、诊断、指标、波动和资源数据 | 否 |

验证配置从构造参数或 `configs/tasks/<task_type>.yaml` 读取，包括交叉验证折数、随机种子、超时和内存上限等。

### 4.2 四级验证内容

| 层级 | 检查内容 | 常见失败诊断 |
| --- | --- | --- |
| ① 静态检查 | 生成源码的语法、允许的导入和调用、接口、元数据与参数一致性 | `E_SYNTAX`、`E_UNSAFE`、`E_INTERFACE` |
| ② 沙箱执行 | 子进程能否加载和执行；是否超时、超内存或触发越权行为 | `E_RUNTIME`、`E_TIMEOUT`、`E_UNSAFE` |
| ③ 契约检查 | 是否修改输入、改变行数、引入 NaN/Inf；预测形状和概率是否合法等 | `E_MUTATION`、`E_SHAPE`、`E_NAN`、`E_PROBA` |
| ④ 指标检查 | 多种子交叉验证的指标是否达到门槛，波动是否过大或分数疑似虚高 | `M_THRESHOLD`、`M_UNSTABLE` |

前置检查失败时停止后续评估。子进程负责加载所需算子并组装 sklearn Pipeline；Reuse 复用的是实现和参数，模型仍在当前训练数据上拟合。

### 4.3 诊断输出与错误定位

`ValidationReport` 包含 `passed`、`diagnostics`、`metrics`、`metric_std`、`resources` 和 `notes`。其中一个诊断可以表示为：

```text
Diagnostic
  code：E_NAN
  stage：contract
  operator：TotalChargesCleaner@2
  message：输出引入了新的 NaN/Inf
  hint：检查该算子产生和处理缺失值的逻辑
```

诊断结构包含**问题类型、发现阶段和出错位置**，Workflow 据此判断是否调用 Synthesizer 修复。`M_THRESHOLD` 表示指标未达到要求，通常进入方案或参数调整流程；该诊断本身不能定位某个算子的实现缺陷。

### 当前验证范围

- 支持二分类与多分类；二分类以少数类作为正类。
- 未注册的指标在 `notes` 中记为跳过，报告列出实际计算的指标。
- `interpretable` 等业务标签用于检索和规划；指标高于 `leakage_ceiling` 时判为疑似泄漏。
- 安全防线分三层：静态检查、子进程审计钩子、资源限制；注册前再做一次静态复查。生产环境可再加容器级隔离（见 README 第 10 节）。

固定代码、数据、配置与随机种子后结果可复现。

### 4.4 代码位置

入口：[factory/verify/__init__.py](../factory/verify/__init__.py) 的 `Verifier.verify()`。静态检查见 [static.py](../factory/verify/static.py)，子进程与行为拦截见 [sandbox.py](../factory/verify/sandbox.py)、[guard.py](../factory/verify/guard.py)，契约与指标检查见 [checks.py](../factory/verify/checks.py)。调用入口是 Workflow 的 `_verify()`。

**一句话总结：Verifier 用规则与执行结果回答“这条方案哪里有问题、指标是多少、是否满足当前验收项”，下一步调度由 Workflow 决定。**

## 5. Curator：算子注册与知识回写

Curator 负责记录运行成果。它的大部分操作由代码完成，只有把修复过程概括为经验时使用 LLM。

### 5.1 输入与输出

| 方法 | 输入 | 输出与副作用 | 是否使用 LLM |
| --- | --- | --- | --- |
| `record()` | `RunState`、关联场景 | 返回验证记录 ID 列表；更新图谱、注册表及状态中的注册 / 经验字段 | 记录和注册不使用；内部按条件调用 `distill()` |
| `distill()` | 一条 `RepairRecord`、算子用途 | `LessonRecord`：现象、通用修复方法、诊断码和适用条件 | 是 |

### 5.2 `record()`：写入内容与条件

| 内容 | 写入条件 | 后续用途 |
| --- | --- | --- |
| 新算子实现 | 仅最优且通过验证的方案所使用的新算子；注册前再次静态复查 | 后续任务直接 Reuse |
| 验证记录 | 当前候选的最终验证结果，失败结果也记录 | 提供可追溯的历史指标和搜索依据 |
| 修复经验 | 存在 `fixed=true` 的修复记录且配置了 LLM | 后续检索同类问题的处理方法 |
| 经验命中次数 | 本次放入规划或修复上下文的经验 | 统计经验被检索使用的次数 |

没有匹配场景时，会根据任务规格新建场景与指标关系。只有最优且通过验证的方案中的新算子才会注册为可复用能力。

启用磁盘回写时，新源码保存到图谱目录下的 `operators/<Name>_v<版本>.py`，图谱也会保存。`--no-writeback` 会禁止这些持久化操作，但当前进程的内存图谱和注册表仍可更新，运行目录中的报告与状态也照常保存。

### 5.3 `distill()`：从一次改动中总结经验

```text
输入
  原诊断：E_NAN，清洗后引入 NaN
  修复差异：fill_value 从 None 改为 0.0
  算子用途：清洗字符串数值列
      ↓ distill()
输出 LessonRecord（教学示意）
  title：数值转换后应处理新产生的缺失值
  trigger：转换脏值后出现新的 NaN
  fix：转换后按业务含义填补缺失值，并保持默认参数与元数据一致
  error_code：E_NAN
  applies_to：相关条件标签
```

写入时按诊断码与标题相似度融合。后续修复按同一诊断码和数据模态过滤，再按报错文本排序检索。

### 5.4 代码位置

主要实现：[factory/agents/curator.py](../factory/agents/curator.py) 的 `record()`、`distill()`，以及 `_register_operators()`、`_write_runs()`、`_distill_lessons()`。Workflow 的 `_curate()` 调用 `record()`，并把其返回值赋给 `state.kg_writes`。

**一句话总结：Curator 保存符合条件的实现、验证结果和修复经验，为下一次任务提供可检索、可复用的材料。**

## 6. Workflow：阶段调度与诊断路由

各阶段由 [factory/workflow.py](../factory/workflow.py) 的 `Workflow.run()` 统一调度。`RunState` 保存可持久化的共享运行状态，`ctx` 保存画像、检索上下文和待修复队列等运行期对象。

### 6.1 根据诊断选择下一步

`_route()` 按以下顺序判断：

1. **先处理可修复的生成算子。** 诊断是代码类、能定位到本次生成算子、且未达修复上限时，进入 `_repair()`，然后重新验证受影响方案。
2. **再检查是否有方案通过。** 有则选优，进入 `_curate()`。
3. **仍无通过方案时，检查能否重新规划。** 重规划次数未耗尽，且诊断全为指标类或当前允许生成新算子时，进入 `_replan()`。
4. **无法继续时记录失败。** 仍进入 Curator 记录结果，随后结束为失败。

因此，“`E_*` 一定修复、`M_*` 一定重规划”只是粗略理解。实际还取决于错误是否可定位、是否为本次生成的代码、是否已有候选通过，以及剩余重试次数。允许生成时，代码问题无法继续修复也可能进入重新规划。

### 6.2 Beam Search 重新规划流程

重新规划沿用已经得到的任务规格和检索上下文，再次调用 `Planner.propose()`，附上失败反馈与父方案摘要：

```text
选出排名前 k 个已验证父方案
    ↓
请求 Planner 扩展 k × m 个子方案
    ↓
按步骤、算子和参数签名去重
    ↓
结合匹配场景的历史记录与本次观测估计先验分
    ↓
最多保留 k 个子方案，进入生成与验证
```

`k` 是 beam 宽度，`m` 是扩展倍数。提示词要求局部修改，诊断为 `M_THRESHOLD` 时至少一个子方案直接调整决定该指标的参数（如召回不足时加大正类 `class_weight`）。新场景没有历史证据时，先验使用本次观测与阈值平滑。真实搜索结果见[评测](../examples/evaluation/README.md)。

实现见 [factory/search.py](../factory/search.py) 和 Workflow 的 `_replan()`。规则 mock 样例见 [Beam Search 样例](../examples/verification/README.md#beam-search)。

### 6.3 运行产物

`_save()` 将状态和报告写入 `runs/<run_id>/`，并保存本次生成源码；有选中的方案和任务规格时还导出 `pipeline.py`。如果方案未通过，脚本会带失败警告，不能仅凭文件生成就认为任务成功。最终状态以 `state.status` 和验证报告为准。

**一句话总结：Workflow 根据共享状态和验证结果安排下一步，各 Agent 提供本阶段的规划、源码、诊断或记录。**

## 7. 补充：历史代码与文档的知识抽取

前面使用的历史能力和经验来自单独的批量抽取流程，在任务运行前完成；是否请求 API 取决于 LLM 模式和缓存。

| 阶段 | 输入 | 输出 | 是否使用 LLM |
| --- | --- | --- | --- |
| `scan_repo()` | 历史 Python 代码仓库 | 通过 AST 提取的函数签名、文档字符串、源码和依赖线索 | 否 |
| `summarize_function()` | 函数信息和源码 | `CapabilityCard`：能力类型、用途、输入输出、条件和参数 | 是，标签 `extract_code` |
| `extract_doc()` | 业务文档、场景、已知算子名等 | `DocKnowledge`：指标、约束、实验和经验 | 是，标签 `extract_doc` |
| `build_kg()` | 抽取结果、材料清单和函数信息 | 融合后的能力知识图谱 | 否 |

例如，文档中的“数值列含空白字符串时，先 strip，再安全转换并处理缺失值”，可以抽取为带 `E_RUNTIME` 诊断码的经验。Planner 检索时可用它安排清洗步骤，Synthesizer 生成或修复时也可将它放入上下文。

代码见 [factory/extract.py](../factory/extract.py)。本体、抽取、融合和存储的详细解释见 [schema.md](schema.md)，这里不重复展开。

## 8. 技术附录

### 关键对象与字段

| 对象 | 关键字段 | 定义位置 |
| --- | --- | --- |
| `TaskSpec` | `description`、`task_type`、`modality`、`target_column`、`text_column`、`metrics`、`constraints` | [schemas.py](../factory/schemas.py) |
| `PlanningContext` | `scenario`、`scenario_metrics`、`options`、`lessons`；`to_prompt()` 将 `options` 映射为 `operator_options` | [planner.py](../factory/agents/planner.py) |
| 检索候选 | `ref`、`executable`、`origin`、`description`、`conditions`、`params`、`defaults`、`score`、`breakdown`、`evidence` | [planner.py](../factory/agents/planner.py) 的 `retrieve()` |
| `PipelinePlan` | `name`、`rationale`、`steps` | [schemas.py](../factory/schemas.py) |
| `Step` | `kind`、`action`、`operator_ref`、`synth_spec`、`params`、`rationale`、`evidence_ids` | 同上 |
| `SynthSpec` | `name`、`kind`、`purpose`、`inputs`、`outputs`、`reference` | 同上 |
| `GeneratedOperator` | `spec`、`source`、`version`、`revision`；`key` 为 `Name@version` | 同上 |
| `Diagnostic` | `code`、`stage`、`message`、`operator`、`hint` | 同上 |
| `ValidationReport` | `plan_name`、`passed`、`diagnostics`、`metrics`、`metric_std`、`resources`、`notes` | 同上 |
| `RepairRecord` | `operator`、`revision`、`diagnostic`、`lessons`、`fixed`、`before`、`after` | 同上 |
| `LessonRecord` | `title`、`trigger`、`fix`、`error_code`、`applies_to` | [extract.py](../factory/extract.py) |
| `SearchRound` | `round`、`parents`、`width`、`candidates`；候选记录先验、证据、是否入选及理由 | [schemas.py](../factory/schemas.py) |

`metrics` 中每项包含 `name`、`threshold`、`primary`，恰好一个主指标。`metric_std` 表示不同种子聚合结果之间的标准差；`resources` 记录训练耗时、千条预测耗时和峰值内存等。

### 消息结构与校验

Planner 和 Curator 的结构化请求通常由角色规则、结构化上下文及 JSON Schema 要求组成：

```text
system：角色与规则
user：<<CONTEXT>> {结构化输入} <</CONTEXT>>
system：只返回符合指定 JSON Schema 的 JSON 对象
```

`LLM.structured()` 先解析响应并进行 Pydantic 校验，再执行调用方提供的语义检查。失败时追加原回复和错误信息，要求更正，达到尝试上限后抛出异常。Planner 的两类结构化请求最多各尝试 3 次。

`TaskSpec` 检查唯一主指标、文本任务必需文本列等，Planner 另检查目标列和文本列存在。`_grounded_metrics()` 保留主指标，对次要指标按场景中同名同阈值记录或需求文本中的阈值字符串过滤。被移除的次要要求是否仍在报告中计算，取决于验证配置的 `report_metrics`，不是自动保证“只报告不设门槛”。

`PipelinePlan` 要求最后且仅有一个模型步骤；Reuse 必须给出引用，Synthesize 必须给出规格。`plan_errors()` 检查可复用算子的存在性、类型、模态和参数名，检查生成数量、参考节点以及已有可执行能力的重复生成等。`bind()` 规范化引用、绑定文本列等参数，并对齐生成算子的名称和版本参考。这些检查不能替代实际执行验证。

Synthesizer 的 `synthesize()` 和 `repair()` 使用 `LLM.chat()` 返回源码字符串，不走结构化 JSON 校验。接口规范要求指定类名、sklearn 兼容接口、`meta`、参数默认值一致、转换不改输入和行数、处理自身引入的非有限值、合法的模型概率输出，以及允许的库导入；落实情况由验证器检查。

调用标签包括 `plan_spec`、`plan`、`synthesize`、`repair`、`lesson`、`extract_code`、`extract_doc`。录制与缓存文件采用 `<tag>-<请求哈希>.json`，哈希包含模型、消息和 JSON 模式，不含密钥与接口地址。抽取请求直接包含代码或文档文本；规则 mock 与真实模型使用同一工作流，但不能代替真实模型能力评估。

### 算子接口与执行

算子负责执行 Pipeline 中的一个步骤。Planner 通过 `Step` 描述使用哪个算子及其参数，执行层通过注册表创建实例，再调用其 sklearn 兼容接口处理数据。**实例化、拟合、变换和预测本身都不调用 LLM。**

#### ① 接口的输入与输出

`X` 表示特征数据，不包含目标列；`y` 表示目标标签。表格预处理通常返回 DataFrame，文本向量化可以返回稀疏矩阵，后续步骤必须支持对应格式。

| 阶段 / 方法 | 输入 | 输出 | 作用 |
| --- | --- | --- | --- |
| `registry.create(ref, **params)` | 算子引用、步骤参数 | 算子实例 | 从注册表取得类，合并 `meta.params` 默认值与步骤参数；同名参数以步骤参数为准 |
| 预处理 `fit(X_train, y=None)` | 训练特征，可选的标签 | 算子自身 `self` | 学习中位数、编码词表或缩放参数等状态 |
| 预处理 `transform(X)` | 与接口兼容的特征 | 变换后的特征，样本行数不变 | 使用已学习的状态处理数据 |
| 模型 `fit(X_train, y_train)` | 经过前置处理的训练特征、标签 | 模型算子自身 `self` | 训练分类模型，保存模型状态与 `classes_` |
| 模型 `predict(X)` | 与模型输入兼容的特征 | 形状为 `(n_samples,)` 的标签数组 | 返回预测类别 |
| 模型 `predict_proba(X)` | 与模型输入兼容的特征 | 形状为 `(n_samples, n_classes)` 的概率数组 | 概率列顺序对应 `classes_`，每行概率和为 1 |
| `build_pipeline(plan)` | 算子均已注册、步骤顺序与数据格式兼容的方案 | sklearn `Pipeline` | 将步骤按顺序实例化并组合 |

`meta` 声明的是能力描述与默认配置，包括 `name`、`version`、`kind`、`modalities`、`inputs`、`outputs`、`conditions`、`dependencies`、`params` 等。`fit()` 学到的状态保存在实例中，例如 `fill_values_`、`encoder_`、`model_`，不属于 `meta.params`。

#### ② 示例：缺失值处理算子

以下示例在已安装项目依赖的环境中、仓库根目录下运行，仅使用已有算子，不调用 LLM：

```python
import pandas as pd
from factory.operators import registry

X_train = pd.DataFrame({
    "age": [20.0, None, 40.0, 30.0],
    "city": ["A", "B", None, "A"],
})
y_train = pd.Series([0, 1, 0, 1])

imputer = registry.create("MissingValueImputer@1")
imputer.fit(X_train, y_train)       # 学习 age 中位数 30.0、city 众数 A
X_filled = imputer.transform(X_train)
print(X_filled)
```

输出：

```text
    age city
0  20.0    A
1  30.0    B
2  40.0    A
3  30.0    A
```

`fit()` 返回算子自身并保存填补规则，`transform()` 返回变换后的数据。新数据也使用训练时学到的中位数和众数，不重新估计这些值，输入 DataFrame 保持不变。

#### ③ 示例：从方案组装并执行 Pipeline

继续使用上面的 `X_train` 和 `y_train`，将缺失值处理、编码、标准化和模型串联起来：

```python
from factory.operators import build_pipeline
from factory.schemas import PipelinePlan, Step

plan = PipelinePlan(
    name="lr_interface_example",
    steps=[
        Step(kind="imputer", action="reuse", operator_ref="MissingValueImputer@1"),
        Step(kind="encoder", action="reuse", operator_ref="OneHotCategoricalEncoder@1"),
        Step(kind="scaler", action="reuse", operator_ref="StandardFeatureScaler@1"),
        Step(kind="model", action="reuse", operator_ref="LogisticRegressionModel@1",
             params={"C": 0.5, "class_weight": "balanced"}),
    ],
    rationale="演示已有算子的实例化、拟合与预测接口。",
)

pipe = build_pipeline(plan)
pipe.fit(X_train, y_train)
labels = pipe.predict(X_train)              # shape: (4,)
probabilities = pipe.predict_proba(X_train)  # shape: (4, 2)
```

这段代码手工构造方案，用于说明接口；实际任务由 Planner 生成方案，再交给 Verifier 验证。`build_pipeline()` 只负责创建和组合实例，不会自行生成缺失代码或训练模型。

执行顺序为：

```text
fit(X_train, y_train)
  MissingValueImputer：fit → transform
      ↓
  OneHotCategoricalEncoder：fit → transform
      ↓
  StandardFeatureScaler：fit → transform
      ↓
  LogisticRegressionModel：fit

predict(X_new) / predict_proba(X_new)
  依次调用已拟合预处理步骤的 transform
      ↓
  调用模型的 predict / predict_proba
```

验证时，每个交叉验证折都会创建并拟合独立 Pipeline，预处理规则只从该折训练集学习。上面的四行数据只用于展示接口，性能验收使用任务配置中的验证流程。

#### ④ Reuse、Synthesize 与注册时机

| 步骤标记 | 实现从哪里来 | 何时可实例化 |
| --- | --- | --- |
| `reuse` | 注册表中已有的内置算子或已加载的历史生成算子 | 注册类可用后，通过 `operator_ref` 创建实例 |
| `synthesize` | Synthesizer 按 `SynthSpec` 返回源码 | 静态检查后，在验证子进程中执行源码并注册类，再按名称创建实例 |

Reuse 复用实现与参数配置，在当前训练数据上重新拟合。Synthesize 补齐实现后也走相同的拟合与预测接口。只有最优且通过验证的方案所使用的新算子，经 Curator 静态复查后才注册进主进程能力库；启用回写时同时保存源码，供后续任务加载。

#### ⑤ 代码位置

- [operators/base.py](../factory/operators/base.py)：`Operator`、`OperatorMeta`、`OperatorRegistry.create()` 和 `register()`。
- [operators/tabular.py](../factory/operators/tabular.py)：缺失值处理、类别编码和标准化的完整实现。
- [operators/models.py](../factory/operators/models.py)、[operators/text.py](../factory/operators/text.py)：分类模型与文本向量化实现。
- [operators/__init__.py](../factory/operators/__init__.py)：`build_pipeline()`，按步骤创建并组装实例。
- [operators/synthesized.py](../factory/operators/synthesized.py)：生成源码的加载、注册与保存。
- [verify/checks.py](../factory/verify/checks.py)：`smoke_and_contract()` 逐步检查接口，`evaluate()` 执行交叉验证。
- [codegen.py](../factory/codegen.py)：导出包含训练和预测入口的 `pipeline.py`；导出脚本将拟合后的完整 Pipeline 保存为 `model.joblib`，其中包含预处理状态和模型参数。

### 共享状态 RunState

下表中的字段由 Workflow 接收各组件结果后写入，部分由 Curator 直接更新：

| 字段 | 内容与主要用途 |
| --- | --- |
| `request`、`data_path`、`target` | 用户输入 |
| `spec`、`profile_tags`、`retrieval` | 任务规格、数据标签、检索依据 |
| `candidates`、`generated` | 候选方案列表、算子名到生成产物的映射 |
| `reports`、`history` | 当前验证结果、因修复被替换的旧报告 |
| `repairs`、`search` | 修复记录、Beam 搜索轮次记录 |
| `best_plan`、`status` | 选中方案与最终状态 |
| `registered`、`kg_writes`、`lessons_learned`、`lessons_used` | 注册算子、验证记录 ID、新经验、放入上下文的经验 ID |
| `verify_config`、`llm_model`、`timings`、`log`、`artifacts` | 配置、模型名、阶段耗时、日志与产物路径 |

状态保存为 `runs/<run_id>/state.json`，便于追踪各阶段输入输出并重新生成报告。
