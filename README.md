# AlgoForge：基于 AI Agent 的算法能力工厂

> AlgoForge 从历史代码和业务文档中抽取算法能力与经验，通过知识检索辅助 LLM 规划 Pipeline，复用已有算子或生成缺失实现。验证器依据固定规则和实际执行结果检查候选方案，对可定位的生成代码问题进行定向修复，并按配置保存验证记录、可复用算子和修复经验。

```
能力抽取 ──▶ 能力复刻（Reuse / Synthesize）──▶ 自动验证 ──▶ 修复优化 ──▶ 知识沉淀
   ▲                                                                     │
   └──────────────────────── 下一次运行直接复用 ◀─────────────────────────┘
```

**快速开始（离线、无需 API Key）**

```bash
conda env create -f environment.yml && conda activate algoforge
python -m factory.cli demo          # 回放录制的真实 LLM 响应；安装完成后运行，耗时随机器配置变化
streamlit run app.py                # Web 界面
```

| 演示（`examples/demo/`） | 验证内容 | 结果 |
| --- | --- | --- |
| [01 工单文本分类](examples/demo/runs/01_tickets_transfer/report.md) | **跨场景迁移**：图谱中没有该场景的历史记录，利用通用经验（中文采用字符级 TF-IDF）规划分类方案 | F1-macro 0.828 ✅ |
| [02 客户流失](examples/demo/runs/02_churn_synthesize/report.md) | **知识驱动的生成**：参考历史代码生成 4 个新算子，3 个候选公平比较，最优方案通过后注册 `TotalChargesCleaner@2`、`TenureBucketizer@2` | recall 0.778 ✅ |
| [03 相似需求](examples/demo/runs/03_churn_reuse/report.md) | **沉淀后复用**：换一种说法提出相似需求，直接复用 02 沉淀的两个算子 | recall 0.778 ✅ |

任务 03 证明了算子可以被后续任务复用；当前没有对照实验验证持续运行是否提高指标、缩短耗时或降低 LLM 调用成本。

**文档导航**：[知识图谱：本体构建 → 知识抽取 → 知识融合 → 知识存储](docs/schema.md) ｜ [各 Agent 的输入与输出](docs/agents_io.md) ｜ [开发路线图](docs/roadmap.md)

### 题目要求与验收证据

对照 [ProjectTarget.md](ProjectTarget.md)，下表区分真实 LLM 录制回放、规则 mock 测试和当前实现边界：

| 题目要求 | 实现 | 可查看或复现的证据 |
| --- | --- | --- |
| 行业场景、示例材料与 LLM 接入 | 客户流失、工单分类；OpenAI 兼容接口及录制回放 | [数据说明](data/README.md)、[LLM 客户端](factory/llm.py)、[三任务演示](examples/demo/SUMMARY.md) |
| 能力抽取与知识图谱 | AST + LLM 抽取、规则融合、NetworkX 图谱 | [schema 与融合规则](docs/schema.md)、[原始抽取结果](kg/extraction.json)、[基础图谱](kg/capability_kg.json) |
| 自然语言到算法代码 | Planner 规划，Synthesizer 生成缺失算子 | 真实 LLM 录制回放：[任务 02 报告](examples/demo/runs/02_churn_synthesize/report.md)、[生成脚本](examples/demo/runs/02_churn_synthesize/pipeline.py) |
| 自动验证与多候选比较 | 四级验证、多种子交叉验证、候选排序 | [任务 02 候选结果](examples/demo/runs/02_churn_synthesize/report.md)、[验证测试](tests/test_verify.py)、[运行时安全测试](tests/test_sandbox_guard.py) |
| 多轮修复与经验提炼 | 有次数上限的定向修复、Curator 提炼 Lesson | [规则 mock 修复样例](examples/verification/README.md#定向修复与后续复用)；[闭环测试](tests/test_synthesis.py)验证一轮修复及后续复用，当前录制演示未触发修复 |
| 知识回写与能力版本复用 | 保存生成源码、验证记录和版本谱系 | [任务 02 注册 → 任务 03 复用](examples/demo/SUMMARY.md)、[沉淀的源码](examples/demo/kg/operators/) |
| Beam Search | 父方案扩展、去重、历史与本次观测先验剪枝 | [规则 mock 搜索样例](examples/verification/README.md#beam-search)、[先验与剪枝测试](tests/test_search.py)；尚无搜索收益的对照实验 |
| UI、报告与可扩展配置 | CLI、Streamlit、Markdown / HTML 报告、算子与指标注册接口 | [Web 入口](app.py)、[报告样例](examples/demo/runs/02_churn_synthesize/report.html)、[任务配置](configs/tasks/)；当前支持二分类和多分类 |

---

## 目录

1. [项目背景和目标](#1-项目背景和目标)
2. [系统架构：算子与 Pipeline](#2-系统架构算子与-pipeline)
3. [能力知识图谱：四个关键步骤](#3-能力知识图谱四个关键步骤)
4. [Agent 工作流与调度](#4-agent-工作流与调度)
5. [环境配置和运行方法](#5-环境配置和运行方法)
6. [示例数据和测试任务说明](#6-示例数据和测试任务说明)
7. [生成算法代码示例](#7-生成算法代码示例)
8. [验证结果和报告样例](#8-验证结果和报告样例)
9. [实现挑战与处理方式](#9-实现挑战与处理方式)
10. [后续可扩展方向](#10-后续可扩展方向)

---

## 1. 项目背景和目标

### 项目需求与实现目标

项目针对自然语言生成算法代码过程中的三类问题，构建能力检索、代码生成、验证修复和知识回写流程：

| 问题 | 影响 | 实现方式 |
| --- | --- | --- |
| **历史能力缺少统一检索入口** | 重复开发；历史上的脏数据、数据泄漏、类别不平衡等问题可能再次出现 | **能力知识图谱**：抽取历史代码与文档知识，供规划和生成前检索 |
| **生成代码缺少独立验收** | 可能修改输入、产生 NaN、出现指标虚高或越权读取文件 | **四级验证器**：不调用 LLM，依据固定检查规则与执行结果判定；固定代码、数据、配置与环境后支持复现 |
| **运行成果缺少持久化与复用机制** | 后续相似任务无法直接利用已有实现和验证结果 | **知识沉淀**：符合条件的新算子注册进能力库，验证记录和修复经验按配置写回图谱 |

系统将题目中的七个环节组织成**可积累知识的闭环**：完成运行后按回写配置保存验证记录；最优且通过的方案中用到的新算子可以注册，原问题已解决的修复可以提炼经验。这些积累为后续任务提供检索依据，效果取决于知识质量与检索结果。

### 能力粒度、职责分工与经验复用

**① 能力粒度：算子（Operator）**

完整脚本通常需要额外拆分和适配才能部分复用。本项目参考 PyTorch 模块组合的方式，将清洗、编码、特征处理和模型封装为算子，通过元数据描述其用途，并在接口与数据格式兼容时组合：

- 知识图谱可以描述"这个算子适用于什么条件、依赖什么、历史表现如何"；
- 验证器在逐步执行和契约检查中可以定位部分错误；能定位到本次生成算子的代码问题才进入定向修复；
- 最优且通过的方案中使用的新算子可以**单独注册、供下次复用**。

**② 职责分工：LLM 负责语义理解与生成，程序负责数据处理、检查与调度**

| 交给 LLM（需要理解与创造） | 交给确定性代码（需要可靠与可复现） |
| --- | --- |
| 理解需求、规划方案、写算子代码、按诊断修复、从文档中抽取知识、把修复过程总结成经验 | 数据画像（列类型、缺失率、是否不平衡）、知识检索打分、方案的语义校验、四级验证、路由决策、版本管理 |

需求与方案输出分别校验为 `TaskSpec`、`PipelinePlan`，不合格时带错误重试。生成源码保存在 `GeneratedOperator` 中，源码本身通过静态检查与执行验证，满足路由条件时按诊断修复。

**③ 经验复用：使用诊断码关联错误与修复经验**

验证器的每个诊断都有固定的**诊断码**（如 `E_NAN`、`E_RUNTIME`、`M_THRESHOLD`），知识图谱中的经验（Lesson）使用同一套诊断码标注：

- 历史文档中"空白字符串转 float 报错"的复盘被抽取为 `E_RUNTIME` 经验；修复时先按诊断码过滤，再按报错文本相关度检索，取前几条放入 prompt；
- 一次修复成功后，Curator 把"诊断 → 修复"总结成新经验，用同一个诊断码写回；
- 诊断码参与**路由**：可定位且未超次数的生成代码错误优先交回 Synthesizer；没有通过的候选且允许重试时，再由 Planner 重新规划。

### 场景选择

- **电信客户流失预测**（表格二分类，主场景）：它集中了需要业务处理的情况——`total_charges` 是带空白的字符串、高基数的客户 ID、正负样本约 1:3、业务要求可解释且召回优先，用于观察历史知识如何进入规划和代码生成。
- **客服工单分类**（中文文本多分类）：图谱中没有这个场景的任何知识，用来检验同一套框架能否迁移。

## 2. 系统架构：算子与 Pipeline

### 组合式设计

组合式设计参考了 PyTorch 的模块接口，以及 [DataFlow](https://github.com/OpenDCAI/DataFlow) 的 Operator / Pipeline 分层思路；本项目的执行层使用 sklearn Pipeline。

| PyTorch | 本项目 | 作用 |
| --- | --- | --- |
| `nn.Module` | `Operator`：sklearn 兼容（预处理实现 `fit/transform`，模型实现 `fit/predict/predict_proba`）+ 元数据 `OperatorMeta` | 能力的最小单元，接口统一 |
| `nn.Sequential` | `sklearn.pipeline.Pipeline` | 把算子按顺序组合成完整方案 |
| `torchvision.models` 注册表 | `OperatorRegistry`（键为 `名称@版本`）+ 能力知识图谱 | 按名字和版本找到能力 |

### 算子接口与执行

算子是 Pipeline 中的一个处理步骤，由 **`meta` 元数据与实现代码**组成。`meta` 描述名称、版本、类型、用途、适用条件、依赖和默认参数；实现代码按 sklearn 接口接收数据并完成处理。

| 方法 | 输入 | 输出 | 用途 |
| --- | --- | --- | --- |
| `registry.create(ref, **params)` | 算子引用，如 `MissingValueImputer@1`；步骤参数 | 算子实例 | 按名称和版本取得类，合并默认参数与步骤参数 |
| `fit(X_train, y_train)` | 训练特征、标签；预处理算子的标签可选 | 已拟合的算子自身 | 学习填补值、编码规则、缩放参数或模型参数 |
| `transform(X)` | 待处理特征 | 变换后的特征，行数保持不变 | 缺失值处理、编码、标准化等预处理 |
| `predict(X)` / `predict_proba(X)` | 模型所需的特征 | 预测标签 / 各类别概率 | 完成分类预测 |

例如，`MissingValueImputer.fit()` 从训练集学习数值列中位数和类别列众数，`transform()` 使用这些值填补训练集或新数据的缺失值。测试或预测阶段使用训练时学到的规则。

规划、实例化和执行的关系如下：

```text
Planner 输出 PipelinePlan（步骤、参数、选择理由）
    ↓
Synthesizer 补齐 Synthesize 步骤的源码；Reuse 步骤使用已有实现
    ↓
Verifier 子进程加载算子，通过注册表实例化并组装 sklearn Pipeline
    ↓
Pipeline.fit()：依次拟合并变换前置步骤，最后拟合模型
    ↓
Pipeline.predict() / predict_proba()：使用已拟合的前置步骤变换数据，再由模型预测
```

上述实例化、拟合、变换和预测方法本身不调用 LLM；LLM 用于规划，以及需要时生成或修复源码。详细输入输出、可运行示例和代码位置见 [算子接口与执行](docs/agents_io.md#算子接口与执行)。新增算子需要实现接口、注册类，并确保验证子进程能够加载它。

能力库持久化的是**算子实现源码**（`kg/operators/<Name>_v<版本>.py`）。Reuse 会在当前数据上重新拟合；训练后的模型由生成脚本另外保存为 `model.joblib`。

### 整体架构

**Workflow 是中央控制器**：统一调用各阶段，通过 `RunState` 保存需求、候选方案、源码和验证结果。下图上层为知识与能力层，下层展示输入、规划、算子实现、验证、沉淀与输出；实线连接主流程，虚线标注知识读取、诊断反馈及回写关系。点击图片可查看原图。

<p align="center">
  <a href="pic/AlgoForge%20系统架构总览.png"><img src="pic/AlgoForge%20系统架构总览.png" alt="AlgoForge 系统架构总览：知识与能力层、Agent 驱动工作流及修复闭环" width="100%"></a>
</p>

一个候选方案可同时包含 Reuse 和 Synthesize 步骤；全部复用时无需调用 Synthesizer。Verifier 在验证子进程中加载所需实现并实例化流水线。Curator 负责运行后的知识回写；Planner 与 Synthesizer 读取图谱以规划或生成。

流水线的组装与执行在 Verifier 的验证流程中完成，产物由 Workflow 导出。诊断路由是简化表示，定向修复的适用条件、指标失败后的 Beam Search 及停止条件，见[第 4 节的状态机](#4-agent-工作流与调度)。

```
├── factory/
│   ├── schemas.py          # 三层中间表示 TaskSpec / PipelinePlan / 代码，诊断码，共享状态 RunState
│   ├── operators/          # Operator 基类与注册表、7 个内置算子、生成算子的加载与持久化
│   ├── extract.py          # 能力抽取：AST + LLM → 知识图谱
│   ├── kg.py / kg_viz.py   # 知识图谱的本体、融合、存储、检索、可视化
│   ├── profile.py          # 数据画像：列角色与适用条件标签（确定性）
│   ├── agents/             # planner / synthesizer / curator；rules.py 为离线规则版
│   ├── verify/             # 四级验证：static / sandbox / checks，guard 为沙箱内的审计钩子
│   ├── workflow.py         # 中央控制器：显式状态机
│   ├── search.py           # 重新规划阶段的 Beam Search
│   ├── llm.py              # OpenAI 兼容客户端、结构化输出与重试、缓存、录制回放、MockLLM
│   ├── codegen.py          # 方案 → 可独立运行的 pipeline.py
│   ├── report.py           # Markdown / HTML 验证报告
│   └── demo.py / cli.py    # 演示与命令行
├── app.py                  # Streamlit 界面
├── configs/                # llm.yaml；tasks/*.yaml 为各任务类型的验证配置
├── data/                   # 示例数据与模拟业务材料
├── kg/                     # 知识图谱（JSON / GraphML / HTML）与原始抽取结果
├── examples/               # 录制的 LLM 响应（cassettes）与演示产物
├── docs/                   # schema.md、agents_io.md、roadmap.md
└── tests/                  # 118 个常规测试 + 1 个完整演示回放测试
```

### 技术选型

| 选择 | 理由 |
| --- | --- |
| 显式状态机 | 通过代码定义阶段转移，便于单独测试，减少工作流框架依赖 |
| NetworkX + JSON / GraphML | 适用于当前百级节点的原型规模；图谱文件可纳入 Git，GraphML 用于交换，迁移到图数据库需补充导入工具与属性映射 |
| BM25+（rank-bm25） | 轻量、可解释、不需要向量模型；中文按字 + 二元组切分 |
| pydantic | 校验需求、方案、抽取结果等 JSON 输出；生成源码另外交给验证器检查 |
| sklearn 接口的算子 | 直接复用 Pipeline、clone、交叉验证生态 |
| OpenAI 兼容接口 + 录制回放 | 可配置兼容的模型与接口；已录制的示例无需 API Key 即可回放 |
| Streamlit + pyvis | 少量代码即可得到可交互的界面与图谱 |

## 3. 能力知识图谱：四个关键步骤

知识图谱保存可检索的长期知识，节点与关系覆盖 Agent 选型所需的适用条件、历史表现、来源和版本谱系。构建过程分为四步，完整说明及关系图见 [docs/schema.md](docs/schema.md)。

| 步骤 | 目标 | 实现 |
| --- | --- | --- |
| **① 本体构建** | 定义能力及其关联信息 | 8 类节点（场景、算子、指标、适用条件、依赖、验证记录、经验、出处）、9 类边；区分可执行的"能力"与只能参考的"知识"；阈值放在"场景 → 指标"的边上；经验与验证器共用诊断码；适用条件使用受控词表 |
| **② 知识抽取** | 将代码和文档转为结构化知识 | 代码：AST 解析签名、依赖与源码（确定性），LLM 归纳为能力卡片；文档：LLM 抽取指标阈值、约束、历史实验和经验；运行时：Curator 从修复过程中提炼经验 |
| **③ 知识融合** | 统一标识、合并重复知识并处理特定来源优先级 | 实体链接到已有节点；`RecallAtTopK(k=0.2)` / `recall@20%` / `recall_at_20` 规范化为同一指标；诊断码相同且标题相似的经验合并（11 → 10 条）；初始材料中的场景阈值取自需求文档；被推翻的实验保留但不计入历史分；生成算子接入已有版本谱系 |
| **④ 知识存储** | 持久化图谱并支持检索 | NetworkX 图 + JSON 主存储 + GraphML 交换格式 + HTML 可视化；检索打分 `0.4 × BM25+ 文本 + 0.3 × 条件匹配 + 0.3 × 历史验证分`，每条结果带分项得分和证据；完成 Curator 阶段后按回写配置保存 |

**历史材料中的经验与验证记录示例：**

```json
{"type": "Lesson", "title": "数值列含空白字符导致类型转换失败", "error_code": "E_RUNTIME",
 "trigger": "新入网客户累计消费为空白字符串，转 float 时抛出 ValueError: could not convert string to float: ' '",
 "fix": "先 str.strip()，再 pd.to_numeric(errors='coerce')，缺失值填补为 0"}
{"type": "ValidationRun", "metrics": {"roc_auc": 0.97}, "passed": false, "origin": "history",
 "note": "作废：全量数据目标编码泄漏标签，上线后 AUC 跌至 0.71"}
```

前者为 Planner 安排 `total_charges` 清洗步骤、Synthesizer 生成实现提供依据；后者将泄漏实验标为无效，计算历史验证分时排除其 AUC 0.97 的记录。

| 文件 | 内容 |
| --- | --- |
| [`kg/capability_kg.json`](kg/capability_kg.json) / `.graphml` / [概览 `.html`](kg/capability_kg.html) / [完整 `_full.html`](kg/capability_kg_full.html) | 基础知识图谱：14 个算子（7 个内置 + 7 个抽取）、10 条经验、5 条历史实验 |
| [`kg/extraction.json`](kg/extraction.json) | 融合前的 LLM 原始抽取结果 |
| [`examples/demo/kg/`](examples/demo/kg/) | 演示 3 个任务后的图谱：算子 14 → 17、验证记录 5 → 12、场景 1 → 2；`operators/` 下是沉淀的生成算子源码 |

可视化：`python -m factory.cli kg viz` 同时生成**概览**（只画场景 / 指标 / 算子 / 经验四类主干节点，演示后为 33 个节点 / 39 条边）与**完整视图**（81 / 174）；`--focus <节点 id>` 只看某个节点的邻域。Web 界面中可以切换。

下面并排展示**初始图谱**与**三个演示任务完成后的图谱概览**，点击图片可查看原图。

<table>
  <tr>
    <th width="50%">初始图谱</th>
    <th width="50%">演示后的图谱概览</th>
  </tr>
  <tr>
    <td width="50%" align="center" valign="top">
      <a href="pic/初始图谱.png"><img src="pic/初始图谱.png" alt="初始能力知识图谱概览，包含客户流失场景及历史算子与经验" width="100%"></a>
    </td>
    <td width="50%" align="center" valign="top">
      <a href="pic/演示后的图谱概览.png"><img src="pic/演示后的图谱概览.png" alt="演示后新增工单场景、生成算子和版本派生关系的知识图谱概览" width="100%"></a>
    </td>
  </tr>
</table>

右图新增了 `tickets` 场景，以及 `TotalChargesCleaner@2`、`TenureBucketizer@2`、`IdColumnDropper@2` 三个生成并注册的算子；虚线连接新旧版本，展示能力沉淀后的版本谱系。

## 4. Agent 工作流与调度

### 角色职责与输入输出

系统将规划、源码生成、验证和知识回写分配给四个组件，由 Workflow 统一调度。需求、方案和验证结果通过结构化对象传递；生成代码以源码字符串保存在 `GeneratedOperator` 中，交由 Verifier 检查。

| 角色 | 职责 | 用 LLM | 关键输出 |
| --- | --- | :-: | --- |
| **Planner** | 需求 + 数据画像 → `TaskSpec`；按步骤检索知识图谱；给出 N 个候选 `PipelinePlan`，每步说明 Reuse 还是 Synthesize 及理由 | 需求理解与方案生成 | `TaskSpec`、`list[PipelinePlan]` |
| **Synthesizer** | 为 synthesize 步骤按规格生成源码，有历史实现时参考它；修复可定位的本次生成算子，并带上检索到的同诊断码经验。reuse 步骤由执行层实例化 | 生成与修复 | `GeneratedOperator` |
| **Verifier** | 四级验证，输出带诊断码的 `Diagnostic`，能定位时附上算子名 | 否 | `ValidationReport` |
| **Curator** | 注册最优方案中验证通过的新算子；把所有候选的验证结果（含失败的）写回图谱；把成功的修复提炼为经验 | 仅提炼经验 | 验证记录 ID、注册记录、经验记录 |

Planner 的处理过程进一步分为三个阶段：

| 阶段 | 输入 | 输出 | 是否使用 LLM |
| --- | --- | --- | --- |
| `understand()` | 自然语言需求、数据画像、指定目标列、历史场景指标 | `TaskSpec`（任务规格） | 是 |
| `retrieve()` | `TaskSpec`、数据画像、关联场景 | `PlanningContext`（候选算子、证据和经验） | 否 |
| `propose()` | `TaskSpec`、画像、检索上下文，可选的失败反馈与父方案 | `list[PipelinePlan]`（候选执行方案列表） | 是 |

主要实现位于 [factory/agents/planner.py](factory/agents/planner.py)，由 Workflow 的 `_plan()` 依次调用。表中的 LLM 阶段是否请求远端 API，取决于[运行模式和缓存](#52-先选模式llm-的回答从哪里来)。

**验证依据与复现条件**：Verifier 以程序化规则和实际执行结果判定，诊断可追溯至具体检查项。固定代码、数据、配置、随机种子与依赖环境后支持复现；资源限制、平台差异仍可能影响运行结果。验证通过表示满足当前检查项，不代表覆盖所有业务正确性和安全风险。

各 Agent 的输入字段、输出 schema、校验方式和真实样例，见 **[docs/agents_io.md](docs/agents_io.md)**。

### 中央控制器：显式状态机

Agent 之间不直接互相调用，全部由 `factory/workflow.py` 调度，通过共享状态 `RunState`（黑板）交换数据：

```mermaid
stateDiagram-v2
    [*] --> PLAN
    PLAN --> BUILD: 候选方案
    BUILD --> VERIFY: 生成缺失算子
    VERIFY --> ROUTE
    ROUTE --> REPAIR: 生成算子有代码类错误 E_*
    REPAIR --> VERIFY: 只重验受影响的方案
    ROUTE --> REPLAN: 指标不达标 M_*（Beam 搜索一轮）
    REPLAN --> BUILD
    ROUTE --> CURATE: 有方案通过 / 无法继续
    CURATE --> [*]
```

### 中间表示、定向修复与方案搜索

1. **分层的结构化中间表示**（借鉴编译器）：`TaskSpec`（做什么）→ `PipelinePlan`（怎么做）→ 源码。前两层做 schema 和部分语义检查，源码再接受静态与执行验证，尽量提前发现问题。
2. **知识驱动的能力组合**：Planner 收到带分项得分和证据的检索结果，报告展示这些依据与生成理由；目前未逐条核验理由是否与证据一致。
3. **诊断驱动的定向修复**：能定位到本次生成算子、且未超次数的代码错误才进入修复。LLM 返回该算子的完整源码，随后重新验证受影响的完整方案；不保证最小代码 diff。先处理可修复候选，再选优。
4. **知识图谱先验引导的 Beam Search**（`factory/search.py`）：需要重新规划时，以排名前 k 的已验证方案为父方案，请求扩展 k × m 个子方案；按步骤、算子和参数签名去重，以场景历史记录和本次观测计算先验，最多保留 k 个验证。它限制的是每轮候选数量；新增的规划调用、不同模型与修复次数仍会影响耗时。当前先验是启发式估计，尚未证明优于无搜索基线。

### 四级验证

| 级别 | 检查内容 | 诊断码 |
| --- | --- | --- |
| 1 静态检查 | 语法；import 白名单；禁止 `eval/exec/open/getattr/__dunder__` 等调用；禁止库自带的文件读写、下载与字符串求值入口（`pd.read_*`、`df.to_*`、`np.load`、`df.query` 等）；接口齐全；`meta.params` 与 `__init__` 默认值一致 | `E_SYNTAX` `E_UNSAFE` `E_INTERFACE` |
| 2 沙箱执行 | 独立子进程，限制超时与内存，清除含 KEY/TOKEN/SECRET 的环境变量；子进程内装审计钩子（PEP 578），按行为拦截越权的文件、网络、子进程与动态库操作；小样本冒烟并定位出错算子 | `E_UNSAFE` `E_RUNTIME` `E_TIMEOUT` |
| 3 契约检查 | 不修改输入；行数一致；不新引入 NaN/Inf；`predict_proba` 为合法概率的 ndarray；能处理未见类别与缺失值 | `E_MUTATION` `E_SHAPE` `E_NAN` `E_PROBA` |
| 4 指标检查 | 多种子 × 分层 K 折交叉验证；阈值；种子间稳定性；指标过高疑似泄漏；耗时与内存 | `M_THRESHOLD` `M_UNSTABLE` |

当前支持二分类和多分类，验证配置从 `configs/tasks/<task_type>.yaml` 读取，可据此调整已有任务的验证参数。指标通过 `@metric("name")` 注册，并需在验证子进程中导入；新增回归、时序等任务还需扩展任务 schema、模型接口、数据切分、指标及验证流程。

使用时还需核对业务正类与指标是否受支持；当前验证在指标缺失、约束验收等方面的边界见 [Verifier 说明](docs/agents_io.md#当前验证范围)。

## 5. 环境配置和运行方法

以下命令都在**仓库根目录**执行。第一次体验建议先跑离线 `demo`；要处理自己的需求，再配置 API 并选择 `live`。

### 5.1 准备环境

```bash
# 首次使用：创建环境，同时安装依赖与本项目
conda env create -f environment.yml

# 每次打开新终端后激活；环境已存在时，直接从这一步开始
conda activate algoforge
```

环境使用 Python 3.11。激活后可用 `which python`（Linux/macOS）或 `where python`（Windows）检查解释器路径是否位于 `algoforge` 环境中。

### 5.2 先选模式：LLM 的回答从哪里来

通过 `--llm <模式>` 显式选择回答来源。**这控制的是 LLM 调用方式，不决定流水线是否包含新算子。**

| 模式 | 回答来源 | 是否请求远端 API | 适用情况 |
| --- | --- | --- | --- |
| `replay` | 读取 `examples/cassettes/` 中匹配的已录制响应 | 否，不需要 API Key；缺少录制时直接报错 | 复现已有示例 |
| `live` | 先读 `.cache/llm/`，未命中时请求 API 并缓存 | 未命中时请求；使用该模式需配置 API Key | 运行自己的新需求 |
| `record` | 先读 `examples/cassettes/`，未命中时请求 API 并录制 | 未命中时请求；使用该模式需配置 API Key | 为新任务准备离线回放 |
| `mock` | 本地规则提供需求理解、规划、源码和修复响应 | 否，不需要 API Key | 测试规则支持的场景和错误分支 |

**不同入口的默认值不同**：`demo`、`extract` 默认 `replay`；`run` 默认 `live`；Web 界面默认选中 `live`。下面的示例显式写出模式，避免混淆。

### 5.3 只看现有项目效果：离线演示

```bash
python -m factory.cli demo --llm replay
```

命令依次运行工单分类、客户流失预测和相似需求复用，结果写入 `runs/demo/`，知识图谱在该目录下的副本中更新。运行耗时随机器配置变化。

回放时，需求理解、方案和生成源码取自以前保存的 LLM 响应；**流水线组装、训练、验证和指标计算仍实际执行**，并重新生成 `pipeline.py`、`state.json`、`report.md` 与 `report.html`。因此，离线运行也会产生代码文件，但本次没有请求远端 LLM 新写源码。

`demo` 默认使用仓库中的基础图谱与录制请求。若修改基础图谱、模型配置或其他影响 prompt 的内容，现有录制可能不再匹配。

### 5.4 处理自己的需求：在线运行

先准备 API 配置：如果没有 `.env`，复制 `.env.example` 为 `.env`，填入自己的 `LLM_API_KEY`；已有 `.env` 时直接编辑，保留现有配置。

**API 平台**：本项目开发及录制示例使用 [XHub API 平台](https://api15.xhub.chat)，通过 OpenAI 兼容客户端调用。[默认配置](configs/llm.yaml) 为：

```dotenv
LLM_BASE_URL=https://api15.xhub.chat/v1
LLM_MODEL=gemini-3.5-flash
```

这两个变量可按需写入 `.env` 覆盖配置；更换模型或接口后需验证兼容性。`LLM_API_KEY` 从环境变量读取，也支持由 `.env` 加载。

```bash
# 请求远端接口检查连通性并列出模型，不执行建模任务
python -m factory.cli llm-ping

# 处理需求；首次试用不持久修改基础知识图谱
python -m factory.cli run "预测30天内可能流失的客户，需可解释，召回优先" \
  --data data/churn.csv --llm live --no-writeback
```

这次运行会获取 Planner 的需求理解与方案：

- 步骤为 **Reuse**：从注册表实例化已有算子，在当前数据上拟合；不需要为这一步编写源码。
- 步骤为 **Synthesize**：获取 Synthesizer 的源码响应；缓存未命中时才请求远端 LLM 编写。
- 出现可修复的生成代码错误：按诊断获取修复响应，再验证。

**`live` 是否请求 API 取决于缓存是否命中，是否生成新算子取决于规划中的步骤标记。** `--no-synthesis` 只禁止生成新算子，需求理解和规划仍使用所选 LLM 模式；离线回放需要选择 `replay`。

结果默认写到 `runs/<run_id>/`。`run` 默认把验证结果与符合条件的新算子写回 `kg/capability_kg.json` 及同目录的 `operators/`；加 `--no-writeback` 保留运行报告和脚本，但不持久修改基础能力库。

### 5.5 保存新任务，之后离线复现：先录制，再回放

```bash
# 首次录制：匹配不到已有响应的请求会调用 API，并保存到 examples/cassettes/
python -m factory.cli run "预测30天内可能流失的客户，需可解释，召回优先" \
  --data data/churn.csv --candidates 1 --llm record --no-writeback

# 保持需求、数据、模型配置、知识图谱和其他参数一致，再切换为 replay
python -m factory.cli run "预测30天内可能流失的客户，需可解释，召回优先" \
  --data data/churn.csv --candidates 1 --llm replay --no-writeback
```

两条命令都使用 `--no-writeback`，避免第一轮回写知识图谱改变第二轮的检索上下文。`record` 发现已有匹配响应时也会直接读取，并不强制刷新。

回放按**模型、消息内容与 JSON 输出模式**匹配录制响应，文件名还带调用标签。修改需求、数据画像、候选数、知识图谱或提示词，都可能产生新的请求。若出现 `no recorded response`，说明没有对应录制，**不会自动转为在线调用**；要么恢复原上下文，要么配置 API 后用 `record` 补充该请求。

`live` 的缓存与 `record` 的录制使用不同目录：只运行过 `live`，不代表该任务已经可以用 `replay` 复现。

### 5.6 Web 界面与其他命令

```bash
streamlit run app.py
```

启动页面本身不会执行 LLM 建模任务。在“运行”页选择模式并点击“开始”才启动工作流；默认选中 `live`。已有录制只适用于匹配的上下文，不保证 Web 中的任意输入都可回放；复现三个固定任务时优先使用 `demo --llm replay`。Web 默认在图谱副本上运行，勾选“回写知识图谱”后才持久修改所选图谱。

| 命令 | 说明 |
| --- | --- |
| `python -m factory.cli extract --llm replay` | 读取已有录制抽取响应，从 `data/materials/` 重建图谱；默认覆盖 `kg/`，可用 `--out <路径>` 指定输出 |
| `python -m factory.cli extract --llm record` | 对新材料获取抽取响应并录制，未命中时调用 API；需配置 Key |
| `python -m factory.cli kg stats / search / lessons / tidy / viz` | 查看、检索、整理图谱或生成可视化，不调用 LLM；其中 `tidy` 会保存整理结果 |
| `python -m factory.cli report runs/<run_id>` | 从已保存状态重新生成报告，不调用 LLM |
| `python -m factory.cli operators` | 列出当前注册表中的算子，不调用 LLM |
| `pytest` / `pytest -m slow` | 118 个常规测试 / 1 个完整演示回放测试，不需要远端 LLM |

`run` 还支持 `--target <目标列>`、`--candidates 3`、`--max-repairs 3`、`--max-replans 2`、`--beam-width 2 --beam-expand 3` 和 `--kg <路径>`；完整参数见 `python -m factory.cli run --help`。

SDK 报错会脱敏，请求缓存键不含密钥；沙箱会清除名称含 KEY / TOKEN / SECRET 等的环境变量。`tests/test_secrets.py` 扫描待跟踪文件中的已配置密钥和常见密钥格式。

**平台支持**：代码包含 Windows 的 psutil 内存看门狗分支，Linux/macOS 使用 `RLIMIT_AS` 资源限制；当前记录的运行结果来自 Linux，其他平台需在对应环境验证。上面的多行命令采用 Bash 续行写法，其他终端可合并为一行执行。

## 6. 示例数据和测试任务说明

数据与材料由脚本生成、随机种子固定（详见 [data/README.md](data/README.md)）。示例包含字符串数值、缺失值、高基数 ID、类别不平衡和标签噪声，用于观察历史知识对规划与生成的影响：

| 数据 / 材料 | 说明 |
| --- | --- |
| `data/churn.csv` | 5000 行电信客户数据，流失率 26.6%；`total_charges` 是字符串且新客户为空白、部分列缺失、高基数 `customer_id` |
| `data/tickets.csv` | 2000 条中文客服工单，4 类、不均衡，含跨类表述与约 6% 标签噪声 |
| `data/materials/legacy_repo/` | 模拟的历史代码仓库 `churn_v1`：清洗、特征、目标编码、LightGBM、阈值调整、recall@k 等 9 个函数 |
| `data/materials/docs/` | 需求说明、实验记录、失败复盘、调参经验 4 篇文档 |

| 测试任务 | 需求 | 考察点 |
| --- | --- | --- |
| 客户流失 | "预测30天内可能流失的客户，需可解释，召回优先" | 阈值来自图谱中的需求文档（recall ≥ 0.75）；脏数据需要生成清洗算子；可解释约束偏向线性模型 |
| 客户流失（相似需求） | "预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户" | 复用上一次沉淀的算子 |
| 工单分类 | "把客服工单自动分到四类，宏平均 F1 不低于 0.8" | 跨场景迁移：新场景无场景知识，依靠通用经验 |

## 7. 生成算法代码示例

工作流有可导出的候选方案时，会生成 `pipeline.py`，在已安装本项目及依赖的环境中可独立调用：内置算子以 import 引用，生成的算子源码内嵌，每一步附上 Planner 的理由。完整代码见 [03 相似需求脚本](examples/demo/runs/03_churn_reuse/pipeline.py)。以下仅节选复用步骤：

```python
# 从完整 Pipeline 的步骤中节选；前后步骤见链接中的脚本。
("1_imputer", TotalChargesCleaner(col='total_charges', fill_value=0.0)),
("2_imputer", MissingValueImputer()),
```

`TotalChargesCleaner@2`、`TenureBucketizer@2` 是任务 02 中由 LLM 参考历史代码生成、验证通过后注册的算子（源码在 [`examples/demo/kg/operators/`](examples/demo/kg/operators/)），任务 03 直接复用。报告保留了检索经验和 Planner 理由，供人工核对。

```bash
python examples/demo/runs/03_churn_reuse/pipeline.py --data data/churn.csv            # 训练并保存 model.joblib
python examples/demo/runs/03_churn_reuse/pipeline.py --predict data/churn.csv          # 预测
```

**修复与搜索示例**：当前三个录制演示没有触发修复或重新规划。另附[规则 mock 的运行节选](examples/verification/README.md)，展示生成 → `E_NAN` → 修复 → 验证通过 → 注册 → 再次复用，以及指标失败后的 Beam 扩展与剪枝。样例经过真实验证器执行，但规划、代码和经验文本由规则提供，不能据此评估真实 LLM 的修复成功率或搜索收益。

## 8. 验证结果和报告样例

每次运行在运行目录生成 `report.md` 与 `report.html`（同一份内容的两种渲染）：需求理解、知识检索的分项得分与证据、候选比较（均值 ± 种子间标准差、指标条形图）、最优方案每一步的理由、生成算子与修复 diff、方案搜索记录、四级验证配置与资源消耗、知识沉淀、生成的代码与运行记录。

样例：[任务 02 报告](examples/demo/runs/02_churn_synthesize/report.md)（[HTML](examples/demo/runs/02_churn_synthesize/report.html)），节选：

| 排名 | 方案 | 结果 | recall | roc_auc | 单次训练 (s) |
| --- | --- | --- | --- | --- | --- |
| 1 | ★ interpretable_logistic_regression | 通过 | 0.778 ± 0.001 | 0.814 ± 0.001 | 0.088 |
| 2 | cost_ratio_random_forest | 未通过（`M_THRESHOLD`） | 0.714 ± 0.003 | 0.804 ± 0.001 | 0.612 |
| 3 | balanced_lightgbm_boosting | 未通过（`M_THRESHOLD`） | 0.630 ± 0.003 | 0.783 ± 0.004 | 0.259 |

在"可解释 + 召回优先 + 不平衡"的约束下，按图谱经验做了标准化、并设置 `class_weight='balanced'` 的逻辑回归是唯一达标的方案，两个树模型的 recall 都没到 0.75。

全部演示结果见 [examples/demo/SUMMARY.md](examples/demo/SUMMARY.md)。测试覆盖：schema 校验、算子契约、验证器对各类坏代码给出正确诊断码（静态安全违规、运行时越权读写文件 / 联网 / 起进程、崩溃、修改输入、行数错误、NaN、超时、指标不达标、泄漏）、知识图谱读写、融合与检索、Planner 的语义校验与重试、Beam 搜索的去重 / 先验 / 剪枝、生成 → 修复 → 注册 → 复用全流程、报告渲染与 HTML 转义、Web 界面、离线回放与录制结果一致。

## 9. 实现挑战与处理方式

| 挑战 | 实现要求 | 处理方式 |
| --- | --- | --- |
| LLM 输出不稳定：引用不存在的算子、使用非法参数名、在只允许复用时生成 | 在执行前检查结构和引用合法性 | 需求与方案使用 pydantic 校验及相应语义检查，错误反馈给 LLM 重试；生成源码另做静态与执行验证；真实录制中可见方案校验失败后的更正 |
| LLM 编造指标阈值（给只需"报告"的 recall@20% 设 0.7，导致方案失败） | 次要指标门槛需要来源依据 | 移除未通过需求文本或图谱来源检查的次要指标要求；是否继续计算该指标取决于验证配置的 `report_metrics` |
| 生成代码存在接口与数据处理缺陷：概率返回 DataFrame、默认参数不一致、转换后留下 NaN | 检查接口契约并提供错误位置 | 静态与契约检查定位到具体算子，诊断附带修复提示 |
| 文件访问入口可绕过单一函数黑名单，如 `pd.read_csv('.env')` | 同时检查源码写法与运行行为 | 静态检查、子进程审计钩子与资源限制共同检查越权行为；注册前再次静态复查；Web 界面通过 `data:` URL 展示生成内容 |
| 重新规划时 LLM 容易给出重复或盲目的新方案 | 历史验证记录可作为候选筛选依据 | Beam 搜索：以排名靠前的方案为父方案，请求局部修改，按签名去重、按启发式先验剪枝，每轮最多验证 k 个；实际耗时和效果需另做对照实验 |
| 多来源抽取带来重复和冲突的知识 | 减少重复实体和无效历史证据对检索的影响 | 指标名规范化、经验按"诊断码 + 标题相似度"合并、初始场景阈值取自需求文档、被推翻的实验不计分；`kg tidy` 整理已有图谱 |
| 评审复现依赖 LLM 与 API Key | 支持无需远端请求的示例复现 | 录制回放：请求哈希不含密钥与地址；`demo --llm replay` 离线复现，测试校验回放结果与仓库一致 |
| 合成数据较易分类，F1 = 1.0 触发疑似泄漏检查 | 增加分类难度以检验候选方案差异 | 工单数据加入跨类表述与标签噪声 |
| 64 核服务器上 LightGBM / OpenMP 线程与内存上限冲突 | 控制正常运行时的资源占用 | 沙箱中限制线程数与 malloc arena |

## 10. 后续可扩展方向

- **方案搜索**：先验目前只基于逐算子的历史统计；可以学习算子组合的交互效应，或改为 MCTS 并用验证结果回传更新先验
- **更强的隔离**：审计钩子是进程内防线（C 扩展的底层调用不经过它），生产环境应再加容器或 gVisor 与网络命名空间隔离
- **更多任务类型**：回归、时序预测、异常检测需扩展任务 schema、模型接口、数据切分与验证流程，再增加任务配置、指标和算子；图像缺陷识别还需要新的数据模态
- **真实代码仓库抽取**：支持类、跨文件调用关系、git 历史与测试用例；接入 Neo4j 承载更大规模图谱
- **人机协同**：在界面中审阅、编辑 Planner 的方案和生成代码后再验证；经验的人工审核与淘汰
- **部署**：从 `pipeline.py` 自动生成 FastAPI 服务、Dockerfile 与接口文档

开发过程与各里程碑的实际结果见 [docs/roadmap.md](docs/roadmap.md)。
