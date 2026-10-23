# 开发路线图

本文件记录 AlgoForge 的里程碑规划与完成情况，每完成一个里程碑同步更新。

状态：✅ 已完成　⏳ 未开始（全部里程碑已完成）

## 目标

跑通题目要求的闭环：**能力抽取 → 能力复刻 → 验证 → 沉淀**。

核心机制是 Reuse / Synthesize：任务需要的每个步骤，如果知识图谱里已有合适的算子就直接复用；没有就由 LLM 按规格生成新算子，验证通过后注册，下次即可复用。

```
NL 需求 → TaskSpec → PipelinePlan ─┬─ Reuse：从知识图谱取已有算子
                                  └─ Synthesize：LLM 按规格生成算子
                                          ↓
                 组装 Pipeline → 验证 → 诊断 ─┬─ 代码错误 → 修复该算子
                                             ├─ 指标不达标 → 重新规划
                                             └─ 通过 → 注册算子，回写知识图谱
```

## 里程碑

| # | 里程碑 | 状态 | 主要产物 |
| --- | --- | --- | --- |
| 1 | 骨架 | ✅ | `factory/schemas.py`、`factory/operators/`、`factory/llm.py` |
| 2 | 数据与能力抽取 | ✅ | `data/`、`factory/extract.py`、`factory/kg.py`、`kg/`、`docs/schema.md` |
| 3 | 最小闭环 | ✅ | `factory/agents/{planner,rules,curator}.py`、`factory/verify/`、`factory/workflow.py`、`factory/profile.py`、`factory/codegen.py` |
| 4 | 生成与修复 | ✅ | `factory/agents/synthesizer.py`、`factory/agents/curator.py`、`factory/operators/synthesized.py` |
| 5 | 候选比较与报告 | ✅ | `factory/report.py`、`examples/demo/runs/*/report.{md,html}` |
| 6 | 界面与跨场景迁移 | ✅ | `app.py`、`factory/kg_viz.py`、`examples/demo/runs/01_tickets_transfer/` |
| 7 | 收尾 | ✅ | `README.md`、`factory/demo.py`、`examples/demo/`、`examples/cassettes/` |

### 1. 骨架 ✅

- Miniconda3 环境 `algoforge`（`environment.yml` + `requirements.txt`）
- 三层结构化中间表示：`TaskSpec`（做什么）→ `PipelinePlan`（怎么做）→ 代码；诊断码分为代码类和指标类
- 算子基类与按版本管理的注册表，7 个内置算子
- LLM 访问层：OpenAI 兼容客户端，结构化输出校验与重试，缓存、录制回放、MockLLM
- API Key 只从 `.env` 读取，报错时脱敏，并有扫描测试防止泄露

### 2. 数据与能力抽取 ✅

- 可复现的电信客户流失数据（5000 行，保留空白 `total_charges` 等脏点）和客服工单数据
- 模拟的历史代码仓库（9 个函数）与 4 篇业务文档
- 抽取：AST 解析代码，LLM 归纳能力卡片；LLM 从文档抽取指标、约束、历史实验和经验
- 知识图谱：8 类节点、9 类边，检索综合文本相关度、条件匹配和历史验证分

### 3. 最小闭环 ✅

- Planner：自然语言需求 → `TaskSpec` → `PipelinePlan`，依据知识图谱检索结果决定复用哪些算子，并给出设计依据
- Verifier：四级验证
  1. 静态检查：AST 安全检查、接口是否齐全
  2. 沙箱执行：子进程运行，限制超时和内存
  3. 契约检查：输出形状、概率范围、不修改输入、能处理缺失值和未见类别
  4. 指标检查：是否达到阈值、多个随机种子下是否稳定、耗时
- Curator：把验证结果（`ValidationRun`）回写知识图谱
- 工作流状态机：plan → build → verify → route →（replan →）curate；指标不达标时带着诊断反馈重新规划
- 先用 Mock 跑通，再接真实 API 跑客户流失场景

实际结果（真实 API，`gemini-3.5-flash`）：
- 客户流失：LLM 解析出 recall ≥ 0.75（阈值来自 KG 中的需求文档）、可解释、召回优先；两个候选中逻辑回归通过（recall 0.777±0.002），随机森林未达标（0.724），结果写回 KG
- 工单分类（迁移场景，宏召回 ≥ 0.85 的严格要求）：首轮两个方案均为 0.83，触发重新规划，LLM 根据反馈调整 n-gram 和正则化强度，仍为 0.833，判定失败并如实回写——这一差距留给里程碑 4 的 Synthesize 处理
- 生成的 `runs/<run_id>/pipeline.py` 可独立训练与预测

### 4. 生成与修复 ✅

- Synthesizer：为缺失步骤生成算子代码。可参考知识图谱里抽取到的历史代码，例如 `TotalChargesCleaner`
- 诊断分流：代码类错误只修复出错的算子，最多 N 轮；指标类问题回到规划阶段，换算子或调参
- 修复时检索同一诊断码的经验放进 prompt；结束后把"诊断 → 修复"提炼成新经验写回
- 新算子验证通过后注册为可执行算子，并记录版本谱系（`DERIVED_FROM`）

实际结果：
- 真实 API 第 1 次运行：Planner 参考 KG 中抽取的历史代码，生成 `TotalChargesCleaner@2`、`TenureBucketizer@2`、`ChargeRatioFeatureGenerator@2`，最优方案通过后注册前两个（`kg/operators/`），并记录 `DERIVED_FROM → @1`
- 真实 API 第 2 次运行（换一种说法的相似需求）：直接 Reuse 上次注册的两个算子，只为新方案生成其余步骤
- 真实 LLM 修复：LLM 生成的 `BalancedLightGBMClassifier` 的 `predict_proba` 返回了 DataFrame，被契约检查判为 `E_SHAPE` 并定位到该算子；带诊断调用 `Synthesizer.repair` 一轮修好，剩下的只是指标不达标（交给重新规划）
- mock 模式稳定复现完整闭环：生成 → `E_NAN` → 修复 → 通过 → 注册 → 提炼 Lesson；再次运行直接复用（见 `tests/test_synthesis.py`）

### 5. 候选比较与报告 ✅

- Planner 每次产出 2–3 个候选方案，全部验证后按指标选出最优
- 生成 Markdown / HTML 验证报告：各候选的指标对比、诊断记录、修复过程、设计依据

实际结果：
- 统一的候选排序规则 `rank_reports`：是否通过 → 主指标 → 次要指标之和 → 训练耗时
- 每次 `run` 自动在运行目录生成 `report.md` 与 `report.html`（同一份内容结构渲染两种格式，HTML 自带深色模式）；`report` 命令可根据 `state.json` 重新生成
- 报告包含：需求理解、知识检索的分项得分与证据（设计依据）、候选比较表与指标条形图、最优方案每一步的理由、生成算子与修复 diff、四级验证配置与资源消耗、知识沉淀情况、生成的代码与运行记录
- 样例：`examples/demo/runs/02_churn_synthesize/`（真实 API，3 个候选：逻辑回归通过，随机森林与生成的 BalancedLightGBM 未达标；里程碑 7 重新录制）

### 6. 界面与跨场景迁移 ✅

- CLI：`run`、`report` 等命令
- Streamlit 界面：提交需求、查看过程、报告和知识图谱
- 知识图谱可视化（pyvis）
- 用工单文本分类任务验证同一套框架可以迁移到其他场景

实际结果：
- `streamlit run app.py`：运行页（示例需求一键填入、可上传 CSV、实时显示各阶段进度、内嵌验证报告与代码下载）；能力库页（算子/经验列表、检索、可交互知识图谱，可按类型过滤、按节点聚焦邻域）；历史运行页
- 默认不回写：界面上的运行在知识图谱副本上进行，勾选后才写回
- 报告与图谱以 `data:` URL 嵌入 iframe，与应用不同源（其中有 LLM 生成的文本）
- `python -m factory.cli kg viz` 生成 `kg/capability_kg.html`
- 跨场景迁移（真实 API）：工单数据没有任何场景知识，系统自动识别为中文文本多分类、建立新场景；检索到调参笔记中"中文短文本用字符级 TF-IDF + 逻辑回归"的经验，最优方案 F1-macro 0.820 通过
- 开发时的一次真实运行中出现了"失败 → 修复 → 通过"：LLM 生成的 `ChineseTextPreprocessor` 使用了 `getattr()`，被静态检查判为 `E_UNSAFE`；带诊断修复一轮后改用 `try/except AttributeError`，通过验证，Curator 将其提炼为新经验。里程碑 7 因知识合并重新录制后，LLM 对同一任务改为给出纯 TF-IDF 方案（又试了 3 次均如此），该修复示例不再出现在录制结果中；修复闭环由 `tests/test_synthesis.py` 的确定性 mock 用例覆盖

### 7. 收尾 ✅

- 录制全流程的 LLM 回放数据，保证离线可复现
- README 覆盖题目要求的 10 项内容
- `examples/` 放端到端产物：生成的代码和验证报告样例
- 补全测试

实际结果：
- `python -m factory.cli demo`：在知识图谱副本上依次运行工单迁移（含真实修复）、客户流失（3 候选，生成并注册算子）、相似需求（复用），输出摘要 `SUMMARY.md`
- 重新录制：只保留实际用到的 28 条 LLM 响应；`demo --llm replay` 离线回放，结果与 `examples/demo/` 逐项一致（测试 `test_demo_replay_matches_examples`）
- 修复代码导出缺陷：参数中的 `float('inf')` 原来被写成 `inf`，导出的脚本无法运行；三个演示的 `pipeline.py` 都已验证可独立训练与预测
- README 覆盖题目要求的 10 项；测试共 90 个（`pytest` 跑常规 89 个，`pytest -m slow` 跑完整演示回放）

## 验收方式

- `pytest -q`：schema 校验、算子契约，验证器对各类坏代码给出正确诊断码，知识图谱回写，基于 Mock 的端到端运行
- 离线演示：`python -m factory.cli run "预测30天内可能流失的客户，需可解释，召回优先" --data data/churn.csv --llm replay` 能生成报告，知识图谱中新增验证记录和经验
- 在线演示：客户流失和工单分类各跑一次真实 API，至少出现一次 Synthesize 和一次"失败 → 修复 → 通过"；再次运行相似需求时直接复用上次注册的算子，并命中经验

## 与原计划的调整

| 调整 | 原因 |
| --- | --- |
| 知识图谱新增 `RUN_ON` 边（验证记录 → 场景） | 需要按场景统计算子的历史表现 |
| 文本检索从 BM25 改为 BM25+ | 候选很少时，BM25 的 IDF 可能为 0 或负数，排序失效 |
| 场景的指标阈值和业务约束只从需求文档抽取 | 实验记录、调参笔记里的描述会混入场景定义 |
| 默认模型改为 `gemini-3.5-flash` | 原占位模型不在接口的可用模型列表中 |
| 新增数据画像 `factory/profile.py` | 检索条件与 prompt 中的数据事实由确定性代码给出，不让 LLM 猜 |
| 新增规则版 Planner `factory/agents/rules.py` | mock 模式解析与真实 LLM 相同的 prompt 上下文，离线也走完整链路 |
| 沙箱子进程清除含 KEY/TOKEN/SECRET 的环境变量 | 防止生成的代码读取 API Key |
| 工单数据加入跨类表述与 6% 标签噪声 | 原数据过于简单，F1 = 1.0，被泄漏检测判为异常 |
| 生成算子沉淀为 `kg/operators/<Name>_v<版本>.py` | 能力库可跨进程复用；加载前重新做静态检查 |
| 生成的算子版本号与 KG 中同名算子顺延 | 抽取的历史实现记为 v1，生成的从 v2 开始，形成版本谱系 |
| 契约检查新增：逐算子检查 NaN/Inf、模型输出必须是 ndarray、`meta.params` 与 `__init__` 默认值一致 | 都是开发中真实遇到的生成代码缺陷 |
| 次要指标阈值必须有出处 | LLM 曾给只需"报告"的 recall@20% 编造 0.7 的阈值，导致所有方案失败 |
| 重新规划后旧方案不再验证和修复 | 避免重复计算和无效修复 |
| 修复与选优的顺序：先修复所有可修复的候选，再比较 | 保证候选之间比较公平 |
| 静态检查的 `E_UNSAFE` 诊断带上修复提示 | 修复时 LLM 能直接看到可用的替代写法 |
| Lesson 的 `LEARNED_FROM` 指向用了该算子的方案 | 原来固定连到第一条验证记录，可能指错 |
| 经验写入时按"诊断码 + 标题相似度"去重；指标名按参数规范化；`kg tidy` 整理已有图谱 | 两份文档抽出重复经验、同一指标出现多个名字；`recall_at_top_k` 与 `recall_at_20` 合并 |
| 生成算子与图谱中已有算子仅大小写不同时，沿用原名并以最新版本为 `reference` | 修复了 `IdColumnDropper@2` 没有接入版本谱系的问题 |
| 沙箱支持原生 Windows：无 `resource` 时改用 psutil 看门狗限制内存，峰值内存改用 psutil 统计 | 原实现依赖 Linux 的 `resource` 与 `preexec_fn`，Windows 上无法导入 |
