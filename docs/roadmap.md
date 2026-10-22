# 开发路线图

本文件记录 AlgoForge 的里程碑规划与完成情况，每完成一个里程碑同步更新。

状态：✅ 已完成　⏳ 未开始

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
| 4 | 生成与修复 | ⏳ | `factory/agents/synthesizer.py`、`factory/agents/curator.py` |
| 5 | 候选比较与报告 | ⏳ | `factory/report.py`、`examples/` |
| 6 | 界面与跨场景迁移 | ⏳ | `factory/cli.py`、`app.py`、KG 可视化 |
| 7 | 收尾 | ⏳ | `README.md`、录制回放数据、测试补全 |

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

### 4. 生成与修复 ⏳

- Synthesizer：为缺失步骤生成算子代码。可参考知识图谱里抽取到的历史代码，例如 `TotalChargesCleaner`
- 诊断分流：代码类错误只修复出错的算子，最多 N 轮；指标类问题回到规划阶段，换算子或调参
- 修复时检索同一诊断码的经验放进 prompt；结束后把"诊断 → 修复"提炼成新经验写回
- 新算子验证通过后注册为可执行算子，并记录版本谱系（`DERIVED_FROM`）

### 5. 候选比较与报告 ⏳

- Planner 每次产出 2–3 个候选方案，全部验证后按指标选出最优
- 生成 Markdown / HTML 验证报告：各候选的指标对比、诊断记录、修复过程、设计依据

### 6. 界面与跨场景迁移 ⏳

- CLI：`run`、`report` 等命令
- Streamlit 界面：提交需求、查看过程、报告和知识图谱
- 知识图谱可视化（pyvis）
- 用工单文本分类任务验证同一套框架可以迁移到其他场景

### 7. 收尾 ⏳

- 录制全流程的 LLM 回放数据，保证离线可复现
- README 覆盖题目要求的 10 项内容
- `examples/` 放端到端产物：生成的代码和验证报告样例
- 补全测试

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
