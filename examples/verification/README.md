# 修复与方案搜索的规则 mock 样例

这里补充三个真实 LLM 录制演示未触发的分支：定向修复、修复经验提炼、指标失败后的 Beam Search。

以下结果来自 2026-10-25 在 Linux 上执行现有测试。规划、源码、修复与经验文本由 `rule_based_mock()` 提供；数据画像、检索、状态机、子进程验证、指标评估和知识回写执行项目实际代码。它们验证流程分支，不用于衡量真实 LLM 的成功率。

## 复现与数据

在项目环境和仓库根目录运行：

```bash
python -m pytest \
  tests/test_synthesis.py::test_synthesize_repair_register_then_reuse \
  tests/test_search.py::test_workflow_replan_runs_beam_search \
  tests/test_search.py::test_prior_uses_kg_history_and_skips_invalidated
```

本次结果为 **3 passed**。测试在 pytest 临时目录中生成 CSV、知识图谱副本、`state.json`、报告和流水线脚本，不改动基础图谱。

数据由 [tests/conftest.py](../../tests/conftest.py) 的固定随机种子生成，共 600 行，包含数值、类别、缺失值、客户 ID 和带空白的 `total_charges`。验证配置使用 `fast_verify`：3 折、种子 `[0, 1]`、120 行冒烟样本。这里的分数不能与 `data/churn.csv` 的 5000 行演示直接比较。

[repair.json](repair.json) 和 [beam.json](beam.json) 是实际运行状态的结构化节选，保留配置、诊断、指标和搜索记录，省去运行 ID、临时路径、资源耗时及大段源码。下文表格为便于阅读对数字做了四舍五入；跨环境重跑可能有差异。

## 定向修复与后续复用

对应 [test_synthesize_repair_register_then_reuse](../../tests/test_synthesis.py)，需求为“预测流失，需可解释，召回率不低于 0.4”。

| 阶段 | 本次观测 |
| --- | --- |
| 生成 | 首版 `TotalChargesCleaner@2` 的 `fill_value=None`，数值转换后留下新的 NaN |
| 诊断 | Verifier 返回 `E_NAN`，阶段为 `contract`，定位到 `TotalChargesCleaner@2` |
| 修复 | 规则 mock 将 `meta.params` 与 `__init__` 中的 `fill_value` 同时改为 `0.0`，修订号变为 1 |
| 重新验证 | 受影响方案通过，recall 为 0.5985，高于门槛 0.4；`RepairRecord.fixed=true` |
| 沉淀 | 注册 `TotalChargesCleaner@2`，记录版本关系、验证结果和一条修复经验 |
| 第二次运行 | 从已保存的源码加载并 reuse `TotalChargesCleaner@2`，没有新生成或修复，recall 仍为 0.5985 |

修复差异节选如下，完整 diff 见 [repair.json](repair.json)：

```diff
-        params={"columns": ['total_charges'], "fill_value": None}, parent="TotalChargesCleaner@1",
+        params={"columns": ['total_charges'], "fill_value": 0.0}, parent="TotalChargesCleaner@1",
-    def __init__(self, columns=['total_charges'], fill_value=None):
+    def __init__(self, columns=['total_charges'], fill_value=0.0):
```

这个样例只经历一轮修复。代码支持每个生成算子设置多轮上限，但不能从本次结果推断多轮复杂错误的修复成功率。第二次运行证明了源码持久化与复用，没有测量其时间或成本收益。

## Beam Search

对应 [test_workflow_replan_runs_beam_search](../../tests/test_search.py)，需求故意设置为“召回率不低于 0.99”，配置为初始候选 1 个、最多重规划 1 轮、扩展倍数 3，beam 宽度默认等于初始候选数。

该测试先完成一次清洗算子的 `E_NAN` 修复，随后随机森林候选的 recall 为 0.4091，触发指标失败后的重新规划。规则 Planner 扩展出三个候选：

| 子方案修改 | 先验 recall | 是否入选验证 | 实际 recall |
| --- | ---: | --- | ---: |
| 随机森林 `max_depth=8` | 0.6996 | 否，超出宽度 1 | 未执行 |
| 随机森林 `n_estimators=500, max_depth=12` | 0.6996 | 否，超出宽度 1 | 未执行 |
| 模型替换为逻辑回归 | 0.8448 | 是 | 0.6061 |

最终运行状态为 **failed**：入选方案仍未达到 0.99。它证明了父方案扩展、先验排序、剪枝和再验证路径，没有证明搜索一定找到达标方案。

这个工作流测试的临时数据路径没有匹配已有场景，因此先验使用本次观测和阈值平滑。逻辑回归没有本次观测时，以 0.99 阈值作为平滑基值，导致先验偏乐观；先验不是可保证的实际指标。详细记录见 [beam.json](beam.json)。签名去重由同文件中的 `test_select_dedups_and_prunes_to_width` 单独验证。

知识图谱历史先验另由 `test_prior_uses_kg_history_and_skips_invalidated` 验证。该单测人工构造 `telecom_churn` 场景，提供逻辑回归 recall=0.80、随机森林 recall=0.60 的记录，并排除一条被作废的 0.99 历史实验。阈值为 0.75 时，计算出的方案先验分别为 **0.7688** 和 **0.6938**；相关输入与输出也保存在 `beam.json`。这验证了历史证据进入打分的规则，尚未评估这种规则相对其他搜索策略的收益。
