# KG 决策、真实修复与搜索评测

本目录保存 2026-10-26 使用 `gemini-3.5-flash`（temperature=0.8）的真实调用录制与执行结果：共 24 次运行、110 次远端调用、545189 tokens。[汇总](SUMMARY.md)（[HTML](SUMMARY.html)）列出每一次运行；失败和未触发修复的运行全部保留。

## 结果

| 实验 | 次数 | 结果 |
| --- | ---: | --- |
| 完整 KG / 移除 Lesson / 仅算子目录 | 每组 3 次 | 完整 KG 与移除 Lesson 都选 LightGBM，recall 0.603；仅算子目录选随机森林加自行生成的清洗器，recall 0.495–0.597，并触发真实修复（`E_NAN`、`E_RUNTIME`，修复后通过）。九次都未达到 0.75 |
| 自然失败探测 | 3 个需求变体 | 2 次出现生成代码错误并修复；1 次整体通过 |
| 首个自然失败任务的重复 | 3 次 | 均触发修复，修复的算子不在最优方案中，未注册 |
| 受控 NaN 注入 → 修复 → 注册 | 3 次 | 从已通过验证的真实生成方案出发，注入 NaN 后由真实 LLM 修复，3 次都通过并注册 |
| 新进程复用修复后的算子 | 3 次 | 都加载并复用修复后的 `TotalChargesCleaner@2`，recall ≈ 0.778 |
| Beam 搜索（recall ≥ 0.80） | 3 次 | 首轮 LightGBM 0.624 未达标；扩展时 LLM 换成逻辑回归并把正类权重调到 4，2 次达到 0.848 通过。第 3 次入选方案出现 MRO 加载错误和缺失值未填补，修复 1 轮后仍未达标 |
| 固定搜索子方案池、去掉 KG 历史 | 3 次搜索共 4 轮 | 3 轮入选集合改变 |

## 读法

- **KG 参与决策**：消融和搜索干预都显示 KG 改变了方案和剪枝结果。
- **修复**：自然缺陷与人为注入分开统计；注入实验用于稳定验证"修复 → 注册 → 复用"整条链。
- **独立重复**：API 网关对内容相同的请求返回缓存回复，提高 temperature 也一样。注入、复用和搜索任务在请求中加入试验编号（`job.json` 中的 `nonce`）以得到独立样本；KG 消融与 `repair_natural` 没有加编号，三次重复的回复相同，实际相当于一个样本。
- **搜索设置**：门槛 0.80 按数据估算设定（正类占 27%，平衡权重逻辑回归约 0.778，正类权重 4 约 0.85）；beam 宽度 2，最多两轮扩展。

## 原始证据

- [清单与环境](manifest.json)：运行顺序、需求文本、条件、数据与基础图谱的 SHA256、依赖版本。KG 消融固定 TaskSpec，排除 KG 改变验收阈值带来的混杂。
- [逐次统计](results.json)：状态、修复是否解决原诊断、注册与复用、调用量、tokens、耗时。
- `<运行名>/runs/<运行名>/`：报告、`state.json`、生成源码与修复 diff；注入运行另有 `injection.json`，搜索运行另有 `intervention.json`。
- `cassettes/<运行名>/`：原始请求、响应、请求哈希、耗时与 usage。
- `demo_initial_attempt/`：早期一次演示录制尝试，不计入上述 24 次。

## 复现

离线回放，无需 API Key；输出目录需为空：

```bash
python -m factory.cli evaluate --llm replay --out runs/evaluation-replay
python -m pytest -m slow tests/test_evaluation_replay.py
```

每个任务在独立子进程中运行，重新执行数据画像、方案校验、沙箱训练与验证、修复路由和知识回写；只有 LLM 响应来自录制。

重新录制（需配置 `.env`，使用新的证据目录）：

```bash
python -m factory.cli evaluate --llm record --temperature 0.8 \
  --out runs/evaluation-new --evidence runs/evaluation-new-evidence --repeats 3
```
