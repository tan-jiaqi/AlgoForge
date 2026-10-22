"""端到端演示：在知识图谱副本上依次运行三个任务，展示能力的生成、修复、沉淀与复用。

1. 工单文本分类（跨场景迁移）：新场景，没有场景知识，靠通用经验规划；生成的算子出现代码错误并被修复；
2. 客户流失（3 个候选）：参考历史代码生成清洗与特征算子，最优方案通过后注册到能力库；
3. 客户流失（换一种说法的相似需求）：直接复用上一步注册的算子。

默认使用录制好的 LLM 响应（--llm replay），完全离线、结果可复现。
"""

from __future__ import annotations

import shutil
from pathlib import Path

from factory.kg import DEFAULT_KG_PATH, CapabilityKG
from factory.kg_viz import render_html
from factory.llm import LLM
from factory.schemas import RunState
from factory.workflow import Workflow, WorkflowConfig, best_report

STEPS = [
    ("01_tickets_transfer", "把客服工单自动分到网络故障、账单争议、套餐变更、销户投诉四类，宏平均 F1 不低于 0.8",
     "data/tickets.csv", 2),
    ("02_churn_synthesize", "预测30天内可能流失的客户，需可解释，召回优先", "data/churn.csv", 3),
    ("03_churn_reuse", "预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户", "data/churn.csv", 2),
]


def run_demo(llm: LLM, out: Path, base_kg: Path = DEFAULT_KG_PATH) -> list[RunState]:
    out = Path(out)
    for sub in ("kg", "runs"):  # 只清理本函数生成的子目录
        shutil.rmtree(out / sub, ignore_errors=True)
    (out / "kg").mkdir(parents=True)
    kg_path = out / "kg" / "capability_kg.json"
    shutil.copy(base_kg, kg_path)

    states = []
    for run_id, request, data, n in STEPS:
        wf = Workflow(llm, kg_path=kg_path, runs_dir=out / "runs", config=WorkflowConfig(n_candidates=n))
        states.append(wf.run(request, data, run_id=run_id))
    kg = CapabilityKG.load(kg_path)
    render_html(kg, out / "kg" / "capability_kg.html")
    (out / "SUMMARY.md").write_text(summary(states, kg, CapabilityKG.load(base_kg)), encoding="utf-8")
    return states


def summary(states: list[RunState], kg: CapabilityKG, base: CapabilityKG) -> str:
    rows = []
    for s in states:
        r = best_report(s)
        metric = s.spec.primary_metric if s.spec else None
        value = f"{r.metrics.get(metric.name, float('nan')):.3f}" if r and metric else "-"
        reused = sorted({st.operator_ref for p in s.candidates for st in p.steps
                         if st.action == "reuse" and st.operator_ref and kg.g.nodes.get(f"Operator:{st.operator_ref}", {}).get("origin") == "synthesized"})
        rows.append(f"| [{s.run_id}](runs/{s.run_id}/report.md) | {s.status} | {s.best_plan or '-'} | "
                    f"{metric.name if metric else '-'} = {value} | {len(s.generated)} | {s.repair_rounds} | "
                    f"{'、'.join(s.registered) or '-'} | {'、'.join(reused) or '-'} | {len(s.lessons_learned)} |")
    before, after = base.stats()["nodes"], kg.stats()["nodes"]
    growth = "；".join(f"{t} {before.get(t, 0)} → {n}" for t, n in after.items() if n != before.get(t, 0))
    return "\n".join([
        "# 演示结果摘要",
        "",
        "由 `python -m factory.cli demo` 生成。三个任务依次在同一份知识图谱副本上运行。",
        "",
        "| 运行 | 状态 | 最优方案 | 主指标 | 生成算子 | 修复轮数 | 注册新算子 | 复用已沉淀算子 | 新增经验 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
        f"知识图谱的变化：{growth or '无'}。",
        "",
        "- 知识图谱：`kg/capability_kg.json`（可视化 `kg/capability_kg.html`）",
        "- 沉淀的生成算子源码：`kg/operators/`",
        "",
    ])
