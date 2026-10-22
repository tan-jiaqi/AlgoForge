"""Curator Agent：把验证结果沉淀回能力知识图谱。

里程碑 3 中负责：把每个候选方案的验证结果写成 ValidationRun（连到所用算子与场景），
并累计被使用过的 Lesson 的命中次数。注册新算子与提炼新 Lesson 在里程碑 4 加入。
"""

from __future__ import annotations

from pathlib import Path

from factory.kg import CapabilityKG, node_id
from factory.operators import registry
from factory.schemas import RunState


class Curator:
    def __init__(self, kg: CapabilityKG, kg_path: Path | None):
        self.kg = kg
        self.kg_path = kg_path

    def ensure_scenario(self, state: RunState, scenario: str | None) -> str:
        """已知场景直接复用；新数据集则按 TaskSpec 新建场景节点。"""
        if scenario:
            return node_id("Scenario", scenario)
        spec = state.spec
        name = Path(state.data_path).stem
        sid = self.kg.add_scenario(
            name, title=spec.description[:40], task_type=spec.task_type.value, modality=spec.modality.value,
            dataset=state.data_path, target_column=spec.target_column, constraints=spec.constraints,
        )
        for m in spec.metrics:
            self.kg.add_metric(sid, m.name, m.threshold, m.primary)
        return sid

    def record(self, state: RunState, scenario: str | None) -> list[str]:
        sid = self.ensure_scenario(state, scenario)
        plans = {p.name: p for p in state.candidates}
        written = []
        for report in state.reports:
            plan = plans.get(report.plan_name)
            if plan is None:
                continue
            ops = []
            for step in plan.steps:
                ref = step.operator_ref or (step.synth_spec.name if step.synth_spec else None)
                nid = node_id("Operator", registry.get(ref).meta.key) if ref in registry else None
                if nid and nid not in self.kg.g:
                    self.kg.add_operator(registry.get(ref).meta, executable=True)
                if nid:
                    ops.append(nid)
            codes = sorted({d.code.value for d in report.diagnostics})
            rid = self.kg.add_validation_run(
                scenario=sid,
                operators=ops,
                metrics=report.metrics,
                passed=report.passed,
                origin="factory",
                note=f"run {state.run_id} / {report.plan_name}" + (f" / {','.join(codes)}" if codes else ""),
                run_id=f"{state.run_id}-{report.plan_name}",
            )
            written.append(rid)
        for lid in state.lessons_used:
            if lid in self.kg.g:
                self.kg.g.nodes[lid]["hits"] = self.kg.g.nodes[lid].get("hits", 0) + 1
        if self.kg_path is not None:
            self.kg.save(self.kg_path)
        return written
