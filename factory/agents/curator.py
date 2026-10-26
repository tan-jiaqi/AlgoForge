"""Curator Agent：把一次运行的成果沉淀回能力库与知识图谱。

1. 注册新算子：最优方案中验证通过的生成算子保存为源码文件、注册到 OperatorRegistry，
   并在 KG 中写为可执行 Operator 节点（DERIVED_FROM 指向参考的历史实现）；
2. 回写验证结果：每个候选方案的最终验证结果写成 ValidationRun（失败的也写）；
3. 提炼经验：把成功的"诊断 → 修复"交给 LLM 归纳成 Lesson（带诊断码），下次修复同类错误时检索使用；
4. 累计被提供给 Agent 的 Lesson 的检索命中次数（不代表采纳）。
"""

from __future__ import annotations

import difflib
from pathlib import Path

from factory.agents.planner import _wrap
from factory.extract import LessonRecord, resolve_target
from factory.kg import CONDITION_TAGS, CapabilityKG, node_id
from factory.llm import LLM
from factory.operators import registry
from factory.operators.synthesized import exec_operator_source, save_operator
from factory.schemas import RepairRecord, RunState
from factory.utils import get_logger
from factory.verify.static import check_source

log = get_logger(__name__)


class Curator:
    def __init__(self, kg: CapabilityKG, kg_path: Path | None, llm: LLM | None = None):
        self.kg = kg
        self.kg_path = kg_path
        self.llm = llm

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

    # ------------------------------------------------------------------

    def record(self, state: RunState, scenario: str | None) -> list[str]:
        sid = self.ensure_scenario(state, scenario)
        self._register_operators(state)
        written = self._write_runs(state, sid)
        self._distill_lessons(state, sid, written)
        for lid in state.lessons_used:
            if lid in self.kg.g:
                self.kg.g.nodes[lid]["hits"] = self.kg.g.nodes[lid].get("hits", 0) + 1
        if self.kg_path is not None:
            self.kg.save(self.kg_path)
        return written

    def _register_operators(self, state: RunState) -> None:
        """只注册最优方案（且已通过）中用到的生成算子。"""
        best = next((p for p in state.candidates if p.name == state.best_plan), None)
        report = next((r for r in state.reports if r.plan_name == state.best_plan), None)
        if best is None or report is None or not report.passed:
            return
        for step in best.steps:
            gen = state.generated.get(step.synth_spec.name) if step.action == "synthesize" else None
            if gen is None or gen.key in registry:
                continue
            if check_source(gen.source, class_name=gen.spec.name, kind=gen.spec.kind):
                log.warning("算子 %s 未通过静态复查，不注册", gen.key)
                continue
            if self.kg_path is not None:
                save_operator(gen.source, gen.key, self.kg_path)
            added = exec_operator_source(gen.source)
            for cls in added:
                # 版本谱系与"参考了哪段历史代码"分开：LLM 没有引用历史实现时 reference 为空，
                # 但同名旧版本仍是它的上一版，编号 @2 也是据此顺延的。
                previous = max((d["version"] for _, d in self.kg.nodes("Operator")
                                if d["name"] == cls.meta.name and d["version"] < cls.meta.version), default=None)
                parent = gen.spec.reference or cls.meta.parent or (f"{cls.meta.name}@{previous}" if previous else None)
                meta = cls.meta.model_copy(update={"origin": "synthesized", "parent": parent})
                self.kg.add_operator(meta, executable=True, code=gen.source)
                state.registered.append(meta.key)

    def _write_runs(self, state: RunState, sid: str) -> list[str]:
        plans = {p.name: p for p in state.candidates}
        written = []
        for report in state.reports:
            plan = plans.get(report.plan_name)
            if plan is None:
                continue
            ops = []
            for step in plan.steps:
                if step.action == "synthesize":
                    gen = state.generated.get(step.synth_spec.name)
                    ref = gen.key if gen else None
                else:
                    ref = step.operator_ref
                nid = node_id("Operator", ref) if ref else None
                if nid and nid not in self.kg.g and ref in registry:
                    self.kg.add_operator(registry.get(ref).meta, executable=True, code=registry.source_of(ref) or "")
                if nid and nid in self.kg.g:
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
        return written

    def _distill_lessons(self, state: RunState, sid: str, runs: list[str]) -> None:
        if self.llm is None:
            return
        scenario = self.kg.g.nodes[sid]["key"]
        for rec in state.repairs:
            if not rec.fixed:
                continue
            gen = state.generated.get(rec.operator.split("@")[0])
            try:
                les = self.distill(rec, purpose=gen.spec.purpose if gen else "")
            except Exception as exc:  # noqa: BLE001 - 经验提炼失败不影响主流程
                log.warning("提炼经验失败：%s", exc)
                continue
            targets = [t for t in (resolve_target(self.kg, n, scenario) for n in les.applies_to) if t]
            if gen and gen.spec.reference:
                targets.append(node_id("Operator", gen.spec.reference))
            # 经验来源：用到该算子的方案的验证记录
            name = rec.operator.split("@")[0]
            using = [f"ValidationRun:{state.run_id}-{p.name}" for p in state.candidates
                     if any(st.action == "synthesize" and st.synth_spec.name == name for st in p.steps)]
            source = next((r for r in using if r in runs), runs[0] if runs else None)
            lid = self.kg.add_lesson(
                title=les.title, trigger=les.trigger, fix=les.fix, error_code=rec.diagnostic.code,
                applies_to=targets, learned_from=source,
            )
            self.kg.g.nodes[lid]["origin"] = "factory"
            state.lessons_learned.append(lid)

    def distill(self, rec: RepairRecord, *, purpose: str) -> LessonRecord:
        diff = "\n".join(difflib.unified_diff(rec.before.splitlines(), rec.after.splitlines(), lineterm="", n=1))
        payload = {
            "operator_purpose": purpose,
            "diagnostic": rec.diagnostic.model_dump(mode="json"),
            "code_diff": diff[:4000],
            "condition_tags": list(CONDITION_TAGS),
        }
        system = (
            "你是算法能力工厂的 Curator。下面是一次被验证器发现、随后修复成功的代码问题。"
            "把它提炼为一条可复用的经验（LessonRecord）：title 一句话概括，trigger 写现象/报错，"
            "fix 写通用的修复方法（不要只描述这一处改动），applies_to 填相关的条件标签（从 condition_tags 中选）。"
            "error_code 与诊断码一致。"
        )
        return self.llm.structured(
            [{"role": "system", "content": system}, {"role": "user", "content": _wrap(payload)}],
            LessonRecord,
            tag="lesson",
        )
