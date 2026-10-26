"""方案搜索：重新规划阶段的小规模 Beam Search。

每轮（一次 REPLAN）：

1. **选父方案**：已验证的方案按 ``rank_reports`` 排序，取前 ``width`` 个作为 beam；
2. **扩展**：把父方案的结构、指标与诊断交给 Planner，要求在其基础上做局部修改，
   一次给出 ``width × expand`` 个子方案；
3. **去重**：结构与参数和已验证过的方案完全相同的子方案直接丢弃（图搜索中的 visited 集合）；
4. **先验打分**：不跑验证，用知识图谱的历史验证记录 + 本次运行已有的观测，估计每个子方案的主指标；
5. **剪枝**：只保留先验最高的 ``width`` 个进入验证（验证是多种子 × K 折交叉验证，开销最大）。

先验的估计方法（确定性，可解释）：

- 每一步的估计值 = 该算子相关观测值的平滑均值 ``(Σv + C·base) / (n + C)``。观测值来自
  KG 中该场景的验证记录（``metric_evidence``）与本次运行中用到该算子的方案的主指标；
  ``base`` 取主指标阈值（没有证据的算子视为"刚好达标"），``C`` 为平滑强度；
- 生成步骤没有自己的记录时，沿用其参考的历史实现（``synth_spec.reference``）的证据；
- 方案先验 = 各步骤估计值的加权平均，模型步骤权重更高（对指标影响最大）。
"""

from __future__ import annotations

from dataclasses import dataclass

from factory.kg import CapabilityKG, node_id
from factory.schemas import OperatorKind, PipelinePlan, RunState, SearchCandidate, Step, TaskSpec, ValidationReport

PRIOR_SMOOTHING = 1.0  # 平滑强度 C：相当于给每个算子 1 条"刚好达标"的虚拟观测
MODEL_WEIGHT = 3.0  # 模型步骤在方案先验中的权重（其余步骤为 1）


@dataclass
class BeamConfig:
    width: int = 2  # 每轮保留并验证的方案数
    expand: int = 2  # 扩展倍数：每轮让 Planner 给出 width × expand 个子方案


def step_key(step: Step) -> str:
    return step.operator_ref if step.action == "reuse" else f"*{step.synth_spec.name}"


def describe(plan: PipelinePlan) -> str:
    return " → ".join((s.operator_ref or "").split("@")[0] if s.action == "reuse" else f"生成:{s.synth_spec.name}"
                      for s in plan.steps)


def signature(plan: PipelinePlan) -> tuple:
    """方案的结构签名：步骤顺序、算子与参数都相同即视为同一方案（名字与理由不计）。"""
    return tuple((s.kind.value, step_key(s), tuple(sorted((k, repr(v)) for k, v in s.params.items())))
                 for s in plan.steps)


def beam_parents(reports: list[ValidationReport], spec: TaskSpec, width: int) -> list[ValidationReport]:
    from factory.workflow import rank_reports

    return rank_reports(reports, spec)[:width]


def parent_summary(plan: PipelinePlan, report: ValidationReport, spec: TaskSpec) -> dict:
    """父方案的摘要，放进 Planner 的 prompt。"""
    return {
        "name": plan.name,
        "steps": [{"kind": s.kind.value, "action": s.action, "operator": step_key(s).lstrip("*"), "params": s.params,
                   **({"synth_spec": s.synth_spec.model_dump(mode="json")} if s.synth_spec else {})}
                  for s in plan.steps],
        spec.primary_metric.name: report.metrics.get(spec.primary_metric.name),
        "passed": report.passed,
        "diagnostics": [f"{d.code.value} {d.message}" for d in report.diagnostics],
    }


def run_observations(state: RunState) -> dict[str, list[float]]:
    """本次运行中每个算子（按 step_key）参与过的方案的主指标。"""
    spec = state.spec
    plans = {p.name: p for p in state.candidates}
    obs: dict[str, list[float]] = {}
    for r in state.reports:
        value = r.metrics.get(spec.primary_metric.name)
        plan = plans.get(r.plan_name)
        if value is None or plan is None:
            continue
        for s in plan.steps:
            obs.setdefault(step_key(s), []).append(float(value))
    return obs


def plan_prior(
    plan: PipelinePlan,
    spec: TaskSpec,
    kg: CapabilityKG,
    scenario: str | None,
    observed: dict[str, list[float]],
) -> tuple[float, dict[str, str]]:
    """估计方案的主指标，返回 (先验, 每步的估计值与证据)。"""
    metric, base = spec.primary_metric.name, spec.primary_metric.threshold
    total = weights = 0.0
    evidence: dict[str, str] = {}
    for s in plan.steps:
        key = step_key(s)
        kg_refs = [s.operator_ref] if s.action == "reuse" else [r for r in (s.synth_spec.reference,) if r]
        kg_vals = [v for ref in kg_refs for v in kg.metric_evidence(node_id("Operator", ref), scenario, metric)] \
            if scenario else []
        run_vals = observed.get(key, [])
        vals = kg_vals + run_vals
        value = (sum(vals) + PRIOR_SMOOTHING * base) / (len(vals) + PRIOR_SMOOTHING)
        w = MODEL_WEIGHT if s.kind == OperatorKind.MODEL else 1.0
        total += w * value
        weights += w
        source = []
        if kg_vals:
            source.append(f"KG {len(kg_vals)} 条")
        if run_vals:
            source.append(f"本次 {len(run_vals)} 条")
        evidence[key.lstrip("*")] = f"{value:.3f}（{'、'.join(source) or '无记录，按阈值'}）"
    return (round(total / weights, 4) if weights else base), evidence


def select(
    children: list[PipelinePlan],
    *,
    seen: set[tuple],
    width: int,
    spec: TaskSpec,
    kg: CapabilityKG,
    scenario: str | None,
    observed: dict[str, list[float]],
) -> tuple[list[PipelinePlan], list[SearchCandidate]]:
    """去重 → 先验打分 → 保留前 width 个。返回 (入选方案, 本轮全部子方案的记录)。"""
    records: list[SearchCandidate] = []
    scored: list[tuple[float, int, PipelinePlan, SearchCandidate]] = []
    local = set(seen)
    for i, plan in enumerate(children):
        sig = signature(plan)
        if sig in local:
            records.append(SearchCandidate(plan=plan.name, chain=describe(plan), reason="与已有方案重复，跳过"))
            continue
        local.add(sig)
        prior, ev = plan_prior(plan, spec, kg, scenario, observed)
        rec = SearchCandidate(plan=plan.name, chain=describe(plan), prior=prior, evidence=ev)
        for step in plan.steps:
            ref = step.operator_ref if step.action == "reuse" else step.synth_spec.reference
            records_for_step = kg.metric_records(node_id("Operator", ref), scenario, spec.primary_metric.name) if ref and scenario else []
            rec.observations[step_key(step)] = [
                *[{**r, "source": "kg"} for r in records_for_step],
                *[{"source": "current_run", "value": v} for v in observed.get(step_key(step), [])],
            ]
        records.append(rec)
        scored.append((prior, -i, plan, rec))  # 先验相同时保持 Planner 给出的顺序
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    kept = []
    for rank, (_, _, plan, rec) in enumerate(scored):
        if rank < width:
            rec.selected, rec.reason = True, f"先验第 {rank + 1}，进入验证"
            kept.append(plan)
        else:
            rec.reason = f"先验第 {rank + 1}，超出 beam 宽度 {width}，剪枝"
    return kept, records
