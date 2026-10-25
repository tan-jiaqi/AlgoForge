"""工作流：一个显式的状态机，在各 Agent 之间传递 RunState。

    PLAN → BUILD → VERIFY → ROUTE ─┬─ 生成算子有代码类错误且未超轮数 → REPAIR → VERIFY ...
                                   ├─ 有方案通过                    → CURATE → DONE
                                   ├─ 指标不达标且可重试            → REPLAN（Beam 搜索一轮）→ BUILD ...
                                   └─ 其他                          → CURATE → FAILED（失败结果同样回写 KG）

- BUILD：为 synthesize 步骤调用 Synthesizer 生成算子（同名算子在一次运行内只生成一次）；
- REPAIR：诊断能定位到生成算子时，只修复该算子（带上同诊断码的 Lesson），然后只重验受影响的方案；
- REPLAN：以排名前 k 的已验证方案为父方案，让 Planner 扩展 k × m 个子方案，去重后按知识图谱
  历史验证分估计的先验保留 k 个再验证（小规模 Beam Search，见 search.py）；
- 先修复、后选优：所有候选都得到修复机会后再比较，保证公平。

不使用 LangGraph 等框架：状态转移一目了然，依赖少，便于单元测试。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable
from datetime import datetime
from pathlib import Path

from factory import search
from factory.agents.curator import Curator
from factory.agents.planner import Planner, PlanningContext, resolve_scenario
from factory.agents.synthesizer import Synthesizer
from factory.codegen import render_pipeline
from factory.kg import DEFAULT_KG_PATH, CapabilityKG, short_hash
from factory.llm import LLM
from factory.operators import registry
from factory.operators.synthesized import load_saved_operators
from factory.profile import DataProfile, profile
from factory.report import write_report
from factory.schemas import (
    Diagnostic, GeneratedOperator, PipelinePlan, RepairRecord, RunState, SearchRound, TaskSpec, ValidationReport,
)
from factory.utils import ROOT, get_logger
from factory.verify import Verifier, VerifyConfig, load_verify_config

log = get_logger(__name__)

DEFAULT_RUNS_DIR = ROOT / "runs"


@dataclass
class WorkflowConfig:
    n_candidates: int = 2
    max_replans: int = 1
    max_repairs: int = 3  # 每个生成算子最多修复的轮数
    beam_width: int | None = None  # 重新规划时每轮保留并验证的方案数；None 表示与 n_candidates 相同
    beam_expand: int = 2  # 扩展倍数：每轮让 Planner 给出 beam_width × beam_expand 个子方案，按先验剪枝
    allow_synthesis: bool = True
    writeback: bool = True
    verify: VerifyConfig | None = None  # None 表示按任务类型读取 configs/tasks/*.yaml


@dataclass
class _Ctx:
    """不需要持久化的运行期对象。"""

    profile: DataProfile | None = None
    scenario: str | None = None
    planning: PlanningContext | None = None
    feedback: list[str] = field(default_factory=list)
    repair_queue: list[tuple[str, Diagnostic]] = field(default_factory=list)
    retired: set[str] = field(default_factory=set)  # 重新规划后被替换的旧方案，不再验证和修复


class Workflow:
    def __init__(
        self,
        llm: LLM,
        *,
        kg_path: Path = DEFAULT_KG_PATH,
        runs_dir: Path = DEFAULT_RUNS_DIR,
        config: WorkflowConfig | None = None,
        on_event: Callable[[str], None] | None = None,
    ):
        self.llm = llm
        self.on_event = on_event  # 每条运行记录产生时回调，用于界面实时展示进度
        self.kg_path = Path(kg_path)
        self.kg = CapabilityKG.load(self.kg_path)
        loaded = load_saved_operators(self.kg_path)
        if loaded:
            log.info("加载已沉淀的生成算子：%s", loaded)
        self.runs_dir = Path(runs_dir)
        self.cfg = config or WorkflowConfig()
        self.planner = Planner(llm, self.kg, allow_synthesis=self.cfg.allow_synthesis)
        self.synthesizer = Synthesizer(llm, self.kg)
        self.verifier = Verifier(self.cfg.verify)
        self.curator = Curator(self.kg, self.kg_path if self.cfg.writeback else None, llm)

    # ------------------------------------------------------------------

    def run(self, request: str, data_path: str, *, target: str | None = None, run_id: str | None = None) -> RunState:
        run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S-") + short_hash(request + data_path)[:4]
        state = RunState(run_id=run_id, request=request, data_path=str(data_path), target=target,
                         llm_model=getattr(self.llm, "model", None))
        if self.on_event is not None:
            state._listener = self.on_event
        ctx = _Ctx()
        handlers = {
            "PLAN": self._plan,
            "BUILD": self._build,
            "VERIFY": self._verify,
            "ROUTE": self._route,
            "REPAIR": self._repair,
            "REPLAN": self._replan,
            "CURATE": self._curate,
        }
        step = "PLAN"
        while step in handlers:
            t0 = time.perf_counter()
            try:
                nxt = handlers[step](state, ctx)
            except Exception as exc:  # noqa: BLE001 - 任一阶段异常都记录并结束，不让工作流崩溃
                log.exception("阶段 %s 出错", step)
                state.note(f"[{step}] 出错：{type(exc).__name__}: {exc}")
                state.status = "failed"
                break
            elapsed = time.perf_counter() - t0
            state.timings[step] = round(state.timings.get(step, 0.0) + elapsed, 2)
            state.note(f"[{step}] 完成，用时 {elapsed:.1f}s → {nxt}")
            step = nxt
        if step == "DONE":
            state.status = "succeeded"
        elif step == "FAILED":
            state.status = "failed"
        self._save(state)
        return state

    # ------------------------------------------------------------------
    # 各阶段
    # ------------------------------------------------------------------

    def _plan(self, state: RunState, ctx: _Ctx) -> str:
        ctx.scenario = resolve_scenario(self.kg, state.data_path)
        rough = profile(state.data_path, state.target)
        state.spec = self.planner.understand(state.request, rough, target=state.target, scenario=ctx.scenario)
        ctx.profile = profile(state.data_path, state.spec.target_column)
        state.profile_tags = ctx.profile.tags
        state.note(f"[PLAN] 场景 {ctx.scenario or '新场景'}；画像标签 {ctx.profile.tags}")
        state.note(f"[PLAN] TaskSpec：{state.spec.task_type.value}，目标 {state.spec.target_column}，"
                   f"主指标 {state.spec.primary_metric.name} >= {state.spec.primary_metric.threshold}")
        ctx.planning = self.planner.retrieve(state.spec, ctx.profile, scenario=ctx.scenario)
        state.retrieval = {**ctx.planning.to_prompt(), "lesson_ids": [les["id"] for les in ctx.planning.lessons]}
        state.verify_config = (self.cfg.verify or load_verify_config(state.spec.task_type)).model_dump()
        state.lessons_used = [les["id"] for les in ctx.planning.lessons]
        state.candidates = self.planner.propose(state.spec, ctx.profile, ctx.planning, n_candidates=self.cfg.n_candidates)
        state.note(f"[PLAN] 候选方案：{[_describe(p) for p in state.candidates]}")
        return "BUILD"

    def _build(self, state: RunState, ctx: _Ctx) -> str:
        """为尚未生成的 synthesize 步骤生成算子代码。"""
        for plan in state.candidates:
            for step in plan.steps:
                if step.action != "synthesize" or step.synth_spec.name in state.generated:
                    continue
                spec = step.synth_spec
                lessons = self.kg.search_lessons(f"{spec.purpose} {spec.inputs} {spec.outputs}", top_k=3)
                self._use_lessons(state, lessons)
                gen = self.synthesizer.synthesize(spec, state.spec, ctx.profile, version=self._next_version(spec.name), lessons=lessons)
                state.generated[spec.name] = gen
                ref = f"（参考 {spec.reference}）" if spec.reference else ""
                state.note(f"[BUILD] 生成算子 {gen.key}{ref}：{spec.purpose}")
        return "VERIFY"

    def _verify(self, state: RunState, ctx: _Ctx) -> str:
        done = {r.plan_name for r in state.reports}
        for plan in state.candidates:
            if plan.name in done or plan.name in ctx.retired:
                continue
            report = self.verifier.verify(plan, state.spec, state.data_path, self._generated_for(plan, state))
            state.reports.append(report)
            primary = state.spec.primary_metric.name
            state.note(f"[VERIFY] {plan.name}：{'通过' if report.passed else '未通过'}，"
                       f"{primary}={report.metrics.get(primary, float('nan')):.4f}"
                       + "".join(f"；{d.code.value} {d.message}" for d in report.diagnostics))
        self._mark_repairs(state)
        return "ROUTE"

    def _route(self, state: RunState, ctx: _Ctx) -> str:
        ctx.repair_queue = self._repairable(state, ctx)
        if ctx.repair_queue:
            return "REPAIR"
        if any(r.passed for r in state.reports):
            state.best_plan = rank_reports(state.reports, state.spec)[0].plan_name
            return "CURATE"
        metric_only = all(d.code.category == "metric" for r in state.reports for d in r.diagnostics)
        if state.replans < self.cfg.max_replans and (metric_only or self.cfg.allow_synthesis):
            return "REPLAN"
        ranked = rank_reports(state.reports, state.spec)
        state.best_plan = ranked[0].plan_name if ranked else None
        state.note("[ROUTE] 没有方案通过验证")
        return "CURATE"

    def _repair(self, state: RunState, ctx: _Ctx) -> str:
        for name, diag in ctx.repair_queue:
            gen = state.generated[name]
            lessons = self.kg.search_lessons(f"{diag.message} {diag.hint}", error_code=diag.code, top_k=3)
            self._use_lessons(state, lessons)
            fixed = self.synthesizer.repair(gen, diag, lessons=lessons)
            state.generated[name] = fixed
            state.repairs.append(RepairRecord(
                operator=gen.key, revision=fixed.revision, diagnostic=diag,
                lessons=[les["id"] for les in lessons], before=gen.source, after=fixed.source,
            ))
            state.repair_rounds += 1
            hit = f"，参考经验：{[les['title'] for les in lessons]}" if lessons else ""
            state.note(f"[REPAIR] 修复 {gen.key} 第 {fixed.revision} 轮：{diag.code.value} {diag.message[:80]}{hit}")
            # 用到该算子的方案需要重新验证
            stale = {p.name for p in state.candidates if _uses(p, name) and p.name not in ctx.retired}
            state.history += [r for r in state.reports if r.plan_name in stale]
            state.reports = [r for r in state.reports if r.plan_name not in stale]
        return "VERIFY"

    def _replan(self, state: RunState, ctx: _Ctx) -> str:
        """Beam 搜索的一轮：以排名前 width 的已验证方案为父方案扩展，按 KG 先验剪枝后再验证（见 search.py）。"""
        state.replans += 1
        names = {p.name for p in state.candidates}
        for r in state.reports:
            if r.plan_name in names and not r.passed:
                ctx.feedback.append(f"{r.plan_name}: " + "；".join(f"{d.code.value} {d.message}" for d in r.diagnostics))
        for name, gen in state.generated.items():
            if gen.revision >= self.cfg.max_repairs:
                ctx.feedback.append(f"生成算子 {name} 修复 {gen.revision} 轮仍未通过，请换一种实现思路或不用它")
        ctx.retired |= names

        width = self.cfg.beam_width or self.cfg.n_candidates
        plans = {p.name: p for p in state.candidates}
        parents = [r for r in search.beam_parents(state.reports, state.spec, width) if r.plan_name in plans]
        children = self.planner.propose(
            state.spec, ctx.profile, ctx.planning, n_candidates=width * max(self.cfg.beam_expand, 1),
            feedback=ctx.feedback, parents=[search.parent_summary(plans[r.plan_name], r, state.spec) for r in parents],
        )
        for p in children:
            if p.name in names:
                p.name = f"{p.name}_r{state.replans}"
        kept, records = search.select(
            children, seen={search.signature(p) for p in state.candidates}, width=width, spec=state.spec,
            kg=self.kg, scenario=ctx.scenario, observed=search.run_observations(state),
        )
        state.search.append(SearchRound(round=state.replans, parents=[r.plan_name for r in parents], width=width,
                                        candidates=records))
        for p in kept:
            for step in p.steps:  # 修复失败的生成算子不再沿用旧代码，重新生成
                gen = state.generated.get(step.synth_spec.name) if step.action == "synthesize" else None
                if gen and gen.revision >= self.cfg.max_repairs:
                    state.generated.pop(step.synth_spec.name)
        state.candidates += kept
        pruned = [f"{c.plan}({c.prior})" for c in records if not c.selected]
        state.note(f"[REPLAN] 第 {state.replans} 轮 Beam 搜索：父方案 {[r.plan_name for r in parents]}，"
                   f"扩展 {len(children)} 个，入选 {[f'{_describe(p)}' for p in kept]}"
                   + (f"，剪枝/去重 {pruned}" if pruned else ""))
        return "BUILD"

    def _curate(self, state: RunState, ctx: _Ctx) -> str:
        state.kg_writes = self.curator.record(state, ctx.scenario)
        state.note(f"[CURATE] 写入 KG：{len(state.kg_writes)} 条 ValidationRun；注册新算子 {state.registered}；"
                   f"新增经验 {len(state.lessons_learned)} 条")
        return "DONE" if any(r.passed for r in state.reports) else "FAILED"

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _next_version(self, name: str) -> int:
        """与能力库、KG 中的同名算子顺延版本号（抽取的历史实现记为 v1，生成的从 v2 开始）。"""
        versions = [d["version"] for _, d in self.kg.nodes("Operator") if d["name"] == name]
        if name in registry:
            versions.append(registry.latest_version(name))
        return max(versions, default=0) + 1

    def _generated_for(self, plan: PipelinePlan, state: RunState) -> dict[str, GeneratedOperator]:
        return {s.synth_spec.name: state.generated[s.synth_spec.name] for s in plan.steps
                if s.action == "synthesize" and s.synth_spec.name in state.generated}

    def _repairable(self, state: RunState, ctx: _Ctx) -> list[tuple[str, Diagnostic]]:
        """找出可以修复的生成算子：诊断为代码类、能定位到该算子、且未超过修复轮数。"""
        queue: dict[str, Diagnostic] = {}
        for report in state.reports:
            if report.plan_name in ctx.retired:
                continue
            for d in report.diagnostics:
                if d.code.category != "code" or not d.operator:
                    continue
                name = d.operator.split("@")[0]
                gen = state.generated.get(name)
                if gen and gen.revision < self.cfg.max_repairs and name not in queue:
                    queue[name] = d
        return list(queue.items())

    def _mark_repairs(self, state: RunState) -> None:
        """修复后的版本重新验证完毕时，判断该次修复是否解决了原来的问题。"""
        for rec in state.repairs:
            if rec.fixed is not None:
                continue
            name = rec.operator.split("@")[0]
            related = [r for r in state.reports if any(_uses(p, name) for p in state.candidates if p.name == r.plan_name)]
            if not related:
                continue
            same = [d for r in related for d in r.diagnostics
                    if d.code == rec.diagnostic.code and (d.operator or "").split("@")[0] == name]
            rec.fixed = not same

    def _use_lessons(self, state: RunState, lessons: list[dict]) -> None:
        for les in lessons:
            if les["id"] not in state.lessons_used:
                state.lessons_used.append(les["id"])

    def _save(self, state: RunState) -> None:
        out = self.runs_dir / state.run_id
        out.mkdir(parents=True, exist_ok=True)
        plan = next((p for p in state.candidates if p.name == state.best_plan), None)
        if plan is not None and state.spec is not None:
            code = render_pipeline(plan, state.spec, self._generated_for(plan, state))
            if state.status != "succeeded":
                code = "# 警告：该方案未通过验证，仅为得分最高的候选，供人工参考。\n" + code
            (out / "pipeline.py").write_text(code, encoding="utf-8")
            state.artifacts["pipeline"] = _rel(out / "pipeline.py")
        for name, gen in state.generated.items():
            (out / f"{name}_v{gen.version}_r{gen.revision}.py").write_text(gen.source, encoding="utf-8")
        state.artifacts["state"] = _rel(out / "state.json")
        state.artifacts.update({k: _rel(v) for k, v in write_report(state, out, kg=self.kg).items()})
        (out / "state.json").write_text(state.model_dump_json(indent=2), encoding="utf-8")


def _rel(path: Path | str) -> str:
    """仓库内的产物用相对路径记录，避免把本机绝对路径写进报告与 state.json。"""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()  # 统一用 /，Windows 与 Linux 生成的 state.json 一致
    except ValueError:
        return path.as_posix()


def _uses(plan: PipelinePlan, name: str) -> bool:
    return any(s.action == "synthesize" and s.synth_spec.name == name for s in plan.steps)


def _describe(plan: PipelinePlan) -> str:
    parts = [(s.operator_ref or "").split("@")[0] if s.action == "reuse" else f"*{s.synth_spec.name}" for s in plan.steps]
    return f"{plan.name}({' → '.join(parts)})"


def rank_reports(reports: list[ValidationReport], spec: TaskSpec) -> list[ValidationReport]:
    """候选排序规则：先看是否通过，再比主指标，再比次要指标之和，最后训练耗时短者优先。"""
    secondary = [m.name for m in spec.metrics if not m.primary]

    def key(r: ValidationReport):
        return (r.passed, r.primary_value(spec), sum(r.metrics.get(m, 0.0) for m in secondary),
                -r.resources.get("fit_seconds", float("inf")))

    return sorted(reports, key=key, reverse=True)


def best_report(state: RunState) -> ValidationReport | None:
    return next((r for r in state.reports if r.plan_name == state.best_plan), None)
