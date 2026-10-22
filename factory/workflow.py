"""工作流：一个显式的状态机，在各 Agent 之间传递 RunState。

    PLAN → BUILD → VERIFY → ROUTE ─┬─ 有方案通过        → CURATE → DONE
                                   ├─ 指标不达标且可重试 → REPLAN → BUILD ...
                                   └─ 其他              → CURATE → FAILED（失败结果同样回写 KG）

里程碑 3 只做 Reuse；代码类错误的修复（REPAIR）与新算子生成（SYNTHESIZE）在里程碑 4 加入。
不使用 LangGraph 等框架：状态转移一目了然，依赖少，便于单元测试。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from factory.agents.curator import Curator
from factory.agents.planner import Planner, PlanningContext, resolve_scenario
from factory.codegen import render_pipeline
from factory.kg import DEFAULT_KG_PATH, CapabilityKG, short_hash
from factory.llm import LLM
from factory.profile import DataProfile, profile
from factory.schemas import RunState, ValidationReport
from factory.utils import ROOT, get_logger
from factory.verify import Verifier, VerifyConfig

log = get_logger(__name__)

DEFAULT_RUNS_DIR = ROOT / "runs"


@dataclass
class WorkflowConfig:
    n_candidates: int = 2
    max_replans: int = 1
    writeback: bool = True
    verify: VerifyConfig | None = None  # None 表示按任务类型读取 configs/tasks/*.yaml


@dataclass
class _Ctx:
    """不需要持久化的运行期对象。"""

    profile: DataProfile | None = None
    scenario: str | None = None
    planning: PlanningContext | None = None
    feedback: list[str] = field(default_factory=list)


class Workflow:
    def __init__(
        self,
        llm: LLM,
        *,
        kg_path: Path = DEFAULT_KG_PATH,
        runs_dir: Path = DEFAULT_RUNS_DIR,
        config: WorkflowConfig | None = None,
    ):
        self.llm = llm
        self.kg_path = Path(kg_path)
        self.kg = CapabilityKG.load(self.kg_path)
        self.runs_dir = Path(runs_dir)
        self.cfg = config or WorkflowConfig()
        self.planner = Planner(llm, self.kg)
        self.verifier = Verifier(self.cfg.verify)
        self.curator = Curator(self.kg, self.kg_path if self.cfg.writeback else None)

    # ------------------------------------------------------------------

    def run(self, request: str, data_path: str, *, target: str | None = None) -> RunState:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + short_hash(request + data_path)[:4]
        state = RunState(run_id=run_id, request=request, data_path=str(data_path), target=target)
        ctx = _Ctx()
        handlers = {
            "PLAN": self._plan,
            "BUILD": self._build,
            "VERIFY": self._verify,
            "ROUTE": self._route,
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
            state.note(f"[{step}] 完成，用时 {time.perf_counter() - t0:.1f}s → {nxt}")
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
        state.lessons_used = [les["id"] for les in ctx.planning.lessons]
        state.candidates = self.planner.propose(state.spec, ctx.profile, ctx.planning, n_candidates=self.cfg.n_candidates)
        state.note(f"[PLAN] 候选方案：{[p.name for p in state.candidates]}")
        return "BUILD"

    def _build(self, state: RunState, ctx: _Ctx) -> str:
        # 里程碑 3：所有步骤均为 reuse，Planner 已校验算子存在，这里无需生成代码
        return "VERIFY"

    def _verify(self, state: RunState, ctx: _Ctx) -> str:
        done = {r.plan_name for r in state.reports}
        for plan in state.candidates:
            if plan.name in done:
                continue
            report = self.verifier.verify(plan, state.spec, state.data_path, state.generated)
            state.reports.append(report)
            primary = state.spec.primary_metric.name
            state.note(f"[VERIFY] {plan.name}：{'通过' if report.passed else '未通过'}，"
                       f"{primary}={report.metrics.get(primary, float('nan')):.4f}"
                       + "".join(f"；{d.code.value} {d.message}" for d in report.diagnostics))
        return "ROUTE"

    def _route(self, state: RunState, ctx: _Ctx) -> str:
        passed = [r for r in state.reports if r.passed]
        if passed:
            state.best_plan = max(passed, key=lambda r: r.primary_value(state.spec)).plan_name
            return "CURATE"
        latest = [r for r in state.reports if r.plan_name in {p.name for p in state.candidates}]
        metric_only = all(d.code.category == "metric" for r in latest for d in r.diagnostics)
        if metric_only and state.replans < self.cfg.max_replans:
            return "REPLAN"
        best = max(state.reports, key=lambda r: r.primary_value(state.spec), default=None)
        state.best_plan = best.plan_name if best else None
        state.note("[ROUTE] 没有方案通过验证" + ("" if metric_only else "（存在代码类错误，里程碑 4 将支持修复）"))
        return "CURATE"

    def _replan(self, state: RunState, ctx: _Ctx) -> str:
        state.replans += 1
        for r in state.reports:
            if r.plan_name in {p.name for p in state.candidates} and not r.passed:
                ctx.feedback.append(f"{r.plan_name}: " + "；".join(f"{d.code.value} {d.message}" for d in r.diagnostics))
        tried = {p.name for p in state.candidates}
        new = self.planner.propose(state.spec, ctx.profile, ctx.planning, n_candidates=self.cfg.n_candidates, feedback=ctx.feedback)
        for p in new:
            if p.name in tried:
                p.name = f"{p.name}_r{state.replans}"
        state.candidates += new
        state.note(f"[REPLAN] 第 {state.replans} 次重新规划：{[p.name for p in new]}")
        return "BUILD"

    def _curate(self, state: RunState, ctx: _Ctx) -> str:
        state.kg_writes = self.curator.record(state, ctx.scenario)
        state.note(f"[CURATE] 写入 KG：{len(state.kg_writes)} 条 ValidationRun")
        return "DONE" if any(r.passed for r in state.reports) else "FAILED"

    # ------------------------------------------------------------------

    def _save(self, state: RunState) -> None:
        out = self.runs_dir / state.run_id
        out.mkdir(parents=True, exist_ok=True)
        plan = next((p for p in state.candidates if p.name == state.best_plan), None)
        if plan is not None and state.spec is not None:
            code = render_pipeline(plan, state.spec, state.generated)
            if state.status != "succeeded":
                code = "# 警告：该方案未通过验证，仅为得分最高的候选，供人工参考。\n" + code
            (out / "pipeline.py").write_text(code, encoding="utf-8")
            state.artifacts["pipeline"] = str(out / "pipeline.py")
        state.artifacts["state"] = str(out / "state.json")
        (out / "state.json").write_text(state.model_dump_json(indent=2), encoding="utf-8")


def best_report(state: RunState) -> ValidationReport | None:
    return next((r for r in state.reports if r.plan_name == state.best_plan), None)
