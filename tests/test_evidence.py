"""KG 引用范围、消融隔离、严格回放和调用来源。"""

import json

import pytest

from factory.agents.planner import Planner
from factory.agents.rules import rule_based_mock
from factory.evaluation import inject_nan, task
from factory.kg import DEFAULT_KG_PATH, CapabilityKG
from factory.llm import CassetteLLM, LLMError, MockLLM
from factory.operators import registry
from factory.profile import profile
from factory.schemas import PipelinePlan, RunState, Step, SynthSpec


def context(churn_csv):
    planner = Planner(rule_based_mock(), CapabilityKG.load(DEFAULT_KG_PATH))
    spec = task("预测流失，召回率不低于0.75")
    prof = profile(str(churn_csv), "churn")
    ctx = planner.retrieve(spec, prof, scenario="telecom_churn", top_k=100)
    return planner, spec, prof, ctx


def model_plan(ref="LogisticRegressionModel@1", evidence=None):
    return PipelinePlan(name="p", steps=[Step(kind="model", action="reuse", operator_ref=ref,
                                             evidence_ids=evidence if evidence is not None else [f"Operator:{ref}"])])


def test_rejects_registry_operator_outside_retrieval(churn_csv):
    planner, _, _, ctx = context(churn_csv)
    ctx.options["model"] = [o for o in ctx.options["model"] if o["ref"] != "RandomForestModel@1"]
    assert "RandomForestModel@1" in registry
    assert "不在本轮检索候选" in " ".join(planner.context_errors(model_plan("RandomForestModel@1"), ctx))


def test_rejects_missing_forged_and_cross_step_evidence(churn_csv):
    planner, _, _, ctx = context(churn_csv)
    assert "缺少" in " ".join(planner.context_errors(model_plan(evidence=[]), ctx))
    for eid, expected in [("Lesson:invented", "不在本轮上下文"), ("Operator:MissingValueImputer@1", "不适用")]:
        assert expected in " ".join(planner.context_errors(model_plan(evidence=["Operator:LogisticRegressionModel@1", eid]), ctx))
    ctx.evidence["Lesson:wrong"] = {"type": "lesson", "applies_to": ["Operator:RandomForestModel@1"],
                                     "matching_conditions": [], "scenario_match": False}
    assert "适用对象不匹配" in " ".join(planner.context_errors(
        model_plan(evidence=["Operator:LogisticRegressionModel@1", "Lesson:wrong"]), ctx))


def test_rejects_history_reference_outside_retrieval(churn_csv):
    planner, _, _, ctx = context(churn_csv)
    ctx.options["imputer"] = [o for o in ctx.options["imputer"] if o["ref"] != "TotalChargesCleaner@1"]
    plan = model_plan()
    plan.steps.insert(0, Step(kind="imputer", action="synthesize", synth_spec=SynthSpec(
        name="Cleaner", kind="imputer", purpose="p", inputs="i", outputs="o", reference="TotalChargesCleaner@1"),
        evidence_ids=["Operator:TotalChargesCleaner@1"]))
    assert "不在本轮检索候选" in " ".join(planner.context_errors(plan, ctx))


def test_context_violation_retries_and_exhausts(churn_csv):
    planner, spec, prof, ctx = context(churn_csv)
    bad = json.dumps({"plans": [model_plan(evidence=["Lesson:fake"]).model_dump(mode="json")]})
    planner.llm.queue("plan", bad)
    assert planner.propose(spec, prof, ctx, n_candidates=1)
    assert "不在本轮上下文" in planner.llm.calls[-1].messages[-1]["content"]
    planner.llm.queue("plan", bad, bad, bad)
    with pytest.raises(LLMError, match="could not obtain valid"):
        planner.propose(spec, prof, ctx, n_candidates=1)


def test_catalog_and_no_lessons_do_not_leak_removed_knowledge(churn_csv, monkeypatch):
    planner, spec, prof, full = context(churn_csv)
    planner.knowledge_mode = "no_lessons"
    no_lessons = planner.retrieve(spec, prof, scenario="telecom_churn", top_k=100)
    assert no_lessons.options == full.options
    assert not no_lessons.lessons
    assert not any(e["type"] == "lesson" for e in no_lessons.evidence.values())
    planner.knowledge_mode = "catalog"

    def forbidden(*args, **kwargs):
        pytest.fail("catalog must not retrieve KG knowledge")

    monkeypatch.setattr(planner.kg, "search_operators", forbidden)
    monkeypatch.setattr(planner.kg, "search_lessons", forbidden)
    catalog = planner.retrieve(spec, prof, scenario="telecom_churn", top_k=100)
    assert not catalog.lessons and not catalog.scenario_metrics
    assert {o["ref"] for opts in catalog.options.values() for o in opts} == {
        o["ref"] for opts in full.options.values() for o in opts if o["executable"]}
    assert all(not o.get("history_records") for opts in catalog.options.values() for o in opts)
    assert all(e["type"] == "operator" for e in catalog.evidence.values())


def test_old_state_and_call_provenance(tmp_path):
    state = RunState.model_validate({"run_id": "old", "request": "r", "data_path": "x"})
    assert not state.evidence_checks and not state.llm_calls
    llm = CassetteLLM(tmp_path, inner=MockLLM(default=lambda m: "response"), model="m")
    assert llm.chat([], tag="x") == "response"
    assert llm.calls[0]["source"] == "provided", "mock output must never be labelled remote"
    replay = CassetteLLM(tmp_path, None, "m")
    replay.chat([], tag="x")
    assert replay.calls[0]["source"] == "replay"
    with pytest.raises(LLMError, match="no recorded response"):
        replay.chat([{"role": "user", "content": "new"}])
    assert replay.calls[-1]["source"] == "missing_recording"


def test_new_agent_namespace_does_not_silently_reuse_legacy_responses(tmp_path):
    legacy = CassetteLLM(tmp_path, MockLLM(default=lambda m: "old"), "m")
    legacy.chat([], tag="plan")
    current = CassetteLLM(tmp_path, None, "m", agent_namespace=True)
    with pytest.raises(LLMError, match="no recorded response"):
        current.chat([], tag="plan")
    # 抽取协议未改变，仍可读取原来的抽取录制。
    legacy.chat([], tag="extract_code")
    assert current.chat([], tag="extract_code") == "old"


def test_injection_creates_new_nan_without_changing_original():
    import pandas as pd
    source = '''import pandas as pd
class Cleaner:
    def transform(self, X):
        X = X.copy()
        X["total_charges"] = pd.to_numeric(X["total_charges"], errors="coerce").fillna(0)
        return X
'''
    ns = {}
    exec(inject_nan(source), ns)
    frame = pd.DataFrame({"total_charges": [" ", "12"]})
    result = ns["Cleaner"]().transform(frame)
    assert result["total_charges"].isna().sum() == 1
    assert frame.iloc[0, 0] == " "


def test_other_code_failure_does_not_prove_original_issue_fixed():
    from factory.schemas import Diagnostic, DiagnosticCode, GeneratedOperator, RepairRecord, ValidationReport
    from factory.workflow import Workflow
    spec = SynthSpec(name="Cleaner", kind="imputer", purpose="p", inputs="i", outputs="o")
    plan = model_plan()
    plan.steps.insert(0, Step(kind="imputer", action="synthesize", synth_spec=spec))
    diag = Diagnostic(code=DiagnosticCode.NAN_OUTPUT, stage="contract", message="NaN", operator="Cleaner@1")
    repair = RepairRecord(operator="Cleaner@1", revision=1, diagnostic=diag)
    state = RunState(run_id="r", request="r", data_path="d", candidates=[plan], repairs=[repair],
                     generated={"Cleaner": GeneratedOperator(spec=spec, source="invalid", revision=1)},
                     reports=[ValidationReport(plan_name="p", passed=False, diagnostics=[Diagnostic(
                         code=DiagnosticCode.SYNTAX_ERROR, stage="static", message="syntax", operator="Cleaner@1")])])
    wf = Workflow.__new__(Workflow)
    wf._mark_repairs(state)
    assert state.repairs[0].fixed is None
    state.reports = [ValidationReport(plan_name="p", passed=False, diagnostics=[Diagnostic(
        code=DiagnosticCode.METRIC_BELOW_THRESHOLD, stage="metric", message="threshold")])]
    wf._mark_repairs(state)
    assert state.repairs[0].fixed is True


def test_evidence_checks_record_actual_results(churn_csv):
    """范围检查不能写死为通过：被拒绝的规划要留下记录，接受方案逐步复核。"""
    from factory.workflow import Workflow, _Ctx

    planner, spec, prof, ctx = context(churn_csv)
    bad = json.dumps({"plans": [model_plan(evidence=["Lesson:fake"]).model_dump(mode="json")]})
    planner.llm.queue("plan", bad)
    plans = planner.propose(spec, prof, ctx, n_candidates=1)
    wf = Workflow.__new__(Workflow)
    wf.planner = planner
    state = RunState(run_id="r", request="r", data_path="d")
    wctx = _Ctx()
    wctx.planning = ctx
    wf._check_evidence(state, wctx, plans)
    rejected, *accepted = state.evidence_checks
    assert not rejected["scope_valid"] and "不在本轮上下文" in "".join(rejected["errors"])
    assert accepted and all(e["scope_valid"] and not e["errors"] for e in accepted)
    assert not planner.rejections, "记录后清空，避免重复计入下一轮"
    forged = model_plan(evidence=["Operator:LogisticRegressionModel@1", "Operator:MissingValueImputer@1"])
    wf._check_evidence(state, wctx, [forged])
    assert not state.evidence_checks[-1]["scope_valid"]
