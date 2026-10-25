"""方案搜索（重新规划阶段的 Beam Search）：去重、KG 先验、剪枝，以及在工作流中的接入。"""

import shutil

import pytest

from factory import search
from factory.agents.rules import rule_based_mock
from factory.kg import DEFAULT_KG_PATH, CapabilityKG
from factory.schemas import MetricTarget, Modality, OperatorKind, PipelinePlan, Step, TaskSpec, TaskType
from factory.workflow import Workflow, WorkflowConfig

SCENARIO = "telecom_churn"


def _spec(threshold=0.75):
    return TaskSpec(description="t", task_type=TaskType.BINARY_CLASSIFICATION, modality=Modality.TABULAR,
                    target_column="churn", metrics=[MetricTarget(name="recall", threshold=threshold)])


def _plan(name, model, **params):
    return PipelinePlan(name=name, steps=[
        Step(kind=OperatorKind.IMPUTER, action="reuse", operator_ref="MissingValueImputer@1"),
        Step(kind=OperatorKind.MODEL, action="reuse", operator_ref=model, params=params),
    ])


@pytest.fixture
def kg():
    kg = CapabilityKG()
    kg.add_node("Scenario", SCENARIO, name=SCENARIO)
    for ref in ("MissingValueImputer@1", "LogisticRegressionModel@1", "RandomForestModel@1"):
        kg.add_node("Operator", ref, name=ref.split("@")[0], version=1)
    # 历史上逻辑回归 recall 高、随机森林低；一条被作废的历史实验不应计入
    kg.add_validation_run(scenario=f"Scenario:{SCENARIO}", operators=["Operator:LogisticRegressionModel@1"],
                          metrics={"recall": 0.80}, passed=True, origin="factory")
    kg.add_validation_run(scenario=f"Scenario:{SCENARIO}", operators=["Operator:RandomForestModel@1"],
                          metrics={"recall": 0.60}, passed=False, origin="factory")
    kg.add_validation_run(scenario=f"Scenario:{SCENARIO}", operators=["Operator:RandomForestModel@1"],
                          metrics={"recall": 0.99}, passed=False, origin="history", note="数据泄漏，作废")
    return kg


def test_signature_ignores_name_but_not_params():
    a = _plan("a", "LogisticRegressionModel@1", C=1.0)
    assert search.signature(a) == search.signature(_plan("b", "LogisticRegressionModel@1", C=1.0))
    assert search.signature(a) != search.signature(_plan("a", "LogisticRegressionModel@1", C=0.1))


def test_prior_uses_kg_history_and_skips_invalidated(kg):
    lr, ev = search.plan_prior(_plan("lr", "LogisticRegressionModel@1"), _spec(), kg, SCENARIO, {})
    rf, _ = search.plan_prior(_plan("rf", "RandomForestModel@1"), _spec(), kg, SCENARIO, {})
    assert lr > rf, "历史表现好的算子先验更高（失败的验证同样计入，作废的历史实验不计入）"
    assert "KG 1 条" in ev["LogisticRegressionModel@1"] and "按阈值" in ev["MissingValueImputer@1"]
    # 本次运行的观测也会拉动先验
    lowered, _ = search.plan_prior(_plan("lr", "LogisticRegressionModel@1"), _spec(), kg, SCENARIO,
                                   {"LogisticRegressionModel@1": [0.5, 0.5]})
    assert lowered < lr


def test_select_dedups_and_prunes_to_width(kg):
    seen = {search.signature(_plan("old", "LogisticRegressionModel@1"))}
    children = [
        _plan("dup", "LogisticRegressionModel@1"),  # 与已验证方案相同
        _plan("rf", "RandomForestModel@1"),
        _plan("lr_c", "LogisticRegressionModel@1", C=0.1),
        _plan("lr_c_again", "LogisticRegressionModel@1", C=0.1),  # 与本轮另一个子方案相同
    ]
    kept, records = search.select(children, seen=seen, width=1, spec=_spec(), kg=kg, scenario=SCENARIO, observed={})
    assert [p.name for p in kept] == ["lr_c"]
    by_name = {r.plan: r for r in records}
    assert by_name["dup"].prior is None and "重复" in by_name["dup"].reason
    assert "重复" in by_name["lr_c_again"].reason
    assert by_name["lr_c"].selected and not by_name["rf"].selected and "剪枝" in by_name["rf"].reason


def test_workflow_replan_runs_beam_search(tmp_path, churn_csv, fast_verify):
    kg_path = tmp_path / "kg.json"
    shutil.copy(DEFAULT_KG_PATH, kg_path)
    wf = Workflow(rule_based_mock(), kg_path=kg_path, runs_dir=tmp_path / "runs",
                  config=WorkflowConfig(n_candidates=1, max_replans=1, beam_expand=3, verify=fast_verify))
    state = wf.run("预测流失，召回率不低于0.99", str(churn_csv))
    assert state.replans == 1 and len(state.search) == 1
    rnd = state.search[0]
    assert rnd.parents == [state.reports[0].plan_name] and rnd.width == 1
    assert len(rnd.candidates) >= 2, "应扩展出 width × expand 个子方案"
    assert sum(c.selected for c in rnd.candidates) == 1, "只有先验最高的 width 个进入验证"
    assert len(state.reports) == 2
    sigs = [search.signature(p) for p in state.candidates]
    assert len(set(sigs)) == len(sigs), "不会重复验证同一方案"
    report = open(state.artifacts["report_md"], encoding="utf-8").read()
    assert "方案搜索（Beam Search）" in report and ("剪枝" in report or "重复" in report)
