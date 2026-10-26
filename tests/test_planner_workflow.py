"""Planner 的语义校验与重试；基于 MockLLM 的端到端工作流；代码生成。"""

import shutil
import subprocess
import sys

import pytest

from factory.agents.planner import PlanSet, Planner, resolve_scenario
from factory.agents.rules import rule_based_mock
from factory.codegen import render_pipeline
from factory.kg import DEFAULT_KG_PATH, CapabilityKG
from factory.llm import MockLLM
from factory.operators import registry
from factory.profile import profile
from factory.schemas import OperatorKind, PipelinePlan, Step
from factory.workflow import Workflow, WorkflowConfig


@pytest.fixture
def kg_path(tmp_path):
    path = tmp_path / "kg.json"
    shutil.copy(DEFAULT_KG_PATH, path)
    return path


def _spec_and_ctx(planner, csv):
    prof = profile(csv)
    spec = planner.understand("预测流失，需可解释，召回优先", prof, target=None, scenario=None)
    return spec, prof, planner.retrieve(spec, prof, scenario=None)


def test_rule_mock_understand_and_retrieve(kg_path, churn_csv):
    planner = Planner(rule_based_mock(), CapabilityKG.load(kg_path), allow_synthesis=False)
    spec, _, ctx = _spec_and_ctx(planner, churn_csv)
    assert spec.target_column == "churn" and spec.primary_metric.name == "recall"
    assert set(spec.constraints) == {"interpretable", "recall_first"}
    assert "model" in ctx.options and ctx.lessons
    assert all(o["ref"] in registry for opts in ctx.options.values() for o in opts), "不允许生成时只检索可执行算子"


def test_planner_retries_on_unknown_operator(kg_path, churn_csv):
    mock = rule_based_mock()
    planner = Planner(mock, CapabilityKG.load(kg_path))
    spec, prof, ctx = _spec_and_ctx(planner, churn_csv)
    bad = PlanSet(plans=[PipelinePlan(name="x", steps=[Step(kind=OperatorKind.MODEL, action="reuse", operator_ref="XGBoost@1")])])
    mock.queue("plan", bad)
    plans = planner.propose(spec, prof, ctx, n_candidates=1)
    plan_calls = [c for c in mock.calls if c.tag == "plan"]
    assert len(plan_calls) == 2 and "未知算子 XGBoost@1" in plan_calls[1].messages[-1]["content"]
    assert plans[0].steps[-1].operator_ref in registry


def test_planner_rejects_bad_params_and_modality(kg_path, churn_csv):
    planner = Planner(MockLLM(), CapabilityKG.load(kg_path))
    spec, _, _ = _spec_and_ctx(Planner(rule_based_mock(), CapabilityKG.load(kg_path)), churn_csv)
    plan = PipelinePlan(name="x", steps=[
        Step(kind=OperatorKind.VECTORIZER, action="reuse", operator_ref="TfidfTextVectorizer@1"),
        Step(kind=OperatorKind.MODEL, action="reuse", operator_ref="LightGBMModel@1", params={"depth": 3}),
    ])
    errors = " ".join(planner.plan_errors(plan, spec))
    assert "不支持 tabular" in errors and "没有参数" in errors


def test_resolve_scenario(kg_path):
    kg = CapabilityKG.load(kg_path)
    assert resolve_scenario(kg, "data/churn.csv") == "telecom_churn"
    assert resolve_scenario(kg, "/abs/path/data/churn.csv") == "telecom_churn"
    assert resolve_scenario(kg, "other.csv") is None


def test_workflow_end_to_end_mock(kg_path, churn_csv, fast_verify, tmp_path):
    wf = Workflow(rule_based_mock(), kg_path=kg_path, runs_dir=tmp_path / "runs",
                  config=WorkflowConfig(n_candidates=2, verify=fast_verify.model_copy(update={"report_metrics": []})))
    state = wf.run("预测流失，需可解释，召回优先，召回率不低于0.4", str(churn_csv))
    assert state.status == "succeeded", state.log
    assert state.best_plan and state.kg_writes and len(state.kg_writes) == len(state.reports)

    kg = CapabilityKG.load(kg_path)
    runs = [d for _, d in kg.nodes("ValidationRun") if d.get("origin") == "factory"]
    assert len(runs) == len(state.reports), "每个候选的验证结果都应回写"
    assert any(n for n, d in kg.nodes("Scenario") if d["key"] == "churn"), "新数据集应创建场景"
    assert sum(d["hits"] for _, d in kg.nodes("Lesson")) > 0, "用到的经验应累计命中次数"

    # 生成的脚本可以独立训练与预测
    script = state.artifacts["pipeline"]
    out = subprocess.run([sys.executable, script, "--data", str(churn_csv), "--model", str(tmp_path / "m.joblib"),
                          "--predict", str(churn_csv)], capture_output=True, text=True, cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert "prediction" in out.stdout
    assert {"report_md", "report_html"} <= set(state.artifacts)


def test_workflow_replans_on_metric_failure(kg_path, churn_csv, fast_verify, tmp_path):
    wf = Workflow(rule_based_mock(), kg_path=kg_path, runs_dir=tmp_path / "runs",
                  config=WorkflowConfig(n_candidates=1, max_replans=1, verify=fast_verify))
    state = wf.run("预测流失，召回率不低于0.99", str(churn_csv))
    assert state.status == "failed" and state.replans == 1
    assert len(state.reports) == 2 and len({r.plan_name for r in state.reports}) == 2
    assert "未通过验证" in open(state.artifacts["pipeline"], encoding="utf-8").read()


def test_render_pipeline_uses_defaults_and_rationale():
    plan = PipelinePlan(name="p", steps=[Step(kind=OperatorKind.MODEL, action="reuse", operator_ref="LightGBMModel@1",
                                              params={"num_leaves": 15}, rationale="不平衡数据")])
    from factory.schemas import MetricTarget, Modality, TaskSpec, TaskType
    spec = TaskSpec(description="d", task_type=TaskType.BINARY_CLASSIFICATION, modality=Modality.TABULAR,
                    target_column="y", metrics=[MetricTarget(name="recall", threshold=0.7)])
    code = render_pipeline(plan, spec)
    assert "LightGBMModel(n_estimators=300" in code and "num_leaves=15" in code and "# 不平衡数据" in code
    compile(code, "pipeline.py", "exec")


def test_literal_handles_special_floats():
    from factory.codegen import literal

    code = literal((-1, 6, float("inf"))) + literal({"a": [float("nan"), "x"]}) + literal((1,))
    assert "float('inf')" in code and "float('nan')" in code and "(1,)" in code
    assert eval(literal((-1, float("inf")))) == (-1, float("inf"))  # noqa: S307 - 测试字面量可还原


def test_plan_without_encoder_rejected_on_categorical_data(churn_csv):
    """表格数据有类别列时，缺少编码步骤的方案在规划阶段就被拒绝（search_3 曾因此把 customer_id 送进逻辑回归）。"""
    from factory.agents.planner import Planner
    from factory.profile import profile
    from factory.schemas import Modality, PipelinePlan, Step, TaskSpec, TaskType, MetricTarget

    spec = TaskSpec(description="d", task_type=TaskType.BINARY_CLASSIFICATION, modality=Modality.TABULAR,
                    target_column="churn", metrics=[MetricTarget(name="recall", threshold=0.7)])
    prof = profile(str(churn_csv), "churn")
    R = lambda k, r: Step(kind=k, action="reuse", operator_ref=r)
    bare = PipelinePlan(name="p", steps=[R("imputer", "MissingValueImputer@1"), R("scaler", "StandardFeatureScaler@1"),
                                         R("model", "LogisticRegressionModel@1")])
    assert "encoder" in " ".join(Planner.data_errors(bare, spec, prof))
    ok = bare.model_copy(update={"steps": [*bare.steps[:1], R("encoder", "OneHotCategoricalEncoder@1"), *bare.steps[1:]]})
    assert Planner.data_errors(ok, spec, prof) == []
