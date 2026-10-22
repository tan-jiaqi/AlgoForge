"""里程碑 4：生成新算子、诊断分流修复、注册复用与经验沉淀。"""

import shutil

import pytest

from factory.agents.planner import Planner
from factory.agents.rules import rule_based_mock
from factory.agents.synthesizer import extract_code
from factory.codegen import render_pipeline
from factory.kg import DEFAULT_KG_PATH, CapabilityKG
from factory.operators import registry
from factory.operators.synthesized import exec_operator_source, load_saved_operators, save_operator
from factory.schemas import (
    DiagnosticCode, MetricTarget, Modality, OperatorKind, PipelinePlan, Step, SynthSpec, TaskSpec, TaskType,
)
from factory.verify import Verifier
from factory.workflow import Workflow, WorkflowConfig

FEATURE_SRC = '''
from sklearn.base import TransformerMixin
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class TenureX2(TransformerMixin, Operator):
    meta = OperatorMeta(name="TenureX2", kind=OperatorKind.FEATURE, version=2, description="t",
                        origin="synthesized", params={"factor": 2})

    def __init__(self, factor=2):
        self.factor = factor

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["tenure_x"] = X["tenure"] * self.factor
        return X
'''


@pytest.fixture
def kg_path(tmp_path):
    path = tmp_path / "kg.json"
    shutil.copy(DEFAULT_KG_PATH, path)
    return path


def _spec(threshold=0.4):
    return TaskSpec(description="t", task_type=TaskType.BINARY_CLASSIFICATION, modality=Modality.TABULAR,
                    target_column="churn", metrics=[MetricTarget(name="recall", threshold=threshold)])


def test_extract_code():
    assert extract_code("说明\n```python\nx = 1\n```\n结尾") == "x = 1\n"
    assert extract_code("x = 2") == "x = 2\n"


def test_save_load_and_reuse_source(tmp_path):
    kg = tmp_path / "kg.json"
    save_operator(FEATURE_SRC, "TenureX2@2", kg)
    (tmp_path / "operators" / "Evil_v1.py").write_text("import os\n", encoding="utf-8")
    assert load_saved_operators(kg) == ["TenureX2@2"], "未通过静态检查的文件应被跳过"
    assert registry.source_of("TenureX2") == FEATURE_SRC
    assert load_saved_operators(kg) == [], "重复加载应是幂等的"


def test_reused_synthesized_op_runs_in_sandbox_and_codegen(churn_csv, fast_verify):
    exec_operator_source(FEATURE_SRC)
    R = lambda k, r: Step(kind=k, action="reuse", operator_ref=r)
    plan = PipelinePlan(name="p", steps=[
        R(OperatorKind.IMPUTER, "MissingValueImputer@1"), R(OperatorKind.FEATURE, "TenureX2@2"),
        R(OperatorKind.ENCODER, "OneHotCategoricalEncoder@1"), R(OperatorKind.MODEL, "LightGBMModel@1"),
    ])
    report = Verifier(fast_verify).verify(plan, _spec(threshold=0.2), str(churn_csv))
    assert report.passed, report.diagnostics
    code = render_pipeline(plan, _spec())
    assert "class TenureX2" in code, "以前沉淀的生成算子应内嵌源码"
    compile(code, "pipeline.py", "exec")


def test_model_must_return_ndarray(churn_csv, fast_verify):
    src = """
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind

class DfModel(ClassifierMixin, Operator):
    meta = OperatorMeta(name="DfModel", kind=OperatorKind.MODEL, description="t", origin="synthesized")

    def fit(self, X, y):
        self.m_ = LogisticRegression(max_iter=500).fit(X, y)
        self.classes_ = self.m_.classes_
        return self

    def predict(self, X):
        return self.m_.predict(X)

    def predict_proba(self, X):
        return pd.DataFrame(self.m_.predict_proba(X))
"""
    from factory.schemas import GeneratedOperator

    spec = SynthSpec(name="DfModel", kind=OperatorKind.MODEL, purpose="t", inputs="x", outputs="y")
    plan = PipelinePlan(name="p", steps=[
        Step(kind=OperatorKind.IMPUTER, action="reuse", operator_ref="MissingValueImputer@1"),
        Step(kind=OperatorKind.ENCODER, action="reuse", operator_ref="OneHotCategoricalEncoder@1"),
        Step(kind=OperatorKind.MODEL, action="synthesize", synth_spec=spec),
    ])
    r = Verifier(fast_verify).verify(plan, _spec(), str(churn_csv), {"DfModel": GeneratedOperator(spec=spec, source=src)})
    assert r.diagnostics[0].code == DiagnosticCode.BAD_SHAPE and r.diagnostics[0].operator == "DfModel@1"


def test_planner_synthesis_checks(kg_path, churn_csv):
    exec_operator_source(FEATURE_SRC)
    planner = Planner(rule_based_mock(), CapabilityKG.load(kg_path))
    model = Step(kind=OperatorKind.MODEL, action="reuse", operator_ref="LightGBMModel@1")
    synth = lambda name, ref=None: Step(kind=OperatorKind.FEATURE, action="synthesize", synth_spec=SynthSpec(
        name=name, kind=OperatorKind.FEATURE, purpose="p", inputs="i", outputs="o", reference=ref))
    errs = lambda *steps: " ".join(planner.plan_errors(PipelinePlan(name="x", steps=[*steps, model]), _spec()))
    assert errs(synth("NewFeature", "TenureBucketizer@1")) == ""
    assert "应使用 reuse" in errs(synth("TenureX2"))
    assert "不在知识图谱中" in errs(synth("Other", "Nope@1"))
    assert "超过上限" in errs(synth("A"), synth("B"), synth("C"))
    no_synth = Planner(rule_based_mock(), CapabilityKG.load(kg_path), allow_synthesis=False)
    assert "只允许 reuse" in " ".join(no_synth.plan_errors(PipelinePlan(name="x", steps=[synth("A"), model]), _spec()))


def test_secondary_metric_thresholds_must_be_grounded():
    spec = TaskSpec(description="d", task_type=TaskType.BINARY_CLASSIFICATION, modality=Modality.TABULAR,
                    target_column="y", metrics=[MetricTarget(name="recall", threshold=0.75),
                                                MetricTarget(name="roc_auc", threshold=0.8, primary=False),
                                                MetricTarget(name="recall_at_20", threshold=0.7, primary=False)])
    known = [{"name": "roc_auc", "threshold": 0.8}, {"name": "recall_at_20", "threshold": None}]
    kept = Planner._grounded_metrics(spec, "召回优先", known)
    assert [m.name for m in kept.metrics] == ["recall", "roc_auc"], "无出处的 recall_at_20 阈值应被去掉"
    assert len(Planner._grounded_metrics(spec, "recall@20 不低于 0.7", known).metrics) == 3


def test_synthesize_repair_register_then_reuse(kg_path, churn_csv, fast_verify, tmp_path):
    """mock 首版清洗算子故意留缺陷：第一次运行 生成 → E_NAN → 修复 → 通过 → 注册 → 提炼经验；
    第二次运行直接复用已注册的算子，不再生成与修复。"""
    cfg = WorkflowConfig(n_candidates=1, verify=fast_verify)
    first = Workflow(rule_based_mock(), kg_path=kg_path, runs_dir=tmp_path / "runs", config=cfg)
    s1 = first.run("预测流失，需可解释，召回率不低于0.4", str(churn_csv))
    assert s1.status == "succeeded", s1.log
    assert "TotalChargesCleaner" in s1.generated
    assert [r.diagnostic.code for r in s1.repairs] == [DiagnosticCode.NAN_OUTPUT] and s1.repairs[0].fixed
    assert s1.history and s1.history[0].diagnostics[0].code == DiagnosticCode.NAN_OUTPUT, "修复前的验证结果应保留"
    assert s1.registered == ["TotalChargesCleaner@2"]
    assert (kg_path.parent / "operators" / "TotalChargesCleaner_v2.py").exists()

    kg = CapabilityKG.load(kg_path)
    node = kg.g.nodes["Operator:TotalChargesCleaner@2"]
    assert node["executable"] and node["origin"] == "synthesized"
    assert "Operator:TotalChargesCleaner@1" in kg.neighbors("Operator:TotalChargesCleaner@2", "DERIVED_FROM")
    lesson = kg.g.nodes[s1.lessons_learned[0]]
    assert lesson["error_code"] == "E_NAN" and lesson["origin"] == "factory"
    src = kg.neighbors(s1.lessons_learned[0], "LEARNED_FROM")
    assert src == [f"ValidationRun:{s1.run_id}-{s1.best_plan}"], "经验应连到用了该算子的方案的验证记录"

    # 模拟新进程：注册表里没有这个算子，需要从 operators/ 目录加载
    registry._ops.pop("TotalChargesCleaner@2")
    registry.sources.pop("TotalChargesCleaner@2")
    second = Workflow(rule_based_mock(), kg_path=kg_path, runs_dir=tmp_path / "runs", config=cfg)
    s2 = second.run("预测流失，需可解释，召回率不低于0.4", str(churn_csv))
    assert s2.status == "succeeded", s2.log
    first_step = next(p for p in s2.candidates if p.name == s2.best_plan).steps[0]
    assert (first_step.action, first_step.operator_ref) == ("reuse", "TotalChargesCleaner@2")
    assert not s2.generated and not s2.repairs and not s2.registered
