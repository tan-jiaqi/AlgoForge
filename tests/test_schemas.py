import pytest
from pydantic import ValidationError

from factory.schemas import (
    DiagnosticCode,
    MetricTarget,
    Modality,
    OperatorKind,
    PipelinePlan,
    Step,
    SynthSpec,
    TaskSpec,
    TaskType,
)


def _spec(**kw):
    base = dict(
        description="predict churn",
        task_type=TaskType.BINARY_CLASSIFICATION,
        modality=Modality.TABULAR,
        target_column="churn",
        metrics=[MetricTarget(name="recall", threshold=0.7)],
    )
    return TaskSpec(**{**base, **kw})


def test_taskspec_valid():
    assert _spec().primary_metric.name == "recall"


def test_text_task_requires_text_column():
    with pytest.raises(ValidationError):
        _spec(modality=Modality.TEXT)


def test_exactly_one_primary_metric():
    with pytest.raises(ValidationError):
        _spec(metrics=[MetricTarget(name="recall", threshold=0.7), MetricTarget(name="f1", threshold=0.5)])


def test_step_requires_matching_payload():
    with pytest.raises(ValidationError):
        Step(kind=OperatorKind.MODEL, action="reuse")
    with pytest.raises(ValidationError):
        Step(kind=OperatorKind.MODEL, action="synthesize")
    spec = SynthSpec(name="RatioFeatures", kind=OperatorKind.FEATURE, purpose="x", inputs="df", outputs="df")
    with pytest.raises(ValidationError):
        Step(kind=OperatorKind.MODEL, action="synthesize", synth_spec=spec)


def test_plan_must_end_with_single_model():
    model = Step(kind=OperatorKind.MODEL, action="reuse", operator_ref="LightGBMModel@1")
    imputer = Step(kind=OperatorKind.IMPUTER, action="reuse", operator_ref="MissingValueImputer@1")
    PipelinePlan(name="ok", steps=[imputer, model])
    with pytest.raises(ValidationError):
        PipelinePlan(name="bad", steps=[model, imputer])
    with pytest.raises(ValidationError):
        PipelinePlan(name="bad", steps=[model, model])


def test_diagnostic_category_routes_repair_vs_replan():
    assert DiagnosticCode.RUNTIME_ERROR.category == "code"
    assert DiagnosticCode.METRIC_BELOW_THRESHOLD.category == "metric"
