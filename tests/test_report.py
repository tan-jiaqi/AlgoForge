"""验证报告：内容完整、HTML 转义正确；候选排序规则。"""

from factory.report import build_blocks, to_html, to_markdown, write_report
from factory.schemas import (
    Diagnostic, DiagnosticCode, GeneratedOperator, MetricTarget, Modality, OperatorKind, PipelinePlan, RepairRecord,
    RunState, Step, SynthSpec, TaskSpec, TaskType, ValidationReport,
)
from factory.workflow import rank_reports

SPEC = TaskSpec(description="d", task_type=TaskType.BINARY_CLASSIFICATION, modality=Modality.TABULAR,
                target_column="churn", metrics=[MetricTarget(name="recall", threshold=0.75),
                                                MetricTarget(name="roc_auc", threshold=0.8, primary=False)])


def _report(name, passed, recall, auc=0.8, fit=1.0, diags=()):
    return ValidationReport(plan_name=name, passed=passed, metrics={"recall": recall, "roc_auc": auc},
                            resources={"fit_seconds": fit}, diagnostics=list(diags))


def test_rank_reports():
    rs = [_report("fail_high", False, 0.95), _report("pass_low", True, 0.76), _report("pass_high", True, 0.80),
          _report("tie_slow", True, 0.80, fit=5.0), _report("tie_better_auc", True, 0.80, auc=0.85, fit=9.0)]
    assert [r.plan_name for r in rank_reports(rs, SPEC)] == ["tie_better_auc", "pass_high", "tie_slow", "pass_low", "fail_high"]


def _state():
    synth = SynthSpec(name="Cleaner", kind=OperatorKind.IMPUTER, purpose="清洗 <脏> 数据", inputs="i", outputs="o",
                      reference="Cleaner@1")
    plan = PipelinePlan(name="best", rationale="理由", steps=[
        Step(kind=OperatorKind.IMPUTER, action="synthesize", synth_spec=synth, rationale="需要清洗"),
        Step(kind=OperatorKind.MODEL, action="reuse", operator_ref="LightGBMModel@1", params={"num_leaves": 15}),
    ])
    diag = Diagnostic(code=DiagnosticCode.NAN_OUTPUT, stage="contract", message="输出引入了新的 NaN", operator="Cleaner@2")
    return RunState(
        run_id="r1", request="预测 <script>alert(1)</script>", data_path="d.csv", status="succeeded", spec=SPEC,
        candidates=[plan], best_plan="best", reports=[_report("best", True, 0.8)], history=[_report("best", False, 0.0, diags=[diag])],
        generated={"Cleaner": GeneratedOperator(spec=synth, source="x = 2\n", version=2, revision=1)},
        repairs=[RepairRecord(operator="Cleaner@2", revision=1, diagnostic=diag, fixed=True, before="x = 1\n", after="x = 2\n")],
        registered=["Cleaner@2"], retrieval={"scenario": "s", "operator_options": {"model": [
            {"ref": "LightGBMModel@1", "executable": True, "score": 0.5, "breakdown": {"text": 1, "condition": 0, "history": 0.5},
             "evidence": ["历史验证 1 次"]}]}, "lessons": [{"title": "L", "fix": "f", "error_code": "E_NAN"}]},
        verify_config={"cv_folds": 5, "seeds": [1], "timeout_seconds": 60, "memory_mb": 1024},
        timings={"PLAN": 1.0, "VERIFY": 2.0}, log=["[PLAN] ok"],
    )


def test_report_contains_key_sections():
    md = to_markdown(build_blocks(_state(), pipeline_code="print('hi')"))
    for text in ["需求理解", "知识检索", "候选方案比较", "最优方案：best", "能力生成与修复", "验证机制", "知识沉淀",
                 "Cleaner@2", "E_NAN", "已修复", "-x = 1", "+x = 2", "历史验证 1 次", "num_leaves=15", "print('hi')"]:
        assert text in md, text


def test_html_is_escaped(tmp_path):
    (tmp_path / "pipeline.py").write_text("a < b\n", encoding="utf-8")
    paths = write_report(_state(), tmp_path)
    page = open(paths["report_html"], encoding="utf-8").read()
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page
    assert "a &lt; b" in page and "<table>" in page and "prefers-color-scheme" in page
    assert open(paths["report_md"], encoding="utf-8").read().startswith("# AlgoForge 验证报告")


def test_report_without_spec():
    state = RunState(run_id="r", request="q", data_path="d", status="failed", log=["[PLAN] 出错"])
    md = to_markdown(build_blocks(state))
    assert "未产生 TaskSpec" in md and "[PLAN] 出错" in md
    assert "<html" in to_html(build_blocks(state))
