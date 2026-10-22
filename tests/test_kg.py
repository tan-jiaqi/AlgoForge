"""知识图谱：写入约束、检索排序、经验检索。"""

import pytest

from factory.kg import CapabilityKG, tokenize
from factory.operators import registry
from factory.schemas import DiagnosticCode, Modality, OperatorKind


@pytest.fixture
def kg():
    kg = CapabilityKG()
    kg.add_registry(registry)
    sc = kg.add_scenario("churn", title="流失", task_type="binary_classification", modality="tabular")
    kg.add_validation_run(scenario=sc, operators=["Operator:RandomForestModel@1"], metrics={"recall": 0.9}, passed=True, origin="history")
    kg.add_validation_run(scenario=sc, operators=["Operator:LightGBMModel@1"], metrics={"recall": 0.6}, passed=True, origin="history")
    return kg


def test_tokenize_mixed_language():
    toks = tokenize("LightGBMModel 召回优先")
    assert {"light", "召回", "优先", "召"} <= set(toks)


def test_edge_and_node_validation(kg):
    with pytest.raises(ValueError):
        kg.add_node("Nope", "x")
    with pytest.raises(ValueError):
        kg.add_edge("Scenario:churn", "LIKES", "Scenario:churn")
    with pytest.raises(KeyError):
        kg.add_edge("Scenario:churn", "RUN_ON", "Scenario:missing")


def test_duplicate_edges_are_merged(kg):
    before = kg.g.number_of_edges()
    kg.add_metric("Scenario:churn", "recall", 0.7, True)
    kg.add_metric("Scenario:churn", "recall", 0.8, True)
    assert kg.g.number_of_edges() == before + 1
    assert list(kg.g.get_edge_data("Scenario:churn", "Metric:recall").values())[0]["threshold"] == 0.8


def test_search_filters_by_kind_and_modality(kg):
    hits = kg.search_operators("模型", kind=OperatorKind.MODEL, modality=Modality.TEXT)
    assert [h.key for h in hits] == ["LogisticRegressionModel@1"]


def test_history_influences_ranking(kg):
    hits = kg.search_operators("模型", kind=OperatorKind.MODEL, scenario="churn", metric="recall")
    assert hits[0].key == "RandomForestModel@1"
    assert hits[0].breakdown["history"] == 0.9
    assert any("历史验证" in e for e in hits[0].evidence)


def test_conditions_influence_ranking(kg):
    hits = kg.search_operators("", kind=OperatorKind.MODEL, conditions=["interpretable"])
    assert hits[0].key == "LogisticRegressionModel@1"


def test_lessons_filtered_by_code(kg):
    kg.add_lesson(title="空白字符串", trigger="could not convert string to float", fix="to_numeric", error_code=DiagnosticCode.RUNTIME_ERROR)
    kg.add_lesson(title="召回不足", trigger="recall 低", fix="class_weight", error_code=DiagnosticCode.METRIC_BELOW_THRESHOLD)
    found = kg.search_lessons("convert string", error_code=DiagnosticCode.RUNTIME_ERROR)
    assert [x["title"] for x in found] == ["空白字符串"]
