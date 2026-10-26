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


def test_canonical_metric():
    from factory.schemas import canonical_metric

    assert canonical_metric("RecallAtTopK", {"k": 0.2}) == canonical_metric("recall@20%") == "recall_at_20"
    assert canonical_metric("AUC") == "roc_auc" and canonical_metric("f1_macro") == "f1_macro"
    assert canonical_metric("recall_at_top_k") == "recall_at_top_k", "缺少 k 时不能猜"


def test_duplicate_lessons_merge_on_insert(kg):
    a = kg.add_lesson(title="样本不平衡导致默认阈值下召回率偏低", trigger="t1", fix="f1",
                      error_code=DiagnosticCode.METRIC_BELOW_THRESHOLD, applies_to=["Operator:LightGBMModel@1"])
    b = kg.add_lesson(title="样本不平衡导致默认模型召回率偏低", trigger="t2", fix="f2",
                      error_code=DiagnosticCode.METRIC_BELOW_THRESHOLD, applies_to=["Operator:RandomForestModel@1"])
    assert a == b, "诊断码相同且标题相似的经验应合并"
    assert kg.g.nodes[a]["aliases"] == ["样本不平衡导致默认模型召回率偏低"]
    assert set(kg.neighbors(a, "APPLIES_TO")) == {"Operator:LightGBMModel@1", "Operator:RandomForestModel@1"}
    other_code = kg.add_lesson(title="样本不平衡导致默认模型召回率偏低", trigger="t", fix="f",
                               error_code=DiagnosticCode.UNSTABLE)
    distinct = kg.add_lesson(title="预测阶段出现未见类别导致编码报错", trigger="t", fix="f",
                             error_code=DiagnosticCode.METRIC_BELOW_THRESHOLD)
    assert len({a, other_code, distinct}) == 3, "诊断码不同或内容不同的经验不能合并"


def test_tidy_existing_graph(kg):
    # 模拟旧图谱：两条重复经验各自成节点，同一指标有两个名字
    for i, title in enumerate(["样本不平衡导致默认阈值下召回率偏低", "样本不平衡导致默认模型召回率偏低"]):
        kg.add_node("Lesson", f"l{i}", title=title, trigger="t", fix="f", error_code="M_THRESHOLD", hits=i + 1)
    kg.add_edge("Lesson:l1", "APPLIES_TO", "Operator:LightGBMModel@1")
    kg.add_node("Metric", "recall_at_top_k", name="recall_at_top_k", params={"k": 0.2}, description="前 20% 召回")
    kg.add_metric("Scenario:churn", "recall_at_20")
    log = kg.tidy()
    assert len(log) == 2
    assert "Lesson:l1" not in kg.g and kg.g.nodes["Lesson:l0"]["hits"] == 3
    assert kg.neighbors("Lesson:l0", "APPLIES_TO") == ["Operator:LightGBMModel@1"], "被合并节点的边要转移"
    assert "Metric:recall_at_top_k" not in kg.g and kg.g.nodes["Metric:recall_at_20"]["description"] == "前 20% 召回"
    assert kg.tidy() == [], "整理应是幂等的"


def test_lessons_filtered_by_modality_and_ranked_by_conditions():
    """中文文本任务的检索不应被流失场景的表格经验占满（演示 01 曾把字符级 TF-IDF 经验挤出前 5）。"""
    from factory.kg import DEFAULT_KG_PATH, CapabilityKG
    from factory.schemas import Modality

    kg = CapabilityKG.load(DEFAULT_KG_PATH)
    query = "把客服工单自动分到网络故障、账单争议、套餐变更、销户投诉四类，宏平均 F1 不低于 0.8 id_columns imbalanced text_input"
    titles = [x["title"] for x in kg.search_lessons(query, modality=Modality.TEXT,
                                                    conditions=["id_columns", "imbalanced", "text_input"], top_k=5)]
    assert "中文短文本字符级 TF-IDF 建模" in titles
    assert "数值列含空白字符导致类型转换失败" not in titles, "表格专用经验不应进入文本任务"
    assert "样本不平衡导致默认模型召回率偏低" in titles, "跨模态通用经验仍可迁移"
    tabular = [x["title"] for x in kg.search_lessons("预测流失 string_typed_numeric", modality=Modality.TABULAR,
                                                     conditions=["string_typed_numeric"], top_k=10)]
    assert "数值列含空白字符导致类型转换失败" in tabular
