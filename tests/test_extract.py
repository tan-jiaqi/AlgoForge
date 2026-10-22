"""能力抽取：AST 解析（确定性）与基于 MockLLM 的端到端建图。"""

import json

import networkx as nx
import pytest

from factory.extract import (
    DEFAULT_MATERIALS,
    CapabilityCard,
    DocKnowledge,
    ExperimentRecord,
    LessonRecord,
    MetricRecord,
    extract_and_save,
    scan_repo,
)
from factory.kg import CapabilityKG
from factory.llm import MockLLM
from factory.schemas import DiagnosticCode, OperatorKind

REPO = DEFAULT_MATERIALS / "legacy_repo"


def test_scan_repo_extracts_public_functions_and_deps():
    fns = {f.name: f for f in scan_repo(REPO, "churn_v1")}
    assert "_log" not in fns, "私有函数应跳过"
    assert {"clean_total_charges", "target_encode", "train_lgbm_balanced", "recall_at_top_k"} <= set(fns)
    assert fns["train_lgbm_balanced"].dependencies == ["lightgbm"]
    assert "scikit-learn" in fns["target_encode"].dependencies
    assert "空白字符串" in fns["clean_total_charges"].docstring
    assert fns["clean_total_charges"].signature.startswith("clean_total_charges(df, col=")


def _card_for(messages) -> str:
    """按源码里的函数名返回一张确定的能力卡片。"""
    user = messages[1]["content"]
    if "def clean_total_charges" in user:
        card = CapabilityCard(function="clean_total_charges", capability_type="operator", name="TotalChargesCleaner",
                              kind=OperatorKind.IMPUTER, description="把字符串累计消费转为数值",
                              conditions=["string_typed_numeric"])
    elif "def recall_at_top_k" in user:
        card = CapabilityCard(function="recall_at_top_k", capability_type="metric", name="RecallAtTopK", description="前 k 召回")
    else:
        card = CapabilityCard(function="x", capability_type="utility", name="Misc", description="辅助函数")
    return card.model_dump_json()


def _doc_for(messages) -> str:
    user = messages[1]["content"]
    know = DocKnowledge()
    if "文档类型：requirement" in user:
        know.metrics = [MetricRecord(name="recall", threshold=0.75, primary=True)]
        know.constraints = ["recall_first", "interpretable"]
    elif "文档类型：experiment" in user:
        know.experiments = [
            ExperimentRecord(pipeline=["MissingValueImputer", "LightGBMModel"], metrics={"recall": 0.78}, valid=True),
            ExperimentRecord(pipeline=["LightGBMModel"], metrics={"recall": 0.99}, valid=False, note="泄漏"),
            ExperimentRecord(pipeline=["UnknownOp"], metrics={"recall": 0.5}, valid=True),
        ]
        # 实验记录里的指标不应覆盖需求文档定义的场景阈值
        know.metrics = [MetricRecord(name="recall", threshold=None, primary=True)]
    elif "文档类型：failure" in user:
        know.lessons = [LessonRecord(title="空白字符串转换失败", trigger="could not convert string to float",
                                     fix="先 strip 再 to_numeric", error_code=DiagnosticCode.RUNTIME_ERROR,
                                     applies_to=["TotalChargesCleaner", "string_typed_numeric", "NotExisting"])]
    return know.model_dump_json()


@pytest.fixture
def built(tmp_path):
    llm = MockLLM().on("extract_code", _card_for).on("extract_doc", _doc_for)
    kg, result = extract_and_save(llm, DEFAULT_MATERIALS, tmp_path / "kg.json", workers=2)
    return kg, result, tmp_path


def test_build_kg_from_mock_extraction(built):
    kg, result, _ = built
    assert not result.errors
    ops = dict(kg.nodes("Operator"))
    cleaner = ops["Operator:TotalChargesCleaner@1"]
    assert cleaner["origin"] == "extracted" and cleaner["executable"] is False
    assert "pd.to_numeric" in cleaner["code"], "抽取算子应保留参考源码"
    assert "Metric:recall_at_top_k" in kg.g, "metric 类卡片写入为 Metric 节点"
    assert "Operator:Misc@1" not in kg.g, "utility 类卡片不入图"
    assert ops["Operator:LightGBMModel@1"]["executable"] is True

    edge = kg.g.get_edge_data("Scenario:telecom_churn", "Metric:recall")
    assert [d["threshold"] for d in edge.values()] == [0.75], "场景阈值只取自需求文档"
    assert kg.g.nodes["Scenario:telecom_churn"]["constraints"] == ["recall_first", "interpretable"]

    runs = dict(kg.nodes("ValidationRun"))
    assert len(runs) == 2, "流水线中算子全部无法识别的实验应丢弃"
    lesson = kg.search_lessons(error_code=DiagnosticCode.RUNTIME_ERROR)[0]
    targets = set(kg.neighbors(lesson["id"], "APPLIES_TO"))
    assert targets == {"Operator:TotalChargesCleaner@1", "Condition:string_typed_numeric"}


def test_history_ignores_invalid_runs(built):
    kg, _, _ = built
    score, n = kg.historical_score("Operator:LightGBMModel@1", "telecom_churn", "recall")
    assert (score, n) == (0.78, 1), "作废（泄漏）的实验不计入历史分"


def test_saved_outputs_roundtrip(built):
    kg, _, tmp = built
    loaded = CapabilityKG.load(tmp / "kg.json")
    assert loaded.stats() == kg.stats()
    g = nx.read_graphml(tmp / "kg.graphml")
    assert g.number_of_nodes() == kg.g.number_of_nodes()
    raw = json.loads((tmp / "extraction.json").read_text(encoding="utf-8"))
    assert "source" not in raw["functions"][0], "原始抽取结果不重复存源码"


def test_one_failed_file_does_not_abort(tmp_path):
    def flaky(messages):
        if "def target_encode" in messages[1]["content"]:
            raise RuntimeError("boom")
        return _card_for(messages)

    llm = MockLLM().on("extract_code", flaky).on("extract_doc", _doc_for)
    _, result = extract_and_save(llm, DEFAULT_MATERIALS, tmp_path / "kg.json", workers=1)
    assert list(result.errors) == ["churn_v1/encoding.py::target_encode"]
    assert len(result.cards) == 8
