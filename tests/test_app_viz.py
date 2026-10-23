"""知识图谱可视化与 Streamlit 界面的冒烟测试（不调用 LLM）。"""

import pytest

from factory.kg import DEFAULT_KG_PATH, CapabilityKG
from factory.kg_viz import render_html, subgraph
from factory.utils import ROOT


def test_overview_folds_details():
    from factory.kg_viz import overview_graph

    kg = CapabilityKG.load(ROOT / "examples" / "demo" / "kg" / "capability_kg.json")
    g = overview_graph(kg)
    types = {d["type"] for _, d in g.nodes(data=True)}
    assert types == {"Scenario", "Metric", "Operator", "Lesson"}, "概览只保留主干节点"
    assert g.number_of_edges() < kg.g.number_of_edges() / 3
    assert not [n for n in g if g.degree(n) == 0 and g.nodes[n]["type"] != "Operator"], "概览中不应有孤立的指标或经验"
    op = g.nodes["Operator:TotalChargesCleaner@2"]
    assert "string_typed_numeric" in op["conditions"] and op["history"], "条件与验证记录折叠进算子属性"
    edge = next(d for d in g.get_edge_data("Operator:TotalChargesCleaner@2", "Scenario:telecom_churn").values())
    assert edge["rel"] == "VALIDATED_ON" and edge["runs"] >= 1
    assert any(d["rel"] == "DERIVED_FROM" for *_, d in g.edges(data=True))


def test_overview_links_condition_only_lesson_to_scenario():
    """只关联到条件的经验在概览中应连到它来源所属的场景，而不是成为孤立点。"""
    from factory.kg_viz import overview_graph

    kg = CapabilityKG.load(DEFAULT_KG_PATH)
    lid = kg.add_lesson(title="仅关联条件的测试经验", trigger="t", fix="f", applies_to=["Condition:text_input"],
                        learned_from="Source:docs/tuning_notes.md")
    g = overview_graph(kg)
    assert list(g.successors(lid)) == ["Scenario:telecom_churn"]


def test_render_full_and_focus(tmp_path):
    kg = CapabilityKG.load(DEFAULT_KG_PATH)
    page = render_html(kg, tmp_path / "kg.html")
    assert "vis-network" in page and "Operator:LightGBMModel@1" in page and "Source:" not in page, "默认概览不含出处节点"
    full = render_html(kg, view="full")
    assert "Source:" in full and "Condition:" in full
    with pytest.raises(ValueError):
        render_html(kg, view="nope")
    focus = "Operator:TotalChargesCleaner@1"
    g = subgraph(kg, focus=focus, radius=1)
    assert focus in g and "Condition:string_typed_numeric" in g and len(g) < len(kg.g)
    only_ops = subgraph(kg, types=["Operator"])
    assert {d["type"] for _, d in only_ops.nodes(data=True)} == {"Operator"}
    with pytest.raises(KeyError):
        subgraph(kg, focus="Operator:Nope@1")


def test_streamlit_app_loads():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("AlgoForge" in t.value for t in at.title)
    assert len(at.tabs) >= 3


def test_streamlit_mock_run(tmp_path, monkeypatch):
    """在界面上点击"开始"（mock 模式、不回写），应完成运行并展示结果，且不改动仓库中的知识图谱。"""
    from streamlit.testing.v1 import AppTest

    before = DEFAULT_KG_PATH.read_bytes()
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    at.selectbox[1].select("mock").run()
    at.number_input[0].set_value(1).run()
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert [s.label for s in at.status] == ["完成：succeeded"]
    assert DEFAULT_KG_PATH.read_bytes() == before
