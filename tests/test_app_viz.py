"""知识图谱可视化与 Streamlit 界面的冒烟测试（不调用 LLM）。"""

import pytest

from factory.kg import DEFAULT_KG_PATH, CapabilityKG
from factory.kg_viz import render_html, subgraph
from factory.utils import ROOT


def test_render_full_and_focus(tmp_path):
    kg = CapabilityKG.load(DEFAULT_KG_PATH)
    page = render_html(kg, tmp_path / "kg.html")
    assert "vis-network" in page and "Operator:LightGBMModel@1" in page and "Lesson" in page
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
