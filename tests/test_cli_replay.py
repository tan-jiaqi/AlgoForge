"""CLI 与离线回放：仓库中的录制数据能完整复现能力抽取与端到端演示。"""

import json
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from factory.cli import app
from factory.extract import extract_and_save
from factory.kg import DEFAULT_KG_PATH, CapabilityKG
from factory.llm import make_llm
from factory.utils import ROOT

runner = CliRunner()


def test_cli_basic_commands():
    for args in (["operators"], ["kg", "stats"], ["kg", "search", "可解释 模型", "--kind", "model"],
                 ["kg", "lessons", "--code", "E_RUNTIME"], ["--help"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, (args, result.output)
    assert "LightGBMModel@1" in runner.invoke(app, ["operators"]).output


def test_extract_replay_reproduces_committed_kg(tmp_path):
    """用录制的 LLM 响应离线重新抽取，结果应与仓库中的 kg/extraction.json 一致。"""
    kg, result = extract_and_save(make_llm("replay"), out=tmp_path / "kg.json")
    committed = json.loads((ROOT / "kg" / "extraction.json").read_text(encoding="utf-8"))
    assert result.model_dump(mode="json") == committed
    assert kg.stats() == CapabilityKG.load(DEFAULT_KG_PATH).stats()


@pytest.mark.slow
def test_demo_replay_matches_examples(tmp_path):
    """完整演示离线回放（约 1 分钟）：结果与 examples/demo 一致，生成的脚本都能独立运行。"""
    from factory.demo import STEPS, run_demo

    states = run_demo(make_llm("replay"), tmp_path)
    for (run_id, _, data, _), state in zip(STEPS, states):
        ref = json.loads((ROOT / "examples" / "demo" / "runs" / run_id / "state.json").read_text(encoding="utf-8"))
        assert state.status == ref["status"] == "succeeded"
        assert [[r.plan_name, r.metrics] for r in state.reports] == [[r["plan_name"], r["metrics"]] for r in ref["reports"]]
        script = tmp_path / "runs" / run_id / "pipeline.py"
        out = subprocess.run([sys.executable, str(script), "--data", str(ROOT / data), "--model", str(tmp_path / "m.joblib")],
                             capture_output=True, text=True, cwd=tmp_path)
        assert out.returncode == 0, out.stderr
    assert states[1].registered and any(
        s.action == "reuse" and s.operator_ref in states[1].registered for p in states[2].candidates for s in p.steps
    ), "第三个任务应复用第二个任务注册的算子"
    assert states[0].repairs and states[0].repairs[0].fixed and states[0].lessons_learned
