"""真实响应的溯源与完整离线执行；不以规则响应替换任何 Agent。"""

import json

import pytest

from factory.evaluation import DEFAULT_EVIDENCE, complete_repair, evaluation_outcome, measurement, read_state, run_evaluation
from factory.llm import request_hash


def test_committed_evaluation_provenance():
    manifest = json.loads((DEFAULT_EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["repeats"] == 3
    assert manifest["environment"]["data_sha256"]
    groups = {job["group"] for job in manifest["jobs"]}
    assert {"kg", "probe", "repair_natural", "repair_injected", "reuse", "search"} <= groups
    for job in manifest["jobs"]:
        cassettes = list((DEFAULT_EVIDENCE / "cassettes" / job["name"]).glob("*.json"))
        assert cassettes, job["name"]
        for path in cassettes:
            raw = json.loads(path.read_text(encoding="utf-8"))
            assert raw["source"] == "remote", path
            assert raw["usage"]["total_tokens"] > 0
            assert path.stem == f"{raw['tag']}-{request_hash(raw['model'], raw['messages'], raw['json_mode'])}"
        state = read_state(DEFAULT_EVIDENCE, job["name"])
        assert not any("出错：" in entry for entry in state.log), job["name"]
        assert all(c["source"] == "remote" for c in state.llm_calls), job["name"]
    for i in range(1, 4):
        repair = read_state(DEFAULT_EVIDENCE, f"repair_injected_{i}")
        assert complete_repair(repair)
        injection = json.loads((DEFAULT_EVIDENCE / f"repair_injected_{i}" / "injection.json").read_text(encoding="utf-8"))
        assert injection["original_validation"]["passed"]
        assert injection["original_sha256"] != injection["injected_sha256"]
        reuse = read_state(DEFAULT_EVIDENCE, f"repair_injected_{i}_reuse")
        assert reuse.status == "succeeded"
        assert any(s.operator_ref in repair.registered for p in reuse.candidates for s in p.steps if s.action == "reuse")


def test_evaluation_accepts_expected_failures_but_requires_repaired_operator_reuse():
    manifest = json.loads((DEFAULT_EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    rows = [dict(measurement(job, read_state(DEFAULT_EVIDENCE, job["name"])), returncode=0) for job in manifest["jobs"]]
    result = evaluation_outcome(rows)
    assert result["repair_reuse_chains"] == 3 and not result["required_evidence_missing"]
    assert result["worker_errors"] == result["execution_errors"] == 0
    assert any(row["status"] == "failed" for row in rows), "真实指标失败不是执行错误"
    without_reuse = [row for row in rows if row["group"] != "reuse"]
    assert evaluation_outcome(without_reuse)["required_evidence_missing"]
    failed_worker = [{**row, "returncode": 1} if row["name"] == rows[0]["name"] else row for row in rows]
    assert evaluation_outcome(failed_worker)["worker_errors"] == 1


@pytest.mark.slow
def test_real_evaluation_replay_runs_verifier_and_persistence(tmp_path, monkeypatch):
    # 子进程继承空密钥和不可用端点；dotenv 不覆盖已设置的环境变量。
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")
    result = run_evaluation("replay", tmp_path / "replay")
    assert result["worker_errors"] == result["execution_errors"] == 0
    assert not result["required_evidence_missing"] and result["repair_reuse_chains"] == 3
    manifest = json.loads((DEFAULT_EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    assert result["jobs"] == len(manifest["jobs"])
    for job in manifest["jobs"]:
        original = read_state(DEFAULT_EVIDENCE, job["name"])
        actual = read_state(tmp_path / "replay", job["name"])
        assert actual.status == original.status, actual.log
        assert [(r.plan_name, r.passed, r.metrics, [(d.code, d.operator) for d in r.diagnostics]) for r in actual.reports] == [
            (r.plan_name, r.passed, r.metrics, [(d.code, d.operator) for d in r.diagnostics]) for r in original.reports]
        assert [(r.operator, r.fixed, r.before, r.after) for r in actual.repairs] == [
            (r.operator, r.fixed, r.before, r.after) for r in original.repairs]
        assert actual.registered == original.registered
        assert actual.lessons_learned == original.lessons_learned
        assert all(c["source"] == "replay" for c in actual.llm_calls)
        assert (tmp_path / "replay" / job["name"] / "runs" / job["name"] / "report.html").exists()
        if job["group"] == "search":
            assert actual.search, actual.log
            file = tmp_path / "replay" / job["name"] / "intervention.json"
            assert json.loads(file.read_text()) == json.loads((DEFAULT_EVIDENCE / job["name"] / "intervention.json").read_text())
