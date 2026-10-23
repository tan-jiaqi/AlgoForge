"""跨平台沙箱：在 Linux 上模拟 Windows（没有 resource 模块、不支持 preexec_fn）的代码路径。"""

import importlib
import sys

import pytest

from factory.schemas import DiagnosticCode, GeneratedOperator, OperatorKind, PipelinePlan, Step, SynthSpec
from factory.verify import Verifier, sandbox
from tests.test_verify import _gen, _plan, _spec, _synth_step


def test_modules_import_without_resource(monkeypatch):
    """原生 Windows 没有 resource 模块：sandbox 与 checks 都必须能导入，并自动改用看门狗。"""
    monkeypatch.setitem(sys.modules, "resource", None)  # 让 import resource 抛出 ImportError
    try:
        mod = importlib.reload(sandbox)
        assert mod.USE_RLIMIT is False
        from factory.verify import checks

        assert checks.peak_rss_mb() > 0, "没有 resource 时应改用 psutil 统计内存"
    finally:
        monkeypatch.delitem(sys.modules, "resource")
        importlib.reload(sandbox)
    assert sandbox.USE_RLIMIT is (sys.platform != "win32")


@pytest.fixture
def watchdog_mode(monkeypatch):
    """强制使用 Windows 的内存看门狗（不传 preexec_fn）。"""
    monkeypatch.setattr(sandbox, "USE_RLIMIT", False)
    popen = sandbox.subprocess.Popen

    def no_preexec(*args, **kwargs):
        assert "preexec_fn" not in kwargs, "Windows 不支持 preexec_fn"
        return popen(*args, **kwargs)

    monkeypatch.setattr(sandbox.subprocess, "Popen", no_preexec)


def test_watchdog_good_plan_passes(watchdog_mode, churn_csv, fast_verify):
    r = Verifier(fast_verify).verify(_plan(), _spec(), str(churn_csv))
    assert r.passed, r.diagnostics
    assert r.resources["peak_rss_mb"] > 0


def test_watchdog_kills_memory_hog(watchdog_mode, churn_csv, fast_verify):
    gen = _gen("Hog", "        import numpy as np\n        self.big_ = np.ones(5 * 10**8)\n        return X.copy()")
    cfg = fast_verify.model_copy(update={"memory_mb": 1500, "timeout_seconds": 60})
    r = Verifier(cfg).verify(_plan(_synth_step(gen)), _spec(), str(churn_csv), {"Hog": gen})
    assert r.diagnostics[0].code == DiagnosticCode.TIMEOUT and "内存上限 1500 MB" in r.diagnostics[0].message


def test_watchdog_timeout_and_crash(watchdog_mode, churn_csv, fast_verify):
    slow = _gen("Slow", "        while True:\n            pass")
    cfg = fast_verify.model_copy(update={"timeout_seconds": 3})
    r = Verifier(cfg).verify(_plan(_synth_step(slow)), _spec(), str(churn_csv), {"Slow": slow})
    assert r.diagnostics[0].code == DiagnosticCode.TIMEOUT and "秒被终止" in r.diagnostics[0].message
    # 普通的运行错误不能被误判为内存超限
    bad = _gen("Bad", "        raise ValueError('boom')")
    r = Verifier(fast_verify).verify(_plan(_synth_step(bad)), _spec(), str(churn_csv), {"Bad": bad})
    assert r.diagnostics[0].code == DiagnosticCode.RUNTIME_ERROR


def test_child_crash_without_result_is_runtime_error(watchdog_mode, monkeypatch, fast_verify):
    """子进程本身崩溃（没有产出结果文件）时给出 E_RUNTIME，而不是内存或超时。"""
    monkeypatch.setattr(sandbox.sys, "executable", sys.executable)
    result, err = sandbox.run_job({"plan": {}}, timeout=30, memory_mb=2048, n_threads=1)
    assert result is None and err.code == DiagnosticCode.RUNTIME_ERROR
