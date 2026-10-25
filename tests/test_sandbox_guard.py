"""沙箱运行时防线（审计钩子）：即使越权代码绕过了静态检查，在子进程里也会被拦截并判为 E_UNSAFE。

这些用例故意跳过静态检查，直接测运行时这一层。
"""

import pytest

from factory import verify as verify_pkg
from factory.schemas import DiagnosticCode
from factory.utils import ROOT
from factory.verify import Verifier
from tests.test_verify import _gen, _plan, _spec, _synth_step

ENV_FILE = str(ROOT / ".env")


@pytest.fixture
def no_static(monkeypatch):
    monkeypatch.setattr(verify_pkg, "check_source", lambda *a, **kw: [])


def _run(name, body, csv, cfg):
    gen = _gen(name, body)
    return Verifier(cfg).verify(_plan(_synth_step(gen)), _spec(0.1), str(csv), {gen.spec.name: gen})


@pytest.mark.parametrize("name, body, expect", [
    # 读仓库里的 .env，并吞掉异常假装无事发生
    ("ReadEnv", f"        try:\n            pd.read_csv({ENV_FILE!r}, header=None)\n        except Exception:\n"
                "            pass\n        return X.copy()", ".env"),
    # 写到沙箱临时目录之外
    ("WriteOut", f"        X.to_csv({str(ROOT / 'leak.csv')!r})\n        return X.copy()", "写入"),
    # 网络
    ("Net", "        import socket\n        try:\n            socket.create_connection(('127.0.0.1', 9), timeout=1)\n"
            "        except Exception:\n            pass\n        return X.copy()", "socket"),
    # 子进程
    ("Proc", "        import subprocess\n        subprocess.run(['echo', 'hi'])\n        return X.copy()", "subprocess"),
])
def test_runtime_violation_is_unsafe(no_static, churn_csv, fast_verify, name, body, expect):
    report = _run(name, body, churn_csv, fast_verify)
    assert not report.passed
    diag = report.diagnostics[0]
    assert diag.code == DiagnosticCode.UNSAFE_CODE and diag.stage == "sandbox"
    assert diag.operator == name, "应定位到越权的生成算子，修复环节才能只修它"
    assert expect in diag.message
    assert not (ROOT / "leak.csv").exists()


def test_module_level_violation(no_static, churn_csv, fast_verify):
    """模块级代码（类定义之外）在加载时就越权，同样被拦截。"""
    gen = _gen("Ok", "        return X.copy()")
    gen = gen.model_copy(update={"source": gen.source + f"\n_ = pd.read_csv({ENV_FILE!r}, header=None)\n"})
    report = Verifier(fast_verify).verify(_plan(_synth_step(gen)), _spec(0.1), str(churn_csv), {"Ok": gen})
    assert not report.passed and report.diagnostics[0].code == DiagnosticCode.UNSAFE_CODE


def test_normal_operator_still_passes(churn_csv, fast_verify):
    """正常算子（含 LightGBM、随机森林等会探测 CPU、调用动态库的模型）不受影响。"""
    gen = _gen("Ok", "        return X.copy()")
    for model in ("LogisticRegressionModel@1", "RandomForestModel@1", "LightGBMModel@1"):
        report = Verifier(fast_verify).verify(_plan(_synth_step(gen), model=model), _spec(0.1), str(churn_csv), {"Ok": gen})
        assert report.passed, (model, report.diagnostics)
