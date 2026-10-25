"""验证器：好方案通过；各类坏代码/坏方案得到对应诊断码。"""

import pytest

from factory.schemas import (
    DiagnosticCode, GeneratedOperator, MetricTarget, Modality, OperatorKind, PipelinePlan, Step, SynthSpec, TaskSpec, TaskType,
)
from factory.verify import Verifier, check_source
from factory.verify.sandbox import child_env

GOOD_SRC = '''
import pandas as pd
from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind
from sklearn.base import TransformerMixin

class {name}(TransformerMixin, Operator):
    meta = OperatorMeta(name="{name}", kind=OperatorKind.FEATURE, description="test", origin="synthesized")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
{body}
'''


def _gen(name: str, body: str) -> GeneratedOperator:
    spec = SynthSpec(name=name, kind=OperatorKind.FEATURE, purpose="t", inputs="df", outputs="df")
    return GeneratedOperator(spec=spec, source=GOOD_SRC.format(name=name, body=body))


def _spec(threshold=0.5):
    return TaskSpec(description="t", task_type=TaskType.BINARY_CLASSIFICATION, modality=Modality.TABULAR,
                    target_column="churn", metrics=[MetricTarget(name="roc_auc", threshold=threshold)])


def _plan(*extra: Step, model="LogisticRegressionModel@1") -> PipelinePlan:
    R = lambda k, r: Step(kind=k, action="reuse", operator_ref=r)
    steps = [R(OperatorKind.IMPUTER, "MissingValueImputer@1"), *extra,
             R(OperatorKind.ENCODER, "OneHotCategoricalEncoder@1"), R(OperatorKind.SCALER, "StandardFeatureScaler@1"),
             R(OperatorKind.MODEL, model)]
    return PipelinePlan(name="p", steps=steps)


def _synth_step(gen: GeneratedOperator) -> Step:
    return Step(kind=OperatorKind.FEATURE, action="synthesize", synth_spec=gen.spec)


# ---- 静态检查 -------------------------------------------------------------

@pytest.mark.parametrize("snippet, code", [
    ("import os", DiagnosticCode.UNSAFE_CODE),
    ("import subprocess", DiagnosticCode.UNSAFE_CODE),
    ("from socket import socket", DiagnosticCode.UNSAFE_CODE),
    ("x = eval('1')", DiagnosticCode.UNSAFE_CODE),
    ("x = (1).__class__.__bases__", DiagnosticCode.UNSAFE_CODE),
    ("def broken(:", DiagnosticCode.SYNTAX_ERROR),
    # 库自带的文件读写 / 下载 / 字符串求值入口
    ("x = pd.read_csv('/repo/.env')", DiagnosticCode.UNSAFE_CODE),
    ("import numpy as np\nx = np.load('a.npy', allow_pickle=True)", DiagnosticCode.UNSAFE_CODE),
    ("pd.DataFrame().to_pickle('a.pkl')", DiagnosticCode.UNSAFE_CODE),
    ("from pandas import read_pickle", DiagnosticCode.UNSAFE_CODE),
    ("from pandas import *", DiagnosticCode.UNSAFE_CODE),
    ("from sklearn.datasets import fetch_openml", DiagnosticCode.UNSAFE_CODE),
    ("import scipy.io", DiagnosticCode.UNSAFE_CODE),
    ("x = pd.DataFrame({'a': [1]}).query('a > 0')", DiagnosticCode.UNSAFE_CODE),
    ("x = pd.io", DiagnosticCode.UNSAFE_CODE),
    # 绕过名字检查的写法
    ("f = open", DiagnosticCode.UNSAFE_CODE),
    ("x = __builtins__", DiagnosticCode.UNSAFE_CODE),
    ("x = '{0.__class__}'.format(1)", DiagnosticCode.UNSAFE_CODE),
    ("import factory.operators.synthesized", DiagnosticCode.UNSAFE_CODE),
    ("from factory.operators.synthesized import exec_operator_source", DiagnosticCode.UNSAFE_CODE),
    ("from factory.utils import ROOT", DiagnosticCode.UNSAFE_CODE),
    ("import lightgbm as lgb\nb = lgb.Booster(model_file='m.txt')", DiagnosticCode.UNSAFE_CODE),
])
def test_static_rejects(snippet, code):
    src = _gen("Ok", "        return X.copy()").source + "\n" + snippet
    diags = check_source(src, class_name="Ok", kind=OperatorKind.FEATURE)
    assert diags and diags[0].code == code


def test_static_interface():
    gen = _gen("Ok", "        return X.copy()")
    assert check_source(gen.source, class_name="Ok", kind=OperatorKind.FEATURE) == []
    no_transform = gen.source.replace("def transform", "def other")
    assert check_source(no_transform, class_name="Ok")[0].code == DiagnosticCode.BAD_INTERFACE
    assert check_source(gen.source, class_name="Missing")[0].code == DiagnosticCode.BAD_INTERFACE


def test_child_env_drops_secrets(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("SOME_TOKEN", "x")
    env = child_env(1)
    assert "LLM_API_KEY" not in env and "SOME_TOKEN" not in env
    assert env["OMP_NUM_THREADS"] == "1"


# ---- 沙箱 + 契约 + 指标 ------------------------------------------------------

def test_good_plan_passes(churn_csv, fast_verify):
    r = Verifier(fast_verify).verify(_plan(), _spec(), str(churn_csv))
    assert r.passed, r.diagnostics
    assert {"roc_auc"} <= set(r.metrics) and "roc_auc" in r.metric_std
    assert r.resources["fit_seconds"] > 0 and r.resources["peak_rss_mb"] > 0


def test_generated_operator_passes(churn_csv, fast_verify):
    gen = _gen("AddRatio", "        X = X.copy()\n        X['ratio'] = X['tenure'] / 10\n        return X")
    r = Verifier(fast_verify).verify(_plan(_synth_step(gen)), _spec(), str(churn_csv), {gen.spec.name: gen})
    assert r.passed, r.diagnostics


@pytest.mark.parametrize("body, code", [
    ("        raise ValueError('boom')", DiagnosticCode.RUNTIME_ERROR),
    ("        X['tenure'] = 0\n        return X", DiagnosticCode.INPUT_MUTATED),
    ("        return X.iloc[:5]", DiagnosticCode.BAD_SHAPE),
    ("        import os\n        return X", DiagnosticCode.UNSAFE_CODE),
])
def test_bad_operators_get_codes(churn_csv, fast_verify, body, code):
    gen = _gen("Bad", body)
    r = Verifier(fast_verify).verify(_plan(_synth_step(gen)), _spec(), str(churn_csv), {"Bad": gen})
    assert not r.passed
    assert r.diagnostics[0].code == code
    if code != DiagnosticCode.UNSAFE_CODE:
        assert r.diagnostics[0].operator == "Bad@1", "错误应定位到出错的算子"


def test_timeout(churn_csv, fast_verify):
    gen = _gen("Slow", "        while True:\n            pass")
    cfg = fast_verify.model_copy(update={"timeout_seconds": 3})
    r = Verifier(cfg).verify(_plan(_synth_step(gen)), _spec(), str(churn_csv), {"Slow": gen})
    assert r.diagnostics[0].code == DiagnosticCode.TIMEOUT


def test_metric_threshold(churn_csv, fast_verify):
    r = Verifier(fast_verify).verify(_plan(), _spec(threshold=0.9999), str(churn_csv))
    codes = {d.code for d in r.diagnostics}
    assert not r.passed and DiagnosticCode.METRIC_BELOW_THRESHOLD in codes
    assert all(d.code.category == "metric" for d in r.diagnostics)


def test_leakage_flagged(churn_csv, fast_verify, tmp_path):
    import pandas as pd

    df = pd.read_csv(churn_csv)
    df["leak"] = df["churn"]
    leaky = tmp_path / "leaky.csv"
    df.to_csv(leaky, index=False)
    r = Verifier(fast_verify).verify(_plan(), _spec(), str(leaky))
    assert DiagnosticCode.UNSTABLE in {d.code for d in r.diagnostics}
