import numpy as np
import pytest
from sklearn.base import clone

from factory.operators import Operator, OperatorMeta, OperatorRegistry, RegistryError, build_pipeline, registry
from factory.schemas import Modality, OperatorKind, PipelinePlan, Step


def _plan(*refs, kinds):
    return PipelinePlan(
        name="t", steps=[Step(kind=k, action="reuse", operator_ref=r) for r, k in zip(refs, kinds)]
    )


TABULAR_PREP = ("MissingValueImputer@1", "OneHotCategoricalEncoder@1", "StandardFeatureScaler@1")
PREP_KINDS = [OperatorKind.IMPUTER, OperatorKind.ENCODER, OperatorKind.SCALER, OperatorKind.MODEL]


def test_builtins_registered():
    assert len(registry) >= 7
    assert "LightGBMModel" in registry and "LightGBMModel@1" in registry
    assert {m.name for m in registry.list(modality=Modality.TEXT)} >= {"TfidfTextVectorizer", "LogisticRegressionModel"}


@pytest.mark.parametrize("model", ["LogisticRegressionModel@1", "RandomForestModel@1", "LightGBMModel@1"])
def test_tabular_pipeline_end_to_end(tabular_df, model):
    X, y = tabular_df.drop(columns="churn"), tabular_df["churn"]
    X_before = X.copy()
    pipe = build_pipeline(_plan(*TABULAR_PREP, model, kinds=PREP_KINDS))
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert X.equals(X_before), "operators must not mutate their input"


def test_unseen_category_is_handled(tabular_df):
    X, y = tabular_df.drop(columns="churn"), tabular_df["churn"]
    pipe = build_pipeline(_plan(*TABULAR_PREP, "LogisticRegressionModel@1", kinds=PREP_KINDS)).fit(X, y)
    X_new = X.head(3).copy()
    X_new["contract"] = "never_seen"
    assert pipe.predict(X_new).shape == (3,)


def test_text_pipeline(text_df):
    plan = _plan(
        "TfidfTextVectorizer@1",
        "LogisticRegressionModel@1",
        kinds=[OperatorKind.VECTORIZER, OperatorKind.MODEL],
    )
    pipe = build_pipeline(plan).fit(text_df[["text"]], text_df["label"])
    assert (pipe.predict(text_df[["text"]]) == text_df["label"]).mean() > 0.9


def test_operators_are_sklearn_clonable():
    for meta in registry.list():
        op = registry.create(meta.key)
        assert type(clone(op)) is type(op)


def test_registry_versions_and_errors():
    reg = OperatorRegistry()

    class Dummy(Operator):
        meta = OperatorMeta(name="Dummy", kind=OperatorKind.FEATURE, description="v1")

    class Dummy2(Operator):
        meta = OperatorMeta(name="Dummy", kind=OperatorKind.FEATURE, version=2, description="v2", parent="Dummy@1")

    reg.register(Dummy)
    reg.register(Dummy2)
    assert reg.get("Dummy") is Dummy2
    assert reg.get("Dummy@1") is Dummy
    with pytest.raises(RegistryError):
        reg.register(type("Clash", (Operator,), {"meta": Dummy.meta}))
    with pytest.raises(RegistryError):
        reg.get("Missing")
    with pytest.raises(TypeError):
        reg.register(type("NoMeta", (Operator,), {}))


def test_class_weight_keys_survive_json_roundtrip():
    """LLM 给出的 class_weight 经 JSON 后键变成字符串，模型仍应按真实类别使用。"""
    import json

    import numpy as np

    from factory.operators import registry

    X = np.random.RandomState(0).normal(size=(60, 3))
    y = np.array([0] * 45 + [1] * 15)
    params = json.loads(json.dumps({"class_weight": {0: 1, 1: 3}}))
    assert list(params["class_weight"]) == ["0", "1"]
    for ref in ("LogisticRegressionModel@1", "RandomForestModel@1", "LightGBMModel@1"):
        model = registry.create(ref, **params).fit(X, y)
        assert model.model_.class_weight == {0: 1, 1: 3}
        assert model.get_params()["class_weight"] == {"0": 1, "1": 3}, "不改动用户给的参数"
