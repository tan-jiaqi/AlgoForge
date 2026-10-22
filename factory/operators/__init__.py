"""算子库。导入本包即注册全部内置算子。"""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from factory.operators import models, tabular, text  # noqa: F401  （导入即注册）
from factory.operators.base import Operator, OperatorMeta, OperatorRegistry, RegistryError, register, registry
from factory.schemas import PipelinePlan


def build_pipeline(plan: PipelinePlan, reg: OperatorRegistry = registry) -> Pipeline:
    """把方案组装为 sklearn Pipeline，要求方案中的每一步都指向已注册的算子。"""
    steps = []
    for i, step in enumerate(plan.steps):
        ref = step.operator_ref or (step.synth_spec.name if step.synth_spec else None)
        steps.append((f"{i}_{step.kind.value}", reg.create(ref, **step.params)))
    return Pipeline(steps)


__all__ = [
    "Operator",
    "OperatorMeta",
    "OperatorRegistry",
    "RegistryError",
    "build_pipeline",
    "register",
    "registry",
]
