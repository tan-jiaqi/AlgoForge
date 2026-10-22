"""算子抽象与注册表。

*算子*（Operator）是能力工厂中的能力单元：兼容 sklearn 的估计器/变换器，
同时声明元数据（类型、输入输出、适用条件、依赖、版本）。由于算子遵循 sklearn 接口，
方案可以直接组装成 ``sklearn.pipeline.Pipeline``。

注册表以 ``"名称@版本"`` 为键映射到算子类。内置、抽取和生成的算子都放在同一个注册表里，
Planner 在决定复用还是生成时对它们一视同仁。
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import pandas as pd
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator

from factory.schemas import Modality, OperatorKind


class OperatorMeta(BaseModel):
    """算子的声明式描述，在知识图谱中对应一个 Operator 节点。"""

    name: str
    kind: OperatorKind
    version: int = Field(default=1, ge=1)
    modalities: list[Modality] = Field(default_factory=lambda: [Modality.TABULAR])
    description: str
    inputs: str = "pandas.DataFrame"
    outputs: str = "pandas.DataFrame"
    conditions: list[str] = Field(default_factory=list, description="算子适用的条件")
    dependencies: list[str] = Field(default_factory=list, description="所需的 Python 包")
    params: dict[str, Any] = Field(default_factory=dict, description="默认超参数")
    origin: Literal["builtin", "extracted", "synthesized"] = "builtin"
    parent: str | None = Field(default=None, description="本版本派生自的注册表键")

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"


class Operator(BaseEstimator):
    """所有算子的基类。子类需设置 ``meta`` 并实现 sklearn 接口。"""

    meta: ClassVar[OperatorMeta]


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """返回 DataFrame 的（数值列, 类别列）列名。"""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


class RegistryError(KeyError):
    pass


class OperatorRegistry:
    """以 ``名称@版本`` 为键的内存算子目录。"""

    def __init__(self) -> None:
        self._ops: dict[str, type[Operator]] = {}

    def register(self, cls: type[Operator]) -> type[Operator]:
        meta = getattr(cls, "meta", None)
        if not isinstance(meta, OperatorMeta):
            raise TypeError(f"{cls.__name__} must define a class attribute 'meta: OperatorMeta'")
        if meta.key in self._ops and self._ops[meta.key] is not cls:
            raise RegistryError(f"operator {meta.key} is already registered")
        self._ops[meta.key] = cls
        return cls

    def get(self, ref: str) -> type[Operator]:
        """解析 ``名称@版本``；只给 ``名称`` 时取最新版本。"""
        if "@" not in ref:
            ref = f"{ref}@{self.latest_version(ref)}"
        try:
            return self._ops[ref]
        except KeyError:
            raise RegistryError(f"unknown operator {ref!r}") from None

    def create(self, ref: str, **params: Any) -> Operator:
        cls = self.get(ref)
        return cls(**{**cls.meta.params, **params})

    def latest_version(self, name: str) -> int:
        versions = [cls.meta.version for cls in self._ops.values() if cls.meta.name == name]
        if not versions:
            raise RegistryError(f"unknown operator {name!r}")
        return max(versions)

    def list(self, kind: OperatorKind | None = None, modality: Modality | None = None) -> list[OperatorMeta]:
        metas = [cls.meta for cls in self._ops.values()]
        if kind is not None:
            metas = [m for m in metas if m.kind == kind]
        if modality is not None:
            metas = [m for m in metas if modality in m.modalities]
        return sorted(metas, key=lambda m: m.key)

    def __contains__(self, ref: str) -> bool:
        try:
            self.get(ref)
        except RegistryError:
            return False
        return True

    def __len__(self) -> int:
        return len(self._ops)


#: 进程级默认注册表，内置算子在导入时自动注册到这里。
registry = OperatorRegistry()


def register(cls: type[Operator]) -> type[Operator]:
    """类装饰器：把算子注册到默认注册表。"""
    return registry.register(cls)
