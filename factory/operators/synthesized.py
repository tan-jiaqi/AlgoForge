"""生成算子的加载与持久化。

验证通过的生成算子保存为 ``<kg 目录>/operators/<Name>_v<version>.py``，并注册到 OperatorRegistry，
下次运行时直接复用（action=reuse），不再调用 LLM 生成。

加载前都会重新做一遍静态白名单检查：父进程只执行已通过四级验证、且通过静态检查的源码。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from factory.operators.base import Operator, OperatorRegistry, registry
from factory.utils import get_logger

log = get_logger(__name__)


def exec_operator_source(source: str, reg: OperatorRegistry = registry) -> list[type[Operator]]:
    """执行源码并注册其中的 Operator 子类，返回新注册的类（已注册的同名同版本跳过）。"""
    ns: dict[str, Any] = {"__name__": "factory_generated"}
    exec(compile(source, "<generated>", "exec"), ns)  # noqa: S102 - 调用方保证源码已通过静态检查
    added = []
    for obj in list(ns.values()):
        if isinstance(obj, type) and issubclass(obj, Operator) and obj is not Operator and hasattr(obj, "meta"):
            if obj.meta.key in reg:
                continue
            reg.register(obj)
            reg.sources[obj.meta.key] = source
            added.append(obj)
    return added


def operator_dir(kg_path: Path) -> Path:
    return Path(kg_path).parent / "operators"


def save_operator(source: str, key: str, kg_path: Path) -> Path:
    name, version = key.split("@")
    path = operator_dir(kg_path) / f"{name}_v{version}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source.rstrip() + "\n", encoding="utf-8")
    return path


def load_saved_operators(kg_path: Path, reg: OperatorRegistry = registry) -> list[str]:
    """启动时加载已沉淀的生成算子；未通过静态检查的文件跳过并告警。"""
    from factory.verify.static import check_source  # 延迟导入，避免循环依赖

    loaded = []
    for file in sorted(operator_dir(kg_path).glob("*.py")):
        source = file.read_text(encoding="utf-8")
        diags = check_source(source)
        if diags:
            log.warning("跳过未通过静态检查的算子文件 %s：%s", file.name, diags[0].message)
            continue
        loaded += [cls.meta.key for cls in exec_operator_source(source, reg)]
    return loaded
