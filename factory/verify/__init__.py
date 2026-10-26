"""四级验证器：静态检查 -> 沙箱执行 -> 契约检查 -> 指标检查。

Verifier 是确定性组件，不调用 LLM：任一级失败即停止，并给出带诊断码的 Diagnostic。
"""

from __future__ import annotations

from pathlib import Path

from factory.operators import registry
from factory.schemas import GeneratedOperator, PipelinePlan, TaskSpec, ValidationReport
from factory.verify.config import VerifyConfig, load_verify_config
from factory.verify.sandbox import run_job
from factory.verify.static import check_source


class Verifier:
    def __init__(self, config: VerifyConfig | None = None):
        self.config = config

    def verify(
        self,
        plan: PipelinePlan,
        spec: TaskSpec,
        data_path: str,
        generated: dict[str, GeneratedOperator] | None = None,
    ) -> ValidationReport:
        cfg = self.config or load_verify_config(spec.task_type)
        generated = generated or {}
        # 第一级：静态检查（只针对生成的算子；内置算子已经过测试）
        for gen in generated.values():
            diags = check_source(gen.source, class_name=gen.spec.name, kind=gen.spec.kind)
            if diags:
                return ValidationReport(plan_name=plan.name, passed=False, diagnostics=diags)
        # 复用的生成算子（来自以前的运行）也要把源码带进子进程
        sources = [g.source for g in generated.values()]
        owners = [g.key for g in generated.values()]  # 与 sources 一一对应，加载失败时据此定位算子
        for step in plan.steps:
            src = registry.source_of(step.operator_ref) if step.action == "reuse" else None
            if src and src not in sources:
                sources.append(src)
                owners.append(step.operator_ref)
        # 第二到四级：沙箱子进程内执行
        job = {
            "plan": plan.model_dump(mode="json"),
            "spec": spec.model_dump(mode="json"),
            "config": cfg.model_dump(),
            "data_path": str(Path(data_path).resolve()),  # 子进程工作目录是临时目录
            "sources": sources,
            "source_owners": owners,
        }
        result, error = run_job(job, timeout=cfg.timeout_seconds, memory_mb=cfg.memory_mb, n_threads=cfg.n_threads)
        if error is not None:
            return ValidationReport(plan_name=plan.name, passed=False, diagnostics=[error])
        report = ValidationReport(plan_name=plan.name, passed=not result["diagnostics"], **result)
        return report


__all__ = ["Verifier", "VerifyConfig", "check_source", "load_verify_config"]
