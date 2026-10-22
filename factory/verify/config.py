"""按任务类型加载验证配置（configs/tasks/<task_type>.yaml）。"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from factory.schemas import TaskType
from factory.utils import CONFIG_DIR


class VerifyConfig(BaseModel):
    cv_folds: int = Field(default=5, ge=2)
    seeds: list[int] = Field(default_factory=lambda: [42])
    smoke_rows: int = 300
    timeout_seconds: float = 180
    memory_mb: int = 8192
    max_metric_std: float = 0.03
    leakage_ceiling: float = 0.995
    report_metrics: list[str] = Field(default_factory=list)
    n_threads: int = 4


def load_verify_config(task_type: TaskType, path: Path | None = None, **overrides) -> VerifyConfig:
    path = path or CONFIG_DIR / "tasks" / f"{task_type.value}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    return VerifyConfig(**{**data, **overrides})
