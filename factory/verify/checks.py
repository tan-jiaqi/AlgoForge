"""第二到四级检查：在沙箱子进程内执行（由 ``python -m factory.verify`` 调用）。

- 沙箱冒烟：小样本上逐步 fit/transform，出错时定位到具体算子；
- 契约检查：不修改输入、行数一致、概率合法、能处理缺失值与未见类别；
- 指标检查：多种子 × 分层 K 折交叉验证，判断阈值、稳定性与疑似泄漏，并记录耗时和内存。

指标以插件方式注册：用 ``@metric("name")`` 装饰一个函数即可新增评估指标。
"""

from __future__ import annotations

import sys
import time
import traceback
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn import metrics as skm
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split

from factory.operators import build_pipeline
from factory.operators.synthesized import exec_operator_source
from factory.profile import load_table
from factory.schemas import Diagnostic, DiagnosticCode, OperatorKind, PipelinePlan, TaskSpec, canonical_metric
from factory.verify import guard
from factory.verify.config import VerifyConfig

# ---------------------------------------------------------------------------
# 指标注册表
# ---------------------------------------------------------------------------

MetricFn = Callable[[np.ndarray, np.ndarray, np.ndarray, list, Any], float]
METRICS: dict[str, MetricFn] = {}


def metric(name: str):
    def deco(fn: MetricFn) -> MetricFn:
        METRICS[name] = fn
        return fn

    return deco


@metric("accuracy")
def _accuracy(y, pred, proba, classes, pos):
    return skm.accuracy_score(y, pred)


for _name, _fn in (("precision", skm.precision_score), ("recall", skm.recall_score), ("f1", skm.f1_score)):
    METRICS[_name] = lambda y, pred, proba, classes, pos, f=_fn: f(y, pred, pos_label=pos, zero_division=0)
    METRICS[f"{_name}_macro"] = lambda y, pred, proba, classes, pos, f=_fn: f(y, pred, average="macro", zero_division=0)


@metric("roc_auc")
def _roc_auc(y, pred, proba, classes, pos):
    if len(classes) == 2:
        return skm.roc_auc_score(y == pos, proba[:, classes.index(pos)])
    return skm.roc_auc_score(y, proba, multi_class="ovr", labels=classes)


@metric("recall_at_20")
def _recall_at_20(y, pred, proba, classes, pos):
    """业务指标：按正类概率排序，前 20% 样本覆盖的正类比例。"""
    score = proba[:, classes.index(pos)]
    top = np.argsort(-score)[: max(1, int(len(score) * 0.2))]
    return float((y[top] == pos).sum() / max((y == pos).sum(), 1))


def resolve_metric(name: str, n_classes: int) -> str:
    """多分类时把 recall/precision/f1 自动映射为 macro 版本。"""
    name = canonical_metric(name)
    if n_classes > 2 and name in ("precision", "recall", "f1"):
        return f"{name}_macro"
    return name


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def run_checks(job: dict) -> dict:
    """执行冒烟、契约与指标检查，返回可 JSON 序列化的结果。"""
    plan = PipelinePlan.model_validate(job["plan"])
    spec = TaskSpec.model_validate(job["spec"])
    cfg = VerifyConfig.model_validate(job["config"])
    result: dict[str, Any] = {"diagnostics": [], "metrics": {}, "metric_std": {}, "resources": {}, "notes": []}
    owners = job.get("source_owners") or [None] * len(job.get("sources", []))
    owner = None
    try:
        for src, owner in zip(job.get("sources", []), owners):
            exec_operator_source(src)
    except Exception as exc:  # noqa: BLE001 - 模块级代码出错（含被沙箱拦截）
        # 标明出错的算子，工作流才能把它交给定向修复（如类继承顺序导致的 MRO 冲突）
        diags = [_diag(DiagnosticCode.RUNTIME_ERROR, "sandbox", "加载生成算子失败：" + _short_exc(exc), owner)]
    else:
        df = load_table(job["data_path"])
        y = df[spec.target_column].to_numpy()
        X = df.drop(columns=[spec.target_column])
        classes = sorted(pd.unique(y).tolist(), key=str)
        counts = pd.Series(y).value_counts()
        pos = counts.idxmin() if len(classes) == 2 else None  # 二分类以少数类为正类（如 churn=1）
        diags = smoke_and_contract(plan, X, y, classes, cfg)
        if not diags:
            diags = evaluate(plan, spec, X, y, classes, pos, cfg, result)
    if guard.VIOLATIONS:  # 生成代码越权：无论它是否吞掉了异常，都判为不安全
        diags = [_violation_diag(guard.VIOLATIONS)]
        result["metrics"], result["metric_std"] = {}, {}
    result["diagnostics"] = [d.model_dump(mode="json") for d in diags]
    result["resources"]["peak_rss_mb"] = round(peak_rss_mb(), 1)
    return result


def _violation_diag(violations: list[dict]) -> Diagnostic:
    first = violations[0]
    more = f"（共 {len(violations)} 次）" if len(violations) > 1 else ""
    return _diag(DiagnosticCode.UNSAFE_CODE, "sandbox", f"运行时越权，已被沙箱拦截：{first['detail']}{more}",
                 first["operator"], "算子只能处理传入的数据：不要读写文件、访问网络或启动进程")


def peak_rss_mb() -> float:
    """当前进程的峰值常驻内存（MB）。POSIX 用 resource，Windows 用 psutil 的 peak_wset。"""
    try:
        import resource
    except ImportError:  # Windows
        import psutil

        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", info.rss) / 1024 / 1024
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024 / 1024 if sys.platform == "darwin" else usage / 1024  # macOS 单位为字节，Linux 为 KB


# ---------------------------------------------------------------------------
# 冒烟 + 契约
# ---------------------------------------------------------------------------

def _diag(code, stage, message, operator=None, hint="") -> Diagnostic:
    return Diagnostic(code=code, stage=stage, message=message, operator=operator, hint=hint)


def _short_exc(exc: BaseException) -> str:
    frame = traceback.extract_tb(exc.__traceback__)[-1] if exc.__traceback__ else None
    where = f" (at {frame.name}:{frame.lineno})" if frame else ""
    return f"{type(exc).__name__}: {str(exc)[:300]}{where}"


def _sample(X: pd.DataFrame, y: np.ndarray, n: int):
    n = min(n, len(X))
    strat = y if pd.Series(y).value_counts().min() >= 2 else None
    return train_test_split(X, y, train_size=int(n * 0.7), test_size=n - int(n * 0.7), stratify=strat, random_state=0)


def _perturb(X: pd.DataFrame) -> pd.DataFrame:
    """构造健壮性样本：类别列填入未见取值，数值列置为缺失。"""
    Xp = X.copy()
    for i, c in enumerate(Xp.columns):
        rows = Xp.index[i % 2 :: 2]
        if pd.api.types.is_numeric_dtype(Xp[c]):
            Xp.loc[rows, c] = np.nan
        else:
            Xp.loc[rows, c] = "__unseen__"
    return Xp


def smoke_and_contract(plan: PipelinePlan, X, y, classes, cfg: VerifyConfig) -> list[Diagnostic]:
    Xtr, Xte, ytr, yte = _sample(X, y, cfg.smoke_rows)
    try:
        steps = build_pipeline(plan).steps
    except Exception as exc:  # noqa: BLE001 - 未知算子、非法参数等
        return [_diag(DiagnosticCode.BAD_INTERFACE, "sandbox", "组装 Pipeline 失败：" + _short_exc(exc))]
    cur_tr, cur_te = Xtr, Xte
    for _, op in steps[:-1]:
        opname = type(op).meta.key
        before_tr = cur_tr.copy() if isinstance(cur_tr, pd.DataFrame) else None
        try:
            out_tr = op.fit(cur_tr, ytr).transform(cur_tr)
            out_te = op.transform(cur_te)
        except Exception as exc:  # noqa: BLE001
            return [_diag(DiagnosticCode.RUNTIME_ERROR, "sandbox", _short_exc(exc), opname)]
        if before_tr is not None and not cur_tr.equals(before_tr):
            return [_diag(DiagnosticCode.INPUT_MUTATED, "contract", "transform 原地修改了输入 DataFrame", opname,
                          "在函数开头使用 X = X.copy()")]
        if out_tr.shape[0] != cur_tr.shape[0] or out_te.shape[0] != cur_te.shape[0]:
            return [_diag(DiagnosticCode.BAD_SHAPE, "contract",
                          f"输出行数 {out_tr.shape[0]} 与输入 {cur_tr.shape[0]} 不一致", opname)]
        if _bad_values(out_tr) > _bad_values(cur_tr):
            return [_diag(DiagnosticCode.NAN_OUTPUT, "contract", "输出引入了新的 NaN/Inf", opname,
                          "本算子产生的缺失值或无穷值需要在本算子内处理（如 fillna）")]
        cur_tr, cur_te = out_tr, out_te

    model_name = type(steps[-1][1]).meta.key
    try:
        model = steps[-1][1].fit(cur_tr, ytr)
        raw_proba, raw_pred = model.predict_proba(cur_te), model.predict(cur_te)
    except Exception as exc:  # noqa: BLE001
        return [_diag(DiagnosticCode.RUNTIME_ERROR, "sandbox", _short_exc(exc), model_name)]
    if not isinstance(raw_proba, np.ndarray) or not isinstance(raw_pred, np.ndarray) or raw_pred.ndim != 1:
        return [_diag(DiagnosticCode.BAD_SHAPE, "contract",
                      f"predict_proba / predict 应返回 numpy 数组（二维 / 一维），实际为 "
                      f"{type(raw_proba).__name__} / {type(raw_pred).__name__}", model_name,
                      "sklearn 约定：predict_proba 返回 shape (n, n_classes) 的 ndarray，predict 返回 shape (n,) 的 ndarray")]
    proba, pred = raw_proba, raw_pred
    if proba.shape != (len(Xte), len(model.classes_)) or pred.shape[0] != len(Xte):
        return [_diag(DiagnosticCode.BAD_SHAPE, "contract", f"predict_proba 形状 {proba.shape} 不符合预期", model_name)]
    if not np.isfinite(proba).all():
        return [_diag(DiagnosticCode.NAN_OUTPUT, "contract", "预测概率中出现 NaN/Inf", model_name)]
    if (proba < -1e-6).any() or (proba > 1 + 1e-6).any() or not np.allclose(proba.sum(axis=1), 1, atol=1e-4):
        return [_diag(DiagnosticCode.BAD_PROBA, "contract", "预测概率不在 [0,1] 或行和不为 1", model_name)]

    # 健壮性：未见类别与缺失值
    cur, culprit = _perturb(Xte), model_name
    try:
        for _, op in steps[:-1]:
            culprit = type(op).meta.key
            cur = op.transform(cur)
        culprit = model_name
        p2 = np.asarray(model.predict_proba(cur))
        if not np.isfinite(p2).all():
            raise ValueError("预测概率出现 NaN")
    except Exception as exc:  # noqa: BLE001
        return [_diag(DiagnosticCode.RUNTIME_ERROR, "contract", "未见类别/缺失值健壮性检查失败：" + _short_exc(exc),
                      culprit, "需能处理预测时出现的未见类别和缺失值")]
    return []


def _bad_values(X) -> int:
    """统计数值部分的 NaN/Inf 个数（稀疏矩阵只看非零元素）。"""
    if hasattr(X, "select_dtypes"):
        arr = X.select_dtypes(include="number").to_numpy(dtype=float)
    elif hasattr(X, "data") and hasattr(X, "nnz"):
        arr = X.data
    else:
        arr = np.asarray(X, dtype=float)
    return int((~np.isfinite(arr)).sum())


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------

def _with_seed(plan: PipelinePlan, seed: int):
    pipe = build_pipeline(plan)
    for _, op in pipe.steps:
        if "random_state" in op.get_params(deep=False):
            op.set_params(random_state=seed)
    return pipe


def evaluate(plan, spec: TaskSpec, X, y, classes, pos, cfg: VerifyConfig, result: dict) -> list[Diagnostic]:
    n_classes = len(classes)
    wanted = [resolve_metric(m.name, n_classes) for m in spec.metrics]
    names = list(dict.fromkeys(wanted + [resolve_metric(m, n_classes) for m in cfg.report_metrics]))
    unknown = [m for m in names if m not in METRICS]
    if unknown:
        result["notes"].append(f"未注册的指标已跳过：{unknown}")
    names = [m for m in names if m in METRICS]

    per_seed: dict[str, list[float]] = {m: [] for m in names}
    fit_s, pred_s, pred_rows = 0.0, 0.0, 0
    last = plan.steps[-1]
    model_name = last.operator_ref if last.action == "reuse" else build_pipeline(plan).steps[-1][1].meta.key
    try:
        for seed in cfg.seeds:
            fold_vals: dict[str, list[float]] = {m: [] for m in names}
            for tr, te in StratifiedKFold(cfg.cv_folds, shuffle=True, random_state=seed).split(X, y):
                pipe = clone(_with_seed(plan, seed))
                t0 = time.perf_counter()
                pipe.fit(X.iloc[tr], y[tr])
                t1 = time.perf_counter()
                proba = pipe.predict_proba(X.iloc[te])
                pred = pipe.predict(X.iloc[te])  # 使用模型自身的决策规则（可能做了阈值调整）
                pred_s += time.perf_counter() - t1
                fit_s += t1 - t0
                pred_rows += len(te)
                order = [list(pipe.classes_).index(c) for c in classes]
                for m in names:
                    fold_vals[m].append(METRICS[m](y[te], pred, proba[:, order], classes, pos))
            for m in names:
                per_seed[m].append(float(np.mean(fold_vals[m])))
    except Exception as exc:  # noqa: BLE001
        return [_diag(DiagnosticCode.RUNTIME_ERROR, "sandbox", "交叉验证失败：" + _short_exc(exc), model_name)]

    result["metrics"] = {m: round(float(np.mean(v)), 4) for m, v in per_seed.items()}
    result["metric_std"] = {m: round(float(np.std(v)), 4) for m, v in per_seed.items()}
    n_fits = len(cfg.seeds) * cfg.cv_folds
    result["resources"] = {
        "fit_seconds": round(fit_s / n_fits, 3),
        "predict_ms_per_1k": round(pred_s / max(pred_rows, 1) * 1e6, 2),
    }

    diags = []
    for target, name in zip(spec.metrics, wanted):
        value = result["metrics"].get(name)
        if value is not None and value < target.threshold:
            diags.append(_diag(DiagnosticCode.METRIC_BELOW_THRESHOLD, "metric",
                               f"{name} = {value:.4f} 低于阈值 {target.threshold}", model_name,
                               "考虑 class_weight、换模型、调参或调整决策阈值"))
    primary = resolve_metric(spec.primary_metric.name, n_classes)
    if result["metric_std"].get(primary, 0) > cfg.max_metric_std:
        diags.append(_diag(DiagnosticCode.UNSTABLE, "metric",
                           f"{primary} 种子间标准差 {result['metric_std'][primary]:.4f} 超过 {cfg.max_metric_std}", model_name))
    if result["metrics"].get(primary, 0) > cfg.leakage_ceiling:
        diags.append(_diag(DiagnosticCode.UNSTABLE, "metric",
                           f"{primary} = {result['metrics'][primary]:.4f} 过高，疑似数据泄漏", model_name,
                           "检查是否有特征直接包含标签信息，或在全量数据上做了目标编码"))
    return diags
