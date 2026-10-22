"""把 PipelinePlan 渲染为可独立运行的训练/预测脚本（"复刻"出的算法代码）。

复用的内置算子以 import 方式引用，生成的算子（本次生成或以前沉淀的）源码直接内嵌；每一步附上 Planner 给出的理由。
"""

from __future__ import annotations

import math
import textwrap

from factory.operators import registry
from factory.schemas import GeneratedOperator, PipelinePlan, TaskSpec

TEMPLATE = '''"""{title}

需求：{request}
方案：{plan_name}
设计依据：{rationale}
主指标：{metric} >= {threshold}

由 AlgoForge 自动生成。用法：
    python pipeline.py --data <train.csv>                      # 训练并保存 model.joblib
    python pipeline.py --predict <new.csv> --model model.joblib  # 预测
"""

import argparse

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

{imports}
{generated}

TARGET = {target!r}


def build_pipeline() -> Pipeline:
    return Pipeline([
{steps}
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="训练数据 CSV")
    parser.add_argument("--predict", help="待预测数据 CSV")
    parser.add_argument("--model", default="model.joblib")
    args = parser.parse_args()
    if args.data:
        df = pd.read_csv(args.data)
        pipe = build_pipeline().fit(df.drop(columns=[TARGET]), df[TARGET])
        joblib.dump(pipe, args.model)
        print(f"模型已保存到 {{args.model}}")
    if args.predict:
        pipe = joblib.load(args.model)
        df = pd.read_csv(args.predict)
        X = df.drop(columns=[TARGET], errors="ignore")
        out = pd.DataFrame(pipe.predict_proba(X), columns=[f"proba_{{c}}" for c in pipe.classes_])
        out["prediction"] = pipe.predict(X)
        print(out.head().to_string())


if __name__ == "__main__":
    main()
'''


def render_pipeline(plan: PipelinePlan, spec: TaskSpec, generated: dict[str, GeneratedOperator] | None = None) -> str:
    generated = generated or {}
    imports: dict[str, set[str]] = {}
    embedded = {name: g.source for name, g in generated.items()}
    steps = []
    for i, step in enumerate(plan.steps):
        if step.action == "synthesize":
            # 本次生成的算子：源码内嵌，参数用其 __init__ 默认值
            cls_name, args = step.synth_spec.name, ", ".join(f"{k}={literal(v)}" for k, v in step.params.items())
        else:
            cls = registry.get(step.operator_ref)
            cls_name = cls.__name__
            src = registry.source_of(step.operator_ref)
            if src:  # 以前沉淀的生成算子也内嵌源码，脚本不依赖运行时注册
                embedded.setdefault(cls_name, src)
            else:
                imports.setdefault(cls.__module__, set()).add(cls_name)
            params = {**cls.meta.params, **step.params}
            args = ", ".join(f"{k}={literal(v)}" for k, v in params.items())
        comment = f"  # {step.rationale}" if step.rationale else ""
        steps.append(f'        ("{i}_{step.kind.value}", {cls_name}({args})),{comment}')
    import_lines = "\n".join(f"from {mod} import {', '.join(sorted(names))}" for mod, names in sorted(imports.items()))
    gen_src = "\n\n".join(_strip_future(textwrap.dedent(src).strip()) for src in embedded.values())
    return TEMPLATE.format(
        title=f"{spec.description[:40]} —— 自动生成的算法流水线",
        request=spec.description,
        plan_name=plan.name,
        rationale=plan.rationale or "-",
        metric=spec.primary_metric.name,
        threshold=spec.primary_metric.threshold,
        imports=import_lines,
        generated=("\n\n" + gen_src + "\n") if gen_src else "",
        target=spec.target_column,
        steps="\n".join(steps),
    )


def literal(v) -> str:
    """把参数值写成合法的 Python 字面量（repr(float('inf')) 是 inf，不能直接写进代码）。"""
    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
        return f"float({str(v)!r})"
    if isinstance(v, (list, tuple)):
        inner = ", ".join(literal(x) for x in v) + ("," if isinstance(v, tuple) and len(v) == 1 else "")
        return f"[{inner}]" if isinstance(v, list) else f"({inner})"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{literal(k)}: {literal(x)}" for k, x in v.items()) + "}"
    return repr(v)


def _strip_future(src: str) -> str:
    """内嵌源码时去掉 from __future__ import（它只能出现在文件开头）。"""
    return "\n".join(line for line in src.splitlines() if not line.startswith("from __future__"))
