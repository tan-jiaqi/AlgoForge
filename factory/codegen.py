"""把 PipelinePlan 渲染为可独立运行的训练/预测脚本（"复刻"出的算法代码）。

复用的算子以 import 方式引用，生成的算子源码直接内嵌；每一步附上 Planner 给出的理由。
"""

from __future__ import annotations

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
    steps = []
    for i, step in enumerate(plan.steps):
        ref = step.operator_ref or step.synth_spec.name
        cls = registry.get(ref)
        if cls.meta.origin != "synthesized":
            imports.setdefault(cls.__module__, set()).add(cls.__name__)
        params = {**cls.meta.params, **step.params}
        args = ", ".join(f"{k}={v!r}" for k, v in params.items())
        comment = f"  # {step.rationale}" if step.rationale else ""
        steps.append(f'        ("{i}_{step.kind.value}", {cls.__name__}({args})),{comment}')
    import_lines = "\n".join(f"from {mod} import {', '.join(sorted(names))}" for mod, names in sorted(imports.items()))
    gen_src = "\n\n".join(textwrap.dedent(g.source).strip() for g in generated.values())
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
