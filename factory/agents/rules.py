"""规则版 Planner：在没有 LLM（mock 模式）时给出确定、合理的输出。

它解析的是与真实 LLM 完全相同的 prompt 上下文（``<<CONTEXT>>`` 块），
因此 mock 模式也走完整的检索与校验链路，可用于测试和离线演示。
"""

from __future__ import annotations

import json
import re

from factory.agents.planner import CONTEXT_BEGIN, CONTEXT_END, PlanSet
from factory.llm import MockLLM
from factory.schemas import MetricTarget, Modality, PipelinePlan, Step, TaskSpec, TaskType


def _context(messages) -> dict:
    text = messages[1]["content"]
    m = re.search(re.escape(CONTEXT_BEGIN) + r"(.*?)" + re.escape(CONTEXT_END), text, re.DOTALL)
    return json.loads(m.group(1))


def rule_spec(ctx: dict) -> TaskSpec:
    request: str = ctx["request"]
    prof = ctx["data_profile"]
    target = ctx.get("user_target") or prof["target"]
    multiclass = len(prof["class_counts"]) > 2
    text_col = next((c.split(":")[0] for c in prof["columns"] if ": text," in c), None)

    constraints = []
    if any(w in request for w in ("可解释", "解释", "interpretable")):
        constraints.append("interpretable")
    if "召回" in request or "recall" in request.lower():
        constraints.append("recall_first")

    known = {m["name"]: m for m in ctx.get("scenario_metrics") or []}
    if multiclass:
        primary = "recall_macro" if "recall_first" in constraints else "f1_macro"
    else:
        primary = "recall" if "recall_first" in constraints else next((n for n, m in known.items() if m.get("primary")), "roc_auc")
    threshold = (known.get(primary) or {}).get("threshold") or 0.7
    m = re.search(r"(?:不低于|至少|>=?)\s*(0\.\d+)", request)
    if m:
        threshold = float(m.group(1))
    metrics = [MetricTarget(name=primary, threshold=threshold, primary=True)]
    for name, info in known.items():
        if name != primary and info.get("threshold") and not multiclass:
            metrics.append(MetricTarget(name=name, threshold=info["threshold"], primary=False))

    return TaskSpec(
        description=request,
        task_type=TaskType.MULTICLASS_CLASSIFICATION if multiclass else TaskType.BINARY_CLASSIFICATION,
        modality=Modality.TEXT if text_col else Modality.TABULAR,
        target_column=target,
        text_column=text_col,
        metrics=metrics,
        constraints=constraints,
    )


def rule_plans(ctx: dict) -> PlanSet:
    spec = TaskSpec.model_validate(ctx["task_spec"])
    tags = set(ctx["data_profile"]["tags"])
    options = ctx["operator_options"]
    n = ctx["n_candidates"]
    offset = len(ctx.get("feedback") or [])  # 每次重新规划换一批模型

    def best(kind: str) -> dict | None:
        return (options.get(kind) or [None])[0]

    models = list(options.get("model", []))
    if "interpretable" in spec.constraints:
        models.sort(key=lambda o: "interpretable" not in " ".join(o["evidence"]))
    plans = []
    for i in range(min(n, len(models))):
        model = models[(i + offset) % len(models)]
        steps: list[Step] = []
        if spec.modality == Modality.TEXT:
            vec = best("vectorizer")
            steps.append(Step(kind="vectorizer", action="reuse", operator_ref=vec["ref"], rationale="文本需先向量化"))
        else:
            if "has_missing_values" in tags and best("imputer"):
                steps.append(Step(kind="imputer", action="reuse", operator_ref=best("imputer")["ref"], rationale="数据存在缺失值"))
            if "has_categorical_features" in tags and best("encoder"):
                steps.append(Step(kind="encoder", action="reuse", operator_ref=best("encoder")["ref"], rationale="存在类别特征"))
            if "LogisticRegression" in model["ref"] and best("scaler"):
                steps.append(Step(kind="scaler", action="reuse", operator_ref=best("scaler")["ref"], rationale="线性模型需要标准化"))
        params = {"class_weight": "balanced"} if "imbalanced" in tags and "class_weight" in model["params"] else {}
        steps.append(Step(kind="model", action="reuse", operator_ref=model["ref"], params=params,
                          rationale="；".join(model["evidence"]) or "检索得分最高的模型"))
        name = re.sub(r"(?<!^)(?=[A-Z])", "_", model["ref"].split("@")[0]).lower().replace("_model", "")
        plans.append(PipelinePlan(name=f"{name}_pipeline", steps=steps, rationale="规则规划：按画像标签选择预处理，按检索得分选择模型"))
    return PlanSet(plans=plans)


def rule_based_mock() -> MockLLM:
    """返回一个按规则响应 Planner 调用的 MockLLM。"""
    return (
        MockLLM()
        .on("plan_spec", lambda msgs: rule_spec(_context(msgs)).model_dump_json())
        .on("plan", lambda msgs: rule_plans(_context(msgs)).model_dump_json())
    )
