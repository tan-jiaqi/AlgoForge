"""规则版 Agent：在没有 LLM（mock 模式）时给出确定、合理的输出。

它解析的是与真实 LLM 完全相同的 prompt 上下文（``<<CONTEXT>>`` 块），
因此 mock 模式也走完整的检索、生成、验证、修复与沉淀链路，可用于测试和离线演示。

注意：为了在离线演示中稳定地展示"失败 → 修复 → 通过"，mock 生成的清洗算子首版
**故意**留下一个缺陷（把空白值转成 NaN 后没有填补，会被验证器判为 E_NAN），
修复时再补上。真实 LLM 模式下没有这种人为注入。
"""

from __future__ import annotations

import json
import re

from factory.agents.planner import CONTEXT_BEGIN, CONTEXT_END, PlanSet
from factory.llm import MockLLM
from factory.extract import LessonRecord
from factory.schemas import DiagnosticCode, MetricTarget, Modality, PipelinePlan, Step, SynthSpec, TaskSpec, TaskType


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
        """该类步骤中得分最高的可执行通用算子（只处理特定列的清洗算子不算）。"""
        return next((o for o in options.get(kind, [])
                     if o.get("executable", True) and "string_typed_numeric" not in o.get("conditions", [])), None)

    # 规则版只为清洗步骤生成新算子，其余步骤都复用可执行算子
    models = [o for o in options.get("model", []) if o.get("executable", True)]
    if "interpretable" in spec.constraints:
        models.sort(key=lambda o: "interpretable" not in " ".join(o["evidence"]))
    clean = _cleaning_step(ctx) if spec.modality == Modality.TABULAR else None
    plans = []
    for i in range(min(n, len(models))):
        model = models[(i + offset) % len(models)]
        steps: list[Step] = [clean] if clean else []
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


def _cleaning_step(ctx: dict) -> Step | None:
    """数据含字符串型数值列时：能力库已有清洗算子就复用，否则参考 KG 中的历史实现生成一个。"""
    if "string_typed_numeric" not in ctx["data_profile"]["tags"]:
        return None
    opts = [o for kind in ("imputer", "feature") for o in ctx["operator_options"].get(kind, [])
            if "string_typed_numeric" in o.get("conditions", [])]
    opts.sort(key=lambda o: not o["executable"])
    if not opts:
        return None
    opt = opts[0]
    kind = next(k for k in ("imputer", "feature") if opt in ctx["operator_options"].get(k, []))
    if opt["executable"]:
        return Step(kind=kind, action="reuse", operator_ref=opt["ref"], rationale="复用此前沉淀的清洗算子")
    if not ctx.get("allow_synthesis"):
        return None
    name = opt["ref"].split("@")[0]
    spec = SynthSpec(name=name, kind=kind, purpose="把被读成字符串的数值列转为数值", inputs="原始 DataFrame",
                     outputs="字符串型数值列已转为 float 的 DataFrame", reference=opt["ref"])
    return Step(kind=kind, action="synthesize", synth_spec=spec, rationale="能力库没有可执行的清洗算子，参考历史实现生成")


CLEANER_TEMPLATE = '''from sklearn.base import TransformerMixin

import pandas as pd

from factory.operators.base import Operator, OperatorMeta
from factory.schemas import OperatorKind


class {name}(TransformerMixin, Operator):
    meta = OperatorMeta(
        name="{name}", kind=OperatorKind.{kind}, version={version}, origin="synthesized",
        description="把被读成字符串的数值列（含空白值）转为 float 并填补缺失",
        conditions=["string_typed_numeric", "has_missing_values"], dependencies=["pandas"],
        params={{"columns": {columns!r}, "fill_value": {fill!r}}}{parent},
    )

    def __init__(self, columns={columns!r}, fill_value={fill!r}):
        self.columns = columns
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col].astype(str).str.strip(), errors="coerce")
                if self.fill_value is not None:
                    X[col] = X[col].fillna(self.fill_value)
        return X
'''


def rule_synthesize(ctx: dict) -> str:
    spec = SynthSpec.model_validate(ctx["synth_spec"])
    cols = [c.split(":")[0] for c in ctx["data_profile"]["columns"] if ": string_numeric," in c]
    version = int(re.search(r"version=(\d+)", ctx.get("_system", "")).group(1)) if "_system" in ctx else 1
    parent = f', parent="{spec.reference}"' if spec.reference else ""
    # 首版故意不填补缺失（fill_value=None），用于离线演示修复流程
    code = CLEANER_TEMPLATE.format(name=spec.name, kind=spec.kind.name, version=version, columns=cols, fill=None, parent=parent)
    return f"```python\n{code}```"


def rule_repair(ctx: dict) -> str:
    code = ctx["previous_code"]
    if ctx["diagnostic"]["code"] in (DiagnosticCode.NAN_OUTPUT.value, DiagnosticCode.RUNTIME_ERROR.value):
        code = code.replace('"fill_value": None', '"fill_value": 0.0').replace("fill_value=None", "fill_value=0.0")
    return f"```python\n{code}```"


def rule_lesson(ctx: dict) -> str:
    d = ctx["diagnostic"]
    return LessonRecord(
        title=f"{d['code']}：{d['message'][:30]}",
        trigger=d["message"],
        fix="数值转换产生的缺失值需在同一算子内填补（如新客户的空白累计消费填 0）",
        error_code=DiagnosticCode(d["code"]),
        applies_to=["string_typed_numeric"],
    ).model_dump_json()


def _with_system(handler):
    """把 system prompt 一并交给规则函数（版本号等信息在其中）。"""
    def wrapped(msgs):
        ctx = _context(msgs)
        ctx["_system"] = msgs[0]["content"]
        return handler(ctx)
    return wrapped


def rule_based_mock() -> MockLLM:
    """返回一个按规则响应各 Agent 调用的 MockLLM。"""
    return (
        MockLLM()
        .on("plan_spec", lambda msgs: rule_spec(_context(msgs)).model_dump_json())
        .on("plan", lambda msgs: rule_plans(_context(msgs)).model_dump_json())
        .on("synthesize", _with_system(rule_synthesize))
        .on("repair", lambda msgs: rule_repair(_context(msgs)))
        .on("lesson", lambda msgs: rule_lesson(_context(msgs)))
    )
