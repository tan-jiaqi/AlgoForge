"""Planner Agent：需求理解 → 知识检索 → 方案规划。

1. understand：自然语言需求 + 数据画像 → ``TaskSpec``（LLM，结构化输出）；
2. retrieve：按 TaskSpec 与画像标签，分步骤从 KG 检索候选算子和相关 Lesson（确定性）；
3. propose：TaskSpec + 画像 + 检索结果 → 若干候选 ``PipelinePlan``（LLM，带语义校验）。

每一步由 Planner 决定 Reuse（能力库中已有可执行算子）还是 Synthesize（交给 Synthesizer 生成；
KG 中有抽取到的历史实现时，在 synth_spec.reference 中引用它）。

所有 LLM 输出都经过 schema 与语义双重校验（算子是否存在、参数是否合法、模态是否匹配），
失败时把错误反馈给 LLM 重试。
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from factory.kg import CapabilityKG, node_id
from factory.llm import LLM
from factory.operators import registry
from factory.operators.base import OperatorRegistry
from factory.profile import DataProfile
from factory.schemas import Modality, OperatorKind, PipelinePlan, TaskSpec

TABULAR_ORDER = [OperatorKind.IMPUTER, OperatorKind.FEATURE, OperatorKind.ENCODER, OperatorKind.SCALER, OperatorKind.MODEL]
TEXT_ORDER = [OperatorKind.VECTORIZER, OperatorKind.MODEL]

CONTEXT_BEGIN, CONTEXT_END = "<<CONTEXT>>", "<</CONTEXT>>"


class PlanSet(BaseModel):
    plans: list[PipelinePlan] = Field(min_length=1)


@dataclass
class PlanningContext:
    """检索结果：每类步骤的候选算子（含分项得分与证据）、相关经验、场景指标。"""

    scenario: str | None
    scenario_metrics: list[dict]
    options: dict[str, list[dict]] = field(default_factory=dict)
    lessons: list[dict] = field(default_factory=list)

    def to_prompt(self) -> dict:
        return {
            "scenario": self.scenario,
            "scenario_metrics": self.scenario_metrics,
            "operator_options": self.options,
            "lessons": [{k: les[k] for k in ("title", "trigger", "fix", "error_code")} for les in self.lessons],
        }


def resolve_scenario(kg: CapabilityKG, data_path: str) -> str | None:
    """按数据集路径匹配 KG 中的场景（场景节点的 dataset 属性）。"""
    norm = str(data_path).replace("\\", "/")
    for nid, d in kg.nodes("Scenario"):
        ds = d.get("dataset")
        if ds and (norm == ds or norm.endswith("/" + ds) or norm.endswith(ds)):
            return d["key"]
    return None


def scenario_metrics(kg: CapabilityKG, scenario: str | None) -> list[dict]:
    if not scenario:
        return []
    sid = node_id("Scenario", scenario)
    out = []
    for _, mid, d in kg.g.out_edges(sid, data=True):
        if d["rel"] == "EVALUATED_BY":
            out.append({"name": kg.g.nodes[mid]["name"], "threshold": d.get("threshold"), "primary": d.get("primary")})
    return out


def operator_params(ref: str, reg: OperatorRegistry = registry) -> list[str]:
    return [p for p in inspect.signature(reg.get(ref).__init__).parameters if p != "self"]


def _wrap(payload: dict) -> str:
    """把结构化上下文包在标记里放进 prompt；MockLLM 也据此解析，保证与真实 LLM 看到的一致。"""
    return f"{CONTEXT_BEGIN}\n{json.dumps(payload, ensure_ascii=False, indent=1, default=str)}\n{CONTEXT_END}"


class Planner:
    MAX_SYNTH_STEPS = 2

    def __init__(self, llm: LLM, kg: CapabilityKG, reg: OperatorRegistry = registry, *, allow_synthesis: bool = True):
        self.llm = llm
        self.kg = kg
        self.reg = reg
        self.allow_synthesis = allow_synthesis

    # ------------------------------------------------------------------
    # 1. 需求理解
    # ------------------------------------------------------------------

    def understand(self, request: str, profile: DataProfile, *, target: str | None, scenario: str | None) -> TaskSpec:
        columns = [c.name for c in profile.columns] + ([profile.target] if profile.target else [])
        payload = {
            "request": request,
            "user_target": target,
            "data_profile": profile.summary(),
            "scenario_metrics": scenario_metrics(self.kg, scenario),
        }
        system = (
            "你是算法能力工厂的 Planner。把用户的自然语言需求解析为 TaskSpec。\n"
            "规则：\n"
            "- description 原样保留用户需求；target_column 必须是数据中的列，用户指定了 user_target 时直接使用；\n"
            "- 目标列 2 类为 binary_classification，多于 2 类为 multiclass_classification；\n"
            "- 数据画像含 text_input 时 modality 为 text，并填写 text_column；\n"
            "- metrics 恰好一个 primary。需求未给出阈值时，优先采用 scenario_metrics 中的阈值，否则取 0.7；\n"
            "- 多分类用 f1_macro、recall_macro 等 macro 指标；\n"
            "- constraints 使用 snake_case 标签，如 interpretable、recall_first。"
        )

        def check(spec: TaskSpec) -> None:
            if spec.target_column not in columns:
                raise ValueError(f"target_column {spec.target_column!r} 不在数据列中：{columns}")
            if spec.text_column and spec.text_column not in columns:
                raise ValueError(f"text_column {spec.text_column!r} 不在数据列中")

        spec = self.llm.structured(
            [{"role": "system", "content": system}, {"role": "user", "content": _wrap(payload)}],
            TaskSpec,
            tag="plan_spec",
            check=check,
            attempts=3,
        )
        if target and spec.target_column != target:
            spec = spec.model_copy(update={"target_column": target})
        return self._grounded_metrics(spec, request, payload["scenario_metrics"])

    @staticmethod
    def _grounded_metrics(spec: TaskSpec, request: str, known: list[dict]) -> TaskSpec:
        """次要指标的阈值必须有出处（用户需求或 KG 中的场景指标），否则只报告、不作为门槛。

        例：需求文档只要求"报告 recall@20%"，LLM 却给它编了 0.7 的阈值，会让所有方案无端失败。
        """
        sources = {(m["name"], m.get("threshold")) for m in known if m.get("threshold") is not None}
        kept = [m for m in spec.metrics
                if m.primary or (m.name, m.threshold) in sources or f"{m.threshold:g}" in request]
        return spec if len(kept) == len(spec.metrics) else spec.model_copy(update={"metrics": kept})

    # ------------------------------------------------------------------
    # 2. 知识检索
    # ------------------------------------------------------------------

    def retrieve(
        self, spec: TaskSpec, profile: DataProfile, *, scenario: str | None, executable_only: bool | None = None, top_k: int = 3
    ) -> PlanningContext:
        conditions = sorted(set(profile.tags) | set(spec.constraints))
        query = f"{spec.description} {' '.join(conditions)}"
        order = TEXT_ORDER if spec.modality == Modality.TEXT else TABULAR_ORDER
        if executable_only is None:
            executable_only = not self.allow_synthesis
        ctx = PlanningContext(scenario=scenario, scenario_metrics=scenario_metrics(self.kg, scenario))
        for kind in order:
            hits = self.kg.search_operators(
                query, kind=kind, modality=spec.modality, conditions=conditions, scenario=scenario,
                metric=spec.primary_metric.name, executable_only=executable_only, top_k=top_k,
            )
            if hits:
                ctx.options[kind.value] = [
                    {
                        "ref": h.key,
                        "executable": h.key in self.reg,
                        "origin": h.attrs.get("origin"),
                        "description": h.attrs["description"],
                        "conditions": sorted(self.kg.operator_conditions(node_id("Operator", h.key))),
                        "params": operator_params(h.key, self.reg) if h.key in self.reg else [],
                        "defaults": h.attrs.get("params", {}),
                        "score": h.score,
                        "breakdown": h.breakdown,
                        "evidence": h.evidence,
                    }
                    for h in hits
                ]
        ctx.lessons = self.kg.search_lessons(query, top_k=5)
        return ctx

    # ------------------------------------------------------------------
    # 3. 方案规划
    # ------------------------------------------------------------------

    def propose(
        self,
        spec: TaskSpec,
        profile: DataProfile,
        ctx: PlanningContext,
        *,
        n_candidates: int = 2,
        feedback: list[str] | None = None,
    ) -> list[PipelinePlan]:
        order = TEXT_ORDER if spec.modality == Modality.TEXT else TABULAR_ORDER
        payload = {
            "task_spec": spec.model_dump(mode="json"),
            "data_profile": profile.summary(),
            "n_candidates": n_candidates,
            "allow_synthesis": self.allow_synthesis,
            "max_synthesize_steps": self.MAX_SYNTH_STEPS if self.allow_synthesis else 0,
            "step_order": [k.value for k in order],
            **ctx.to_prompt(),
            "feedback": feedback or [],
        }
        system = (
            "你是算法能力工厂的 Planner。根据 TaskSpec、数据画像和从知识图谱检索到的候选算子，"
            f"给出 {n_candidates} 个互不相同的候选 PipelinePlan，放在 plans 列表中；优先使用不同的模型，只有一个可选模型时改变预处理或超参数。\n"
            "规则：\n"
            "- 步骤按 step_order 排列（清洗类 synthesize 步骤可放最前），可省略不需要的步骤，最后一步必须是 model；\n"
            + (
                "- executable=true 的选项用 action=reuse，operator_ref 取其 ref；\n"
                "- 能力库中没有合适算子、但数据或需求确有需要时（如 string_typed_numeric、id_columns、特征工程），"
                "用 action=synthesize 并给出 synth_spec（name 为 CamelCase 英文，purpose/inputs/outputs 用中文）；"
                "若 operator_options 中有 executable=false 的历史实现，name 沿用其名字并把 ref 填入 synth_spec.reference；"
                "每个方案最多 max_synthesize_steps 个 synthesize 步骤；清洗类的 synthesize 步骤放在最前面；\n"
                "- 只处理特定列的清洗算子（如 TotalChargesCleaner）不能代替通用的缺失值填补，其他列有缺失时仍需 imputer；\n"
                if self.allow_synthesis
                else "- 每步 action 为 reuse，operator_ref 只能取 operator_options 中给出的 ref；\n"
            )
            + "- reuse 步骤的 params 只能使用该算子 params 列表中的参数名，不需要改动时留空；\n"
            "- 每步 rationale 用一句中文说明理由，引用 evidence 或 lessons；plan 的 rationale 说明整体设计依据；\n"
            "- 结合 lessons 中的经验，例如不平衡数据使用 class_weight=balanced；\n"
            "- feedback 非空时，表示之前的方案未通过验证，新方案必须针对这些问题调整（换算子或调参）；\n"
            "- plan 的 name 用简短的 snake_case 英文，彼此不同。"
        )

        def check(ps: PlanSet) -> None:
            errors = []
            names = [p.name for p in ps.plans]
            if len(set(names)) != len(names):
                errors.append("plan name 重复")
            for p in ps.plans:
                errors += [f"{p.name}: {e}" for e in self.plan_errors(p, spec)]
            if errors:
                raise ValueError("；".join(errors))

        plans = self.llm.structured(
            [{"role": "system", "content": system}, {"role": "user", "content": _wrap(payload)}],
            PlanSet,
            tag="plan",
            check=check,
            attempts=3,
        ).plans
        return [self.bind(p, spec) for p in plans[:n_candidates]]

    def _known_name(self, name: str) -> str:
        """返回知识图谱中与 name 仅大小写不同的算子名（没有则原样返回）。"""
        for _, d in self.kg.nodes("Operator"):
            if d["name"].lower() == name.lower():
                return d["name"]
        return name

    def _align_synth_spec(self, spec):
        """生成算子与图谱中已有算子同名（忽略大小写）时：沿用已有名字，并把最新版本记为 reference，
        这样新版本会通过 DERIVED_FROM 接入原有的版本谱系，而不是变成一个孤立的"新"算子。"""
        name = self._known_name(spec.name)
        ref = spec.reference
        if ref and "@" not in ref:
            ref = self._known_name(ref)
        versions = [d["key"] for _, d in self.kg.nodes("Operator") if d["name"] == (ref if ref and "@" not in ref else name)]
        if (ref is None or "@" not in ref) and versions:
            ref = max(versions, key=lambda k: int(k.split("@")[1]))
        return spec.model_copy(update={"name": name, "reference": ref})

    def plan_errors(self, plan: PipelinePlan, spec: TaskSpec, *, allow_synthesis: bool = False) -> list[str]:
        """语义校验：算子存在、类型与模态匹配、参数合法。"""
        allow_synthesis = allow_synthesis or self.allow_synthesis
        errors = []
        n_synth = sum(s.action == "synthesize" for s in plan.steps)
        if n_synth > self.MAX_SYNTH_STEPS:
            errors.append(f"synthesize 步骤 {n_synth} 个，超过上限 {self.MAX_SYNTH_STEPS}")
        for i, step in enumerate(plan.steps):
            if step.action == "synthesize":
                if not allow_synthesis:
                    errors.append(f"第 {i + 1} 步：当前只允许 reuse")
                    continue
                name = step.synth_spec.name
                existing = self._known_name(name)  # 名字大小写不同也算同一个算子
                if existing in self.reg and self.reg.get(existing).meta.origin != "extracted":
                    errors.append(f"第 {i + 1} 步：能力库已有可执行算子 {self.reg.get(existing).meta.key}，应使用 reuse")
                ref = step.synth_spec.reference
                if ref and node_id("Operator", ref) not in self.kg.g:
                    errors.append(f"第 {i + 1} 步：reference {ref} 不在知识图谱中")
                continue
            if step.operator_ref not in self.reg:
                errors.append(f"第 {i + 1} 步：未知算子 {step.operator_ref}")
                continue
            meta = self.reg.get(step.operator_ref).meta
            if meta.kind != step.kind:
                errors.append(f"第 {i + 1} 步：{meta.key} 的类型是 {meta.kind.value}，不是 {step.kind.value}")
            if spec.modality not in meta.modalities:
                errors.append(f"第 {i + 1} 步：{meta.key} 不支持 {spec.modality.value} 数据")
            bad = set(step.params) - set(operator_params(step.operator_ref, self.reg))
            if bad:
                errors.append(f"第 {i + 1} 步：{meta.key} 没有参数 {sorted(bad)}")
        return errors

    def bind(self, plan: PipelinePlan, spec: TaskSpec) -> PipelinePlan:
        """确定性补全：规范化算子引用为 Name@version，文本列等参数由 TaskSpec 绑定。"""
        steps = []
        for step in plan.steps:
            update: dict = {}
            if step.action == "synthesize":
                update["synth_spec"] = self._align_synth_spec(step.synth_spec)
            if step.action == "reuse" and step.operator_ref:
                meta = self.reg.get(step.operator_ref).meta
                update["operator_ref"] = meta.key
                params = dict(step.params)
                if spec.text_column and "column" in operator_params(meta.key, self.reg):
                    params["column"] = spec.text_column
                if "ngram_range" in params:
                    params["ngram_range"] = tuple(params["ngram_range"])
                update["params"] = params
            steps.append(step.model_copy(update=update))
        return plan.model_copy(update={"steps": steps})

