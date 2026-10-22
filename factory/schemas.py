"""各 Agent 共享的结构化中间表示。

用三层表示把"做什么""怎么做""代码"解耦：

    TaskSpec（做什么）  ->  PipelinePlan（怎么做）  ->  代码

LLM 的每次输出都要解析成其中一种模型，因此每一层都能单独校验和测试。
验证结果以 ``Diagnostic`` 表示，带有固定的诊断码，工作流据此决定是修复代码还是重新规划。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# 第一层：TaskSpec —— 做什么
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"


class Modality(str, Enum):
    TABULAR = "tabular"
    TEXT = "text"


class MetricTarget(BaseModel):
    """优化指标及流水线必须达到的阈值。"""

    name: str = Field(description="sklearn 风格的指标名，如 'recall'、'f1_macro'、'roc_auc'")
    threshold: float = Field(ge=0.0, le=1.0)
    primary: bool = True


class TaskSpec(BaseModel):
    """对自然语言能力需求的规范化理解。"""

    description: str = Field(description="原始自然语言需求")
    task_type: TaskType
    modality: Modality
    target_column: str
    text_column: str | None = Field(default=None, description="modality 为 text 时必填")
    metrics: list[MetricTarget] = Field(min_length=1)
    constraints: list[str] = Field(
        default_factory=list,
        description="业务约束，如 'interpretable'、'recall_first'、'latency<50ms'",
    )

    @model_validator(mode="after")
    def _check(self) -> "TaskSpec":
        if self.modality == Modality.TEXT and not self.text_column:
            raise ValueError("text_column is required for text tasks")
        if sum(m.primary for m in self.metrics) != 1:
            raise ValueError("exactly one metric must be primary")
        return self

    @property
    def primary_metric(self) -> MetricTarget:
        return next(m for m in self.metrics if m.primary)


# ---------------------------------------------------------------------------
# 第二层：PipelinePlan —— 怎么做
# ---------------------------------------------------------------------------

class OperatorKind(str, Enum):
    IMPUTER = "imputer"
    ENCODER = "encoder"
    SCALER = "scaler"
    FEATURE = "feature"          # 特征工程 / 特征生成
    VECTORIZER = "vectorizer"    # 文本 -> 数值
    MODEL = "model"              # 最终的预测模型


class SynthSpec(BaseModel):
    """尚不存在、需要生成的算子的规格说明。"""

    name: str = Field(pattern=r"^[A-Z][A-Za-z0-9]*$", description="CamelCase 类名")
    kind: OperatorKind
    purpose: str
    inputs: str = Field(description="期望输入，如 '含数值列的 DataFrame'")
    outputs: str = Field(description="产出，如 '新增比值特征后的 DataFrame'")
    reference: str | None = Field(default=None, description="KG 中可参考的历史实现（抽取算子的 Name@version）")


class Step(BaseModel):
    """流水线中的一步：复用已注册算子，或生成新算子。"""

    kind: OperatorKind
    action: Literal["reuse", "synthesize"]
    operator_ref: str | None = Field(default=None, description="注册表键，如 'MissingValueImputer@1'")
    synth_spec: SynthSpec | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""

    @model_validator(mode="after")
    def _check(self) -> "Step":
        if self.action == "reuse" and not self.operator_ref:
            raise ValueError("reuse step requires operator_ref")
        if self.action == "synthesize" and not self.synth_spec:
            raise ValueError("synthesize step requires synth_spec")
        if self.synth_spec and self.synth_spec.kind != self.kind:
            raise ValueError("synth_spec.kind must match step kind")
        return self


class PipelinePlan(BaseModel):
    """有序的步骤列表，最后一步必须是模型。"""

    name: str
    steps: list[Step] = Field(min_length=1)
    rationale: str = Field(default="", description="该方案为何适合此 TaskSpec")

    @model_validator(mode="after")
    def _check(self) -> "PipelinePlan":
        kinds = [s.kind for s in self.steps]
        if kinds[-1] != OperatorKind.MODEL:
            raise ValueError("the last step must be a model")
        if kinds.count(OperatorKind.MODEL) != 1:
            raise ValueError("a plan must contain exactly one model step")
        return self


# ---------------------------------------------------------------------------
# 第三层：代码产物与验证结果
# ---------------------------------------------------------------------------

class GeneratedOperator(BaseModel):
    """Synthesizer 按 SynthSpec 生成的源码。"""

    spec: SynthSpec
    source: str
    version: int = Field(default=1, description="注册时的算子版本号（与已有同名算子顺延）")
    revision: int = Field(default=0, description="本次运行内的修复轮次，0 为首版")

    @property
    def key(self) -> str:
        return f"{self.spec.name}@{self.version}"


class DiagnosticCode(str, Enum):
    # 静态检查
    SYNTAX_ERROR = "E_SYNTAX"
    UNSAFE_CODE = "E_UNSAFE"
    BAD_INTERFACE = "E_INTERFACE"
    # 沙箱执行
    RUNTIME_ERROR = "E_RUNTIME"
    TIMEOUT = "E_TIMEOUT"
    # 契约检查
    BAD_SHAPE = "E_SHAPE"
    BAD_PROBA = "E_PROBA"
    INPUT_MUTATED = "E_MUTATION"
    NAN_OUTPUT = "E_NAN"
    # 指标检查
    METRIC_BELOW_THRESHOLD = "M_THRESHOLD"
    UNSTABLE = "M_UNSTABLE"

    @property
    def category(self) -> Literal["code", "metric"]:
        """代码类错误就地修复；指标类问题触发重新规划。"""
        return "metric" if self.value.startswith("M_") else "code"


#: 诊断码说明，供抽取经验和修复代码时放入 prompt
DIAGNOSTIC_DESCRIPTIONS: dict[DiagnosticCode, str] = {
    DiagnosticCode.SYNTAX_ERROR: "代码语法错误",
    DiagnosticCode.UNSAFE_CODE: "使用了禁止的模块或调用（subprocess、socket、eval、exec 等）",
    DiagnosticCode.BAD_INTERFACE: "缺少 fit/transform/predict 等必需接口或元数据",
    DiagnosticCode.RUNTIME_ERROR: "运行时异常（类型转换失败、未见类别、KeyError 等）",
    DiagnosticCode.TIMEOUT: "执行超时或内存超限",
    DiagnosticCode.BAD_SHAPE: "输出行数或维度不符合约定",
    DiagnosticCode.BAD_PROBA: "预测概率不在 [0,1] 或各类之和不为 1",
    DiagnosticCode.INPUT_MUTATED: "算子原地修改了输入数据",
    DiagnosticCode.NAN_OUTPUT: "输出中出现 NaN/Inf",
    DiagnosticCode.METRIC_BELOW_THRESHOLD: "指标未达到任务要求的阈值（如召回率不足）",
    DiagnosticCode.UNSTABLE: "不同随机种子下指标波动过大，或疑似数据泄漏导致指标虚高",
}


class Diagnostic(BaseModel):
    code: DiagnosticCode
    stage: Literal["static", "sandbox", "contract", "metric"]
    message: str
    operator: str | None = Field(default=None, description="出错的算子（能定位时）")
    hint: str = ""


class ValidationReport(BaseModel):
    """对一条组装好的流水线运行验证器的结果。"""

    plan_name: str
    passed: bool
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict, description="各指标在多种子 × 多折上的均值")
    metric_std: dict[str, float] = Field(default_factory=dict, description="各指标在不同种子间的标准差")
    resources: dict[str, float] = Field(default_factory=dict, description="fit_seconds / predict_ms_per_1k / peak_rss_mb")
    notes: list[str] = Field(default_factory=list)

    def primary_value(self, spec: TaskSpec) -> float:
        return self.metrics.get(spec.primary_metric.name, float("-inf"))


class RepairRecord(BaseModel):
    """一次"诊断 → 修复"，任务结束后提炼为 Lesson。"""

    operator: str
    revision: int = Field(description="修复后的修订号")
    diagnostic: Diagnostic
    lessons: list[str] = Field(default_factory=list, description="修复时放进 prompt 的 Lesson id")
    fixed: bool | None = Field(default=None, description="修复后的版本是否通过了该项检查")
    before: str = Field(default="", description="修复前源码")
    after: str = Field(default="", description="修复后源码")


# ---------------------------------------------------------------------------
# 在各 Agent 之间传递的工作流状态
# ---------------------------------------------------------------------------

class RunState(BaseModel):
    """各 Agent 读写的共享状态（黑板）。"""

    run_id: str
    request: str
    data_path: str
    target: str | None = Field(default=None, description="用户指定的目标列（可选，优先于 LLM 推断）")
    status: Literal["running", "succeeded", "failed"] = "running"
    spec: TaskSpec | None = None
    profile_tags: list[str] = Field(default_factory=list)
    candidates: list[PipelinePlan] = Field(default_factory=list)
    generated: dict[str, GeneratedOperator] = Field(default_factory=dict)
    reports: list[ValidationReport] = Field(default_factory=list)
    best_plan: str | None = None
    repair_rounds: int = 0
    replans: int = 0
    history: list[ValidationReport] = Field(default_factory=list, description="被修复替换掉的旧验证结果")
    repairs: list[RepairRecord] = Field(default_factory=list)
    registered: list[str] = Field(default_factory=list, description="本次注册到能力库的新算子")
    lessons_learned: list[str] = Field(default_factory=list, description="本次提炼写入 KG 的 Lesson")
    lessons_used: list[str] = Field(default_factory=list, description="规划/修复时放进 prompt 的 Lesson 节点 id")
    kg_writes: list[str] = Field(default_factory=list, description="回写到 KG 的 ValidationRun 节点 id")
    artifacts: dict[str, str] = Field(default_factory=dict)
    log: list[str] = Field(default_factory=list)

    def note(self, message: str) -> None:
        self.log.append(message)
