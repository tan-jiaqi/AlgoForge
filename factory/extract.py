"""能力抽取：从历史代码仓库与业务文档中抽取算法能力，构建能力知识图谱。

流程：
1. 代码：用 ``ast`` 解析函数签名、docstring、依赖与源码（确定性），再由 LLM 归纳为 ``CapabilityCard``；
2. 文档：LLM 从需求、实验记录、失败复盘、调参经验中抽取指标、约束、历史实验和经验（``DocKnowledge``）；
3. 建图：内置算子 + 抽取到的算子/指标/实验/经验写入 ``CapabilityKG``，并保存 JSON / GraphML 与原始抽取结果。

LLM 输出都经过 pydantic 校验；单个文件抽取失败只记录错误，不影响其他文件。
"""

from __future__ import annotations

import ast
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import yaml
from pydantic import BaseModel, Field

from factory.kg import CONDITION_TAGS, DEFAULT_KG_PATH, CapabilityKG, node_id
from factory.llm import LLM
from factory.operators import registry
from factory.operators.base import OperatorMeta
from factory.schemas import DIAGNOSTIC_DESCRIPTIONS, DiagnosticCode, OperatorKind
from factory.utils import ROOT, get_logger

log = get_logger(__name__)

DEFAULT_MATERIALS = ROOT / "data" / "materials"

# import 名 → pip 包名；标准库不计入依赖
IMPORT_TO_PACKAGE = {"sklearn": "scikit-learn", "lightgbm": "lightgbm", "pandas": "pandas", "numpy": "numpy"}


# ---------------------------------------------------------------------------
# 抽取结果的数据结构
# ---------------------------------------------------------------------------

@dataclass
class CodeFunction:
    """AST 解析出的一个顶层函数。"""

    repo: str
    module: str
    name: str
    signature: str
    docstring: str
    dependencies: list[str]
    source: str
    lineno: int

    @property
    def path(self) -> str:
        return f"{self.repo}/{self.module}.py::{self.name}"


class CapabilityCard(BaseModel):
    """LLM 对一个历史函数的能力归纳。"""

    function: str = Field(description="原函数名")
    capability_type: Literal["operator", "metric", "utility"] = Field(
        description="operator=可作为流水线步骤的数据变换/模型；metric=评价指标；utility=日志等辅助函数"
    )
    name: str = Field(description="CamelCase 英文能力名，例如 TotalChargesCleaner")
    kind: OperatorKind | None = Field(default=None, description="capability_type=operator 时必填")
    description: str = Field(description="一句话中文描述，说明做什么、为什么")
    inputs: str = ""
    outputs: str = ""
    conditions: list[str] = Field(default_factory=list, description="适用条件标签，snake_case")
    params: dict[str, object] = Field(default_factory=dict, description="关键参数及默认值")


class MetricRecord(BaseModel):
    name: str = Field(description="sklearn 风格指标名，如 recall、roc_auc、recall_at_20")
    threshold: float | None = None
    primary: bool = False


class ExperimentRecord(BaseModel):
    pipeline: list[str] = Field(description="按顺序列出的已知算子名")
    metrics: dict[str, float]
    valid: bool = Field(description="结论是否有效；被作废（如数据泄漏）的实验为 false")
    note: str = ""


class LessonRecord(BaseModel):
    title: str
    trigger: str = Field(description="什么情况下会出现这个问题（现象/报错）")
    fix: str = Field(description="如何修复或规避")
    error_code: DiagnosticCode | None = None
    applies_to: list[str] = Field(default_factory=list, description="相关的算子名、条件标签或场景名")


class DocKnowledge(BaseModel):
    """LLM 从一篇业务文档中抽取的知识。"""

    metrics: list[MetricRecord] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list, description="业务约束标签，snake_case")
    experiments: list[ExperimentRecord] = Field(default_factory=list)
    lessons: list[LessonRecord] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    functions: list[dict] = Field(default_factory=list)
    cards: dict[str, CapabilityCard] = Field(default_factory=dict, description="函数路径 → 能力卡片")
    docs: dict[str, DocKnowledge] = Field(default_factory=dict, description="文档路径 → 抽取知识")
    errors: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. 代码：AST 解析（确定性）
# ---------------------------------------------------------------------------

def scan_repo(repo_dir: Path, repo_name: str | None = None) -> list[CodeFunction]:
    """解析仓库中所有 .py 文件的公开顶层函数（以下划线开头的视为私有，跳过）。"""
    repo_dir = Path(repo_dir)
    repo_name = repo_name or repo_dir.name
    functions: list[CodeFunction] = []
    for file in sorted(repo_dir.rglob("*.py")):
        text = file.read_text(encoding="utf-8")
        tree = ast.parse(text)
        imports = _module_imports(tree)
        module = file.relative_to(repo_dir).with_suffix("").as_posix().replace("/", ".")
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                continue
            source = ast.get_source_segment(text, node) or ""
            used = {alias for alias in imports if alias in source}
            deps = sorted({IMPORT_TO_PACKAGE[imports[a]] for a in used if imports[a] in IMPORT_TO_PACKAGE})
            functions.append(
                CodeFunction(
                    repo=repo_name,
                    module=module,
                    name=node.name,
                    signature=f"{node.name}({ast.unparse(node.args)})",
                    docstring=ast.get_docstring(node) or "",
                    dependencies=deps,
                    source=source,
                    lineno=node.lineno,
                )
            )
    return functions


def _module_imports(tree: ast.Module) -> dict[str, str]:
    """返回 {本地别名: 顶层包名}，例如 ``import lightgbm as lgb`` → {"lgb": "lightgbm"}。"""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out[a.asname or a.name.split(".")[0]] = a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                out[a.asname or a.name] = node.module.split(".")[0]
    return out


# ---------------------------------------------------------------------------
# 2. LLM 归纳
# ---------------------------------------------------------------------------

def _condition_vocab() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in CONDITION_TAGS.items())


def summarize_function(llm: LLM, fn: CodeFunction) -> CapabilityCard:
    system = (
        "你是算法能力工厂的能力抽取专家。阅读一个历史项目中的 Python 函数，把它归纳为可复用的能力卡片。\n"
        f"kind 可选值：{[k.value for k in OperatorKind]}。\n"
        "conditions 优先从以下标签中选择，确有必要时可新增 snake_case 标签：\n" + _condition_vocab()
    )
    user = (
        f"仓库：{fn.repo}  模块：{fn.module}\n签名：{fn.signature}\n依赖：{fn.dependencies}\n"
        f"源码：\n```python\n{fn.source}\n```"
    )
    return llm.structured(
        [{"role": "system", "content": system}, {"role": "user", "content": user}], CapabilityCard, tag="extract_code"
    )


def extract_doc(
    llm: LLM, text: str, *, doc_kind: str, scenario: str, builtin: list[str], extracted: list[str]
) -> DocKnowledge:
    codes = "\n".join(f"- {c.value}: {d}" for c, d in DIAGNOSTIC_DESCRIPTIONS.items())
    system = (
        "你是算法能力工厂的知识抽取专家。从一篇业务文档中抽取结构化知识：评价指标与阈值、业务约束、"
        "历史实验（流水线与指标）、可复用的经验教训。没有的内容返回空列表，不要编造。\n"
        f"当前场景：{scenario}\n"
        f"已知算子名（experiments.pipeline 与 lessons.applies_to 必须使用这些名字；"
        f"内置可执行算子与抽取算子含义相同时优先用内置算子）：\n内置：{builtin}\n抽取：{extracted}\n"
        "条件/约束标签优先使用：\n" + _condition_vocab() + "\n"
        "经验的 error_code 从以下诊断码中选择最贴切的一个，无法对应时为 null：\n" + codes
    )
    user = f"文档类型：{doc_kind}\n\n{text}"
    return llm.structured(
        [{"role": "system", "content": system}, {"role": "user", "content": user}], DocKnowledge, tag="extract_doc"
    )


def _run_parallel(jobs: dict[str, Callable[[], Any]], workers: int, errors: dict[str, str]) -> dict[str, Any]:
    """并发执行 LLM 抽取任务；单个失败只记录到 errors。"""
    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {key: pool.submit(job) for key, job in jobs.items()}
        for key, fut in futures.items():
            try:
                results[key] = fut.result()
            except Exception as exc:  # noqa: BLE001
                errors[key] = f"{type(exc).__name__}: {exc}"
                log.warning("抽取失败 %s：%s", key, errors[key])
    return results


def run_extraction(llm: LLM, materials: Path = DEFAULT_MATERIALS, workers: int = 4) -> tuple[ExtractionResult, dict]:
    """执行全部抽取，返回 (抽取结果, manifest)。"""
    manifest = yaml.safe_load((materials / "manifest.yaml").read_text(encoding="utf-8"))
    scenario = manifest["scenario"]["name"]
    result = ExtractionResult()

    functions = []
    for repo in manifest.get("code_repos", []):
        functions += scan_repo(materials / repo["path"], repo.get("name"))
    result.functions = [{k: v for k, v in asdict(f).items() if k != "source"} | {"path": f.path} for f in functions]
    log.info("AST 解析到 %d 个函数，开始 LLM 归纳", len(functions))
    cards = _run_parallel({f.path: (lambda f=f: summarize_function(llm, f)) for f in functions}, workers, result.errors)
    result.cards = {k: cards[k] for k in sorted(cards)}

    builtin = sorted({m.name for m in registry.list()})
    extracted = sorted({c.name for c in result.cards.values() if c.capability_type == "operator"} - set(builtin))
    docs = {d["path"]: d for d in manifest.get("docs", [])}
    jobs = {
        path: (
            lambda d=d: extract_doc(
                llm,
                (materials / d["path"]).read_text(encoding="utf-8"),
                doc_kind=d["kind"],
                scenario=scenario,
                builtin=builtin,
                extracted=extracted,
            )
        )
        for path, d in docs.items()
    }
    log.info("开始抽取 %d 篇文档", len(jobs))
    doc_results = _run_parallel(jobs, workers, result.errors)
    result.docs = {k: doc_results[k] for k in sorted(doc_results)}
    return result, manifest


# ---------------------------------------------------------------------------
# 3. 建图
# ---------------------------------------------------------------------------

def build_kg(result: ExtractionResult, manifest: dict, functions: list[CodeFunction] | None = None) -> CapabilityKG:
    """把内置算子与抽取结果写入一张新的能力知识图谱。"""
    kg = CapabilityKG()
    kg.add_registry(registry)
    sc = manifest["scenario"]
    scenario = kg.add_scenario(
        sc["name"],
        title=sc["title"],
        task_type=sc["task_type"],
        modality=sc["modality"],
        dataset=sc.get("dataset"),
        target_column=sc.get("target_column"),
    )
    code_by_path = {f.path: f for f in functions or []}
    fn_meta = {f["path"]: f for f in result.functions}

    # 代码 → 算子 / 指标
    for path, card in result.cards.items():
        src = kg.add_source("code", path, title=card.function)
        if card.capability_type == "metric":
            kg.add_edge(kg.add_node("Metric", _snake(card.name), name=_snake(card.name), description=card.description), "EXTRACTED_FROM", src)
            continue
        if card.capability_type != "operator" or card.kind is None:
            continue
        meta = OperatorMeta(
            name=card.name,
            kind=card.kind,
            description=card.description,
            inputs=card.inputs or "pandas.DataFrame",
            outputs=card.outputs or "pandas.DataFrame",
            conditions=card.conditions,
            dependencies=fn_meta.get(path, {}).get("dependencies", []),
            params=card.params,
            origin="extracted",
        )
        if node_id("Operator", meta.key) in kg.g:
            log.warning("抽取的算子 %s 与已有算子重名，跳过", meta.key)
            continue
        code = code_by_path[path].source if path in code_by_path else ""
        kg.add_operator(meta, executable=False, source=src, code=code)

    # 文档 → 场景指标、约束、历史实验、经验
    doc_kinds = {d["path"]: d["kind"] for d in manifest.get("docs", [])}
    constraints: list[str] = []
    for path, know in result.docs.items():
        src = kg.add_source("doc", path, title=doc_kinds.get(path, ""))
        kg.add_edge(scenario, "EXTRACTED_FROM", src)
        # 场景的指标阈值与业务约束只以需求文档为准，避免实验记录、调参笔记里的描述混入
        if doc_kinds.get(path) == "requirement":
            for m in know.metrics:
                kg.add_metric(scenario, m.name, m.threshold, m.primary)
            constraints += [t for t in know.constraints if t not in constraints]
        for exp in know.experiments:
            ops = [o for o in (_resolve_operator(kg, n) for n in exp.pipeline) if o]
            if ops:
                kg.add_validation_run(
                    scenario=scenario, operators=ops, metrics=exp.metrics, passed=exp.valid,
                    origin="history", note=exp.note, source=src,
                )
        for les in know.lessons:
            targets = [t for t in (_resolve_target(kg, name, sc["name"]) for name in les.applies_to) if t]
            kg.add_lesson(
                title=les.title, trigger=les.trigger, fix=les.fix, error_code=les.error_code,
                applies_to=targets, learned_from=src,
            )
    kg.g.nodes[scenario]["constraints"] = constraints
    return kg


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _resolve_operator(kg: CapabilityKG, name: str) -> str | None:
    """按名字找算子节点，同名多版本时取可执行的最新版本。"""
    name = name.split("@")[0]
    cands = [(d.get("executable", False), d["version"], n) for n, d in kg.nodes("Operator") if d["name"] == name]
    return max(cands)[2] if cands else None


def _resolve_target(kg: CapabilityKG, name: str, scenario: str) -> str | None:
    if op := _resolve_operator(kg, name):
        return op
    if name == scenario:
        return node_id("Scenario", scenario)
    if name in CONDITION_TAGS or node_id("Condition", name) in kg.g:
        return kg.add_condition(name)
    return None


def extract_and_save(
    llm: LLM, materials: Path = DEFAULT_MATERIALS, out: Path = DEFAULT_KG_PATH, workers: int = 4
) -> tuple[CapabilityKG, ExtractionResult]:
    """一键执行：抽取 → 建图 → 保存 KG（JSON + GraphML）与原始抽取结果。"""
    result, manifest = run_extraction(llm, materials, workers)
    functions = [f for repo in manifest.get("code_repos", []) for f in scan_repo(materials / repo["path"], repo.get("name"))]
    kg = build_kg(result, manifest, functions)
    kg.save(out)
    raw = Path(out).parent / "extraction.json"
    raw.write_text(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("KG 已保存到 %s（原始抽取结果：%s）", out, raw)
    return kg, result
