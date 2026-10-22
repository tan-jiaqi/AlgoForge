"""命令行入口：llm-ping / operators / extract / kg / run / report。"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from factory.llm import LLMError, LLMSettings, OpenAILLM, make_llm

app = typer.Typer(help="AlgoForge —— 算法能力工厂", no_args_is_help=True)
kg_app = typer.Typer(help="查看与检索能力知识图谱", no_args_is_help=True)
app.add_typer(kg_app, name="kg")
console = Console()

LLM_MODE_HELP = "mock / live（带缓存）/ record（录制到 examples/cassettes）/ replay（仅回放，离线）"


@app.command("llm-ping")
def llm_ping() -> None:
    """检查 LLM 连通性并列出可用模型（不会打印 API Key）。"""
    settings = LLMSettings.load()
    console.print(f"endpoint: {settings.base_url}  model: {settings.model}  key: {'set' if settings.api_key else 'MISSING'}")
    try:
        models = OpenAILLM(settings).list_models()
    except LLMError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    table = Table("available models")
    for m in models:
        table.add_row(m + ("  [green]<- configured[/green]" if m == settings.model else ""))
    console.print(table)


@app.command("operators")
def list_operators() -> None:
    """列出已注册的算子。"""
    from factory.operators import registry

    table = Table("key", "kind", "modalities", "description")
    for meta in registry.list():
        table.add_row(meta.key, meta.kind.value, ",".join(m.value for m in meta.modalities), meta.description)
    console.print(table)


@app.command()
def run(
    request: str = typer.Argument(..., help="自然语言能力描述，如：预测30天内可能流失的客户，需可解释，召回优先"),
    data: Path = typer.Option(..., "--data", help="CSV 数据路径"),
    target: str = typer.Option(None, help="目标列（不填则由 LLM 根据需求和数据推断）"),
    llm: str = typer.Option("live", help=LLM_MODE_HELP),
    candidates: int = typer.Option(2, help="候选方案数"),
    max_replans: int = typer.Option(1, help="指标不达标时最多重新规划的次数"),
    kg: Path = typer.Option(None, help="知识图谱路径（默认 kg/capability_kg.json）"),
    runs_dir: Path = typer.Option(None, help="运行产物目录（默认 runs/）"),
    no_writeback: bool = typer.Option(False, "--no-writeback", help="不把验证结果写回 KG"),
    no_synthesis: bool = typer.Option(False, "--no-synthesis", help="只复用能力库中已有算子，不生成新算子"),
    max_repairs: int = typer.Option(3, help="每个生成算子最多修复轮数"),
) -> None:
    """端到端：需求理解 → 知识检索 → 方案规划（复用/生成）→ 自动验证 → 修复优化 → 知识沉淀。"""
    from factory.kg import DEFAULT_KG_PATH
    from factory.workflow import DEFAULT_RUNS_DIR, Workflow, WorkflowConfig

    cfg = WorkflowConfig(n_candidates=candidates, max_replans=max_replans, max_repairs=max_repairs,
                         allow_synthesis=not no_synthesis, writeback=not no_writeback)
    wf = Workflow(make_llm(llm), kg_path=kg or DEFAULT_KG_PATH, runs_dir=runs_dir or DEFAULT_RUNS_DIR, config=cfg)
    with console.status("运行中…"):
        state = wf.run(request, str(data), target=target)

    spec = state.spec
    if spec:
        console.rule("TaskSpec")
        console.print(f"{spec.task_type.value} / {spec.modality.value}，目标列 [bold]{spec.target_column}[/bold]，"
                      f"指标 {[f'{m.name}>={m.threshold}' + ('*' if m.primary else '') for m in spec.metrics]}，"
                      f"约束 {spec.constraints}")
        console.print(f"画像标签：{state.profile_tags}")
    if state.reports:
        console.rule("候选方案验证结果")
        names = [m.name for m in spec.metrics]
        names += [k for k in state.reports[0].metrics if k not in names][: max(0, 3 - len(names))]
        table = Table("方案", "结果", *names, "fit(s)")
        for r in state.reports:
            table.add_row(
                ("★ " if r.plan_name == state.best_plan else "") + r.plan_name,
                "[green]通过[/green]" if r.passed else "[red]未通过[/red]",
                *[f"{r.metrics[k]:.3f}±{r.metric_std.get(k, 0):.3f}" if k in r.metrics else "-" for k in names],
                str(r.resources.get("fit_seconds", "-")),
            )
        console.print(table)
        for r in state.reports:
            for d in r.diagnostics:
                console.print(f"  [yellow]{r.plan_name}[/yellow] {d.code.value} {d.message}" + (f"（{d.hint}）" if d.hint else ""))
    plan = next((p for p in state.candidates if p.name == state.best_plan), None)
    if plan:
        console.rule(f"最优方案：{plan.name}")
        console.print(plan.rationale)
        for i, step in enumerate(plan.steps):
            op = step.operator_ref if step.action == "reuse" else f"[magenta]生成 {step.synth_spec.name}[/magenta]"
            console.print(f"  {i + 1}. [{step.kind.value}] {op} {step.params or ''} —— {step.rationale}")
    if state.generated or state.registered or state.lessons_learned:
        console.rule("能力生成与沉淀")
        for name, gen in state.generated.items():
            console.print(f"  生成 {gen.key}（修复 {gen.revision} 轮）：{gen.spec.purpose}"
                          + (f"，参考 {gen.spec.reference}" if gen.spec.reference else ""))
        for rec in state.repairs:
            mark = {True: "[green]已修复[/green]", False: "[red]未修复[/red]", None: "-"}[rec.fixed]
            console.print(f"  修复 {rec.operator} r{rec.revision}：{rec.diagnostic.code.value} {rec.diagnostic.message[:60]} → {mark}")
        console.print(f"  注册到能力库：{state.registered or '无'}；新增经验：{len(state.lessons_learned)} 条；"
                      f"使用经验：{len(state.lessons_used)} 条")
    console.rule("运行记录")
    for line in state.log:
        console.print("  " + line)
    console.print(f"状态：[bold]{state.status}[/bold]")
    for name, path in state.artifacts.items():
        console.print(f"  {name}: {path}")
    if state.status != "succeeded":
        raise typer.Exit(1)


@app.command()
def report(
    run_dir: Path = typer.Argument(..., help="运行目录（含 state.json），如 runs/20261022-120000-abcd"),
    kg: Path = typer.Option(None, help="知识图谱路径，用于显示经验标题"),
) -> None:
    """根据 state.json 重新生成验证报告（report.md / report.html）。"""
    from factory.kg import DEFAULT_KG_PATH, CapabilityKG
    from factory.report import write_report
    from factory.schemas import RunState

    run_dir = run_dir.parent if run_dir.suffix == ".json" else run_dir
    state = RunState.model_validate_json((run_dir / "state.json").read_text(encoding="utf-8"))
    kg_path = kg or DEFAULT_KG_PATH
    paths = write_report(state, run_dir, kg=CapabilityKG.load(kg_path) if kg_path.exists() else None)
    for name, path in paths.items():
        console.print(f"{name}: {path}")


@app.command()
def demo(
    llm: str = typer.Option("replay", help=LLM_MODE_HELP),
    out: Path = typer.Option(Path("runs/demo"), help="输出目录（examples/demo 为提交到仓库的参考结果）"),
) -> None:
    """端到端演示：跨场景迁移与修复 → 生成并注册新算子 → 复用已沉淀的算子。"""
    from factory.demo import run_demo

    with console.status("运行演示（3 个任务）…"):
        run_demo(make_llm(llm), out)
    console.print((out / "SUMMARY.md").read_text(encoding="utf-8"))


@app.command()
def extract(
    llm: str = typer.Option("replay", help=LLM_MODE_HELP),
    materials: Path = typer.Option(None, help="业务材料目录（含 manifest.yaml）"),
    out: Path = typer.Option(None, help="KG 输出路径（.json，同时导出 .graphml）"),
    workers: int = typer.Option(4, help="并发 LLM 调用数"),
) -> None:
    """从历史代码与业务文档抽取能力，构建知识图谱。"""
    from factory.extract import DEFAULT_MATERIALS, extract_and_save
    from factory.kg import DEFAULT_KG_PATH

    kg, result = extract_and_save(make_llm(llm), materials or DEFAULT_MATERIALS, out or DEFAULT_KG_PATH, workers)
    _print_stats(kg.stats())
    if result.errors:
        console.print(f"[yellow]{len(result.errors)} 项抽取失败：[/yellow]")
        for k, v in result.errors.items():
            console.print(f"  {k}: {v}")


def _print_stats(stats: dict) -> None:
    table = Table("类别", "类型", "数量")
    for group, label in (("nodes", "节点"), ("edges", "边")):
        for t, n in stats[group].items():
            table.add_row(label, t, str(n))
    console.print(table)


@kg_app.command("stats")
def kg_stats(path: Path = typer.Option(None, help="KG 文件路径")) -> None:
    """统计各类节点与边的数量。"""
    from factory.kg import DEFAULT_KG_PATH, CapabilityKG

    _print_stats(CapabilityKG.load(path or DEFAULT_KG_PATH).stats())


@kg_app.command("search")
def kg_search(
    query: str,
    kind: str = typer.Option(None, help="算子类型过滤，如 model / imputer"),
    condition: list[str] = typer.Option([], "--condition", "-c", help="期望的适用条件，可多次指定"),
    scenario: str = typer.Option(None, help="按场景计算历史验证分"),
    metric: str = typer.Option(None, help="历史验证分使用的指标"),
    path: Path = typer.Option(None, help="KG 文件路径"),
) -> None:
    """检索算子：结构化过滤 + BM25 + 条件重合 + 历史验证分。"""
    from factory.kg import DEFAULT_KG_PATH, CapabilityKG
    from factory.schemas import OperatorKind

    kg = CapabilityKG.load(path or DEFAULT_KG_PATH)
    hits = kg.search_operators(
        query, kind=OperatorKind(kind) if kind else None, conditions=condition, scenario=scenario, metric=metric
    )
    table = Table("算子", "来源", "可执行", "总分", "文本/条件/历史", "证据")
    for h in hits:
        b = h.breakdown
        table.add_row(
            h.key, h.attrs["origin"], "是" if h.attrs.get("executable") else "否", f"{h.score:.3f}",
            f"{b['text']:.2f}/{b['condition']:.2f}/{b['history']:.2f}", "；".join(h.evidence),
        )
    console.print(table)


@kg_app.command("viz")
def kg_viz(
    out: Path = typer.Option(Path("kg/capability_kg.html"), help="输出 HTML 路径"),
    focus: str = typer.Option(None, help="中心节点 id，如 Operator:TotalChargesCleaner@1；只看其邻域"),
    radius: int = typer.Option(1, help="邻域半径（跳数）"),
    types: list[str] = typer.Option([], "--type", "-t", help="只显示这些节点类型，可多次指定"),
    path: Path = typer.Option(None, help="KG 文件路径"),
) -> None:
    """生成可交互的知识图谱 HTML（pyvis），浏览器打开即可拖拽、缩放、悬停查看属性。"""
    from factory.kg import DEFAULT_KG_PATH, CapabilityKG
    from factory.kg_viz import render_html

    render_html(CapabilityKG.load(path or DEFAULT_KG_PATH), out, types=types or None, focus=focus, radius=radius)
    console.print(f"已生成 {out}")


@kg_app.command("lessons")
def kg_lessons(
    query: str = typer.Argument("", help="检索文本"),
    code: str = typer.Option(None, help="诊断码，如 E_RUNTIME"),
    path: Path = typer.Option(None, help="KG 文件路径"),
) -> None:
    """检索经验教训（按诊断码过滤 + 文本排序）。"""
    from factory.kg import DEFAULT_KG_PATH, CapabilityKG
    from factory.schemas import DiagnosticCode

    kg = CapabilityKG.load(path or DEFAULT_KG_PATH)
    for les in kg.search_lessons(query, error_code=DiagnosticCode(code) if code else None, top_k=10):
        console.print(f"[bold]{les['title']}[/bold] [{les.get('error_code') or '-'}]\n  触发：{les['trigger']}\n  修复：{les['fix']}")


if __name__ == "__main__":
    app()
