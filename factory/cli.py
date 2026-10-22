"""命令行入口：llm-ping / operators / extract / kg。run、report 在后续里程碑加入。"""

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
