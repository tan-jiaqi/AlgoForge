"""验证报告：把一次运行的 RunState 渲染为 Markdown 与 HTML。

两种格式共用同一份"块"结构（标题 / 段落 / 列表 / 表格 / 代码），保证内容一致。
报告回答评审最关心的问题：做了什么、为什么这样做（设计依据）、验证了什么、结果如何、沉淀了什么。
"""

from __future__ import annotations

import difflib
import html
import re
from dataclasses import dataclass, field
from pathlib import Path

from factory.kg import CapabilityKG
from factory.schemas import RunState, ValidationReport


@dataclass
class Block:
    kind: str  # h1 / h2 / h3 / p / ul / table / code
    text: str = ""
    items: list[str] = field(default_factory=list)
    header: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    lang: str = ""


def H(level: int, text: str) -> Block:
    return Block(f"h{level}", text)


def P(text: str) -> Block:
    return Block("p", text)


def UL(items: list[str]) -> Block:
    return Block("ul", items=items)


def T(header: list[str], rows: list[list[str]]) -> Block:
    return Block("table", header=header, rows=rows)


def CODE(text: str, lang: str = "") -> Block:
    return Block("code", text, lang=lang)


def _fmt(v: float | None, std: float | None = None) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}" + (f" ± {std:.3f}" if std else "")


def _num(v, digits: int) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def _bar(value: float, threshold: float | None, width: int = 20) -> str:
    """文本进度条：■ 为指标值，| 为阈值位置。"""
    filled = max(0, min(width, round(value * width)))
    bar = list("■" * filled + "·" * (width - filled))
    if threshold is not None:
        bar[min(width - 1, round(threshold * width))] = "|"
    return "".join(bar)


# ---------------------------------------------------------------------------
# 内容组装
# ---------------------------------------------------------------------------

def build_blocks(state: RunState, *, pipeline_code: str | None = None, kg: CapabilityKG | None = None) -> list[Block]:
    from factory.workflow import rank_reports

    spec = state.spec
    blocks: list[Block] = [H(1, "AlgoForge 验证报告")]
    status = {"succeeded": "✅ 成功", "failed": "❌ 未通过", "running": "⏳ 运行中"}[state.status]
    total = sum(state.timings.values())
    blocks.append(T(["项目", "内容"], [
        ["需求", state.request],
        ["数据", state.data_path],
        ["运行 ID", state.run_id],
        ["LLM", state.llm_model or "-"],
        ["状态", status],
        ["最优方案", state.best_plan or "-"],
        ["总耗时", f"{total:.1f} s"],
    ]))
    if spec is None:
        blocks.append(P("需求理解阶段失败，未产生 TaskSpec。"))
        return blocks + _log_blocks(state)

    ranked = rank_reports(state.reports, spec)
    best = ranked[0] if ranked else None
    primary = spec.primary_metric
    summary = (f"共验证 {len(state.reports)} 个候选方案，{sum(r.passed for r in state.reports)} 个通过；"
               f"重新规划 {state.replans} 次，修复 {state.repair_rounds} 轮。")
    if best is not None:
        summary += (f"最优方案 `{best.plan_name}` 的 {primary.name} = {_fmt(best.metrics.get(primary.name))}"
                    f"（要求 ≥ {primary.threshold}）。")
    blocks.append(P(summary))

    blocks += [H(2, "1. 需求理解（TaskSpec）"), T(["字段", "值"], [
        ["任务类型", spec.task_type.value],
        ["数据模态", spec.modality.value],
        ["目标列", spec.target_column + (f"（文本列 {spec.text_column}）" if spec.text_column else "")],
        ["评价指标", "；".join(f"{m.name} ≥ {m.threshold}" + ("（主）" if m.primary else "") for m in spec.metrics)],
        ["业务约束", "、".join(spec.constraints) or "-"],
        ["数据画像标签", "、".join(state.profile_tags) or "-"],
    ])]
    blocks += _retrieval_blocks(state)
    blocks += _comparison_blocks(state, ranked)
    blocks += _search_blocks(state)
    blocks += _best_plan_blocks(state)
    blocks += _synthesis_blocks(state)
    blocks += _verification_blocks(state, best)
    blocks += _knowledge_blocks(state, kg)
    if pipeline_code:
        blocks += [H(2, "附录 A. 生成的算法代码（pipeline.py）"), CODE(pipeline_code, "python")]
    return blocks + _log_blocks(state)


def _retrieval_blocks(state: RunState) -> list[Block]:
    r = state.retrieval
    if not r:
        return []
    out = [H(2, "2. 知识检索（设计依据）"),
           P(f"匹配场景：{r.get('scenario') or '新场景'}。每类步骤从知识图谱检索候选算子，"
             "得分 = 0.4 × 文本相关度 + 0.3 × 条件匹配 + 0.3 × 历史验证分。")]
    rows = []
    for kind, opts in r.get("operator_options", {}).items():
        for o in opts:
            b = o.get("breakdown", {})
            rows.append([kind, o["ref"], "是" if o.get("executable", True) else "否（历史实现，可参考生成）",
                         f"{o['score']:.3f}", f"{b.get('text', 0):.2f} / {b.get('condition', 0):.2f} / {b.get('history', 0):.2f}",
                         "；".join(o.get("evidence", [])) or "-"])
    out.append(T(["步骤", "候选算子", "可直接复用", "得分", "文本/条件/历史", "证据"], rows))
    lessons = r.get("lessons", [])
    if lessons:
        out += [H(3, "参考的历史经验"),
                UL([f"**{les['title']}**" + (f"（{les['error_code']}）" if les.get("error_code") else "") + f"：{les['fix']}"
                    for les in lessons])]
    if state.evidence_checks:
        out += [H(3, "知识引用检查"),
                P("检索并传入上下文不等于采纳。以下引用由模型声明，程序仅核对可见范围与结构化适用对象；不证明语义落实或指标收益。"),
                T(["方案", "步骤", "模型引用", "范围检查", "语义落实"],
                  [[e["plan"], str(e["step"] or "-"), "、".join(e["evidence_ids"]) or "无",
                    "通过" if e["scope_valid"] else "未通过：" + "；".join(e.get("errors", []))[:300], "未验证"]
                   for e in state.evidence_checks])]
    return out


def _comparison_blocks(state: RunState, ranked: list[ValidationReport]) -> list[Block]:
    spec = state.spec
    if not ranked:
        return []
    names = [m.name for m in spec.metrics]
    names += [k for k in ranked[0].metrics if k not in names]
    rows = []
    for i, r in enumerate(ranked, 1):
        rows.append([str(i), ("★ " if r.plan_name == state.best_plan else "") + r.plan_name,
                     "通过" if r.passed else "未通过",
                     *[_fmt(r.metrics.get(m), r.metric_std.get(m)) for m in names],
                     _fmt(r.resources.get("fit_seconds")), _num(r.resources.get("peak_rss_mb"), 0)])
    out = [H(2, "3. 候选方案比较"),
           P("排序规则：先看是否通过全部检查，再比主指标，再比次要指标之和，最后训练耗时短者优先。"
             "指标为多随机种子 × 分层 K 折交叉验证的均值 ± 种子间标准差。"),
           T(["排名", "方案", "结果", *names, "单次训练 (s)", "峰值内存 (MB)"], rows)]
    p = spec.primary_metric
    bars = [f"{r.plan_name:<36} {_bar(r.metrics.get(p.name, 0), p.threshold)} {_fmt(r.metrics.get(p.name))}" for r in ranked]
    out.append(CODE(f"{p.name}（| 为阈值 {p.threshold}）\n" + "\n".join(bars)))
    plans = {pl.name: pl for pl in state.candidates}
    out.append(H(3, "各方案的结构与诊断"))
    items = []
    for r in ranked:
        pl = plans.get(r.plan_name)
        chain = " → ".join((s.operator_ref or "").split("@")[0] if s.action == "reuse" else f"生成:{s.synth_spec.name}"
                           for s in pl.steps) if pl else "-"
        diag = "；".join(f"`{d.code.value}` {d.message}" for d in r.diagnostics) or "无"
        items.append(f"`{r.plan_name}`：{chain}。诊断：{diag}")
    out.append(UL(items))
    return out


def _search_blocks(state: RunState) -> list[Block]:
    """重新规划时的 Beam 搜索：每轮的父方案、扩展出的子方案、先验分与剪枝结果。"""
    if not state.search:
        return []
    metric = state.spec.primary_metric.name
    actual = {r.plan_name: r.metrics.get(metric) for r in state.reports}
    out = [H(3, "方案搜索（Beam Search）"),
           P("指标不达标时，以已验证结果排名前 k 的方案为父方案，由 Planner 扩展 k × m 个子方案；"
             "与已验证方案结构相同的直接去重，其余按先验（知识图谱中该场景的历史验证记录 + 本次运行的观测，"
             f"平滑后按步骤加权，模型步骤权重更高）估计 {metric}，只保留前 k 个进入验证。")]
    for rnd in state.search:
        rows = [[("✔ " if c.selected else "") + c.plan, c.chain, _fmt(c.prior),
                 _fmt(actual.get(c.plan)) if c.selected else "-", c.reason] for c in rnd.candidates]
        out += [P(f"第 {rnd.round} 轮：父方案 {', '.join(f'`{n}`' for n in rnd.parents) or '-'}，beam 宽度 {rnd.width}。"),
                T(["子方案", "结构", f"先验 {metric}", f"实际 {metric}", "结果"], rows)]
        provenance = [[c.plan, step, o.get("id", "本次运行观测"), str(o["value"])]
                      for c in rnd.candidates for step, obs in c.observations.items() for o in obs]
        if provenance:
            out.append(T(["子方案", "步骤", "先验证据来源", "数值"], provenance))
    return out


def _best_plan_blocks(state: RunState) -> list[Block]:
    plan = next((p for p in state.candidates if p.name == state.best_plan), None)
    if plan is None:
        return []
    rows = [[str(i), s.kind.value,
             s.operator_ref if s.action == "reuse" else f"生成 {s.synth_spec.name}",
             ", ".join(f"{k}={v}" for k, v in s.params.items()) or "-", s.rationale or "-"]
            for i, s in enumerate(plan.steps, 1)]
    return [H(2, f"4. 最优方案：{plan.name}"), P(plan.rationale or "-"),
            T(["#", "步骤", "算子", "参数", "选择理由"], rows)]


def _synthesis_blocks(state: RunState) -> list[Block]:
    if not state.generated and not state.repairs:
        return [H(2, "5. 能力生成与修复"), P("本次运行所有步骤均复用能力库中已有算子，未生成新算子。")]
    out = [H(2, "5. 能力生成与修复")]
    rows = [[g.key, g.spec.kind.value, g.spec.purpose, g.spec.reference or "-", str(g.revision),
             "是" if g.key in state.registered else "否"] for g in state.generated.values()]
    out.append(T(["生成算子", "类型", "用途", "参考的历史实现", "修复轮数", "注册到能力库"], rows))
    for rec in state.repairs:
        mark = {True: "✅ 已修复", False: "❌ 未修复", None: "-"}[rec.fixed]
        d = rec.diagnostic
        out += [H(3, f"修复 {rec.operator} 第 {rec.revision} 轮：{mark}"),
                UL([f"诊断：`{d.code.value}`（{d.stage}）{d.message}", f"提示：{d.hint or '-'}",
                    f"参考经验：{len(rec.lessons)} 条"])]
        diff = "\n".join(difflib.unified_diff(rec.before.splitlines(), rec.after.splitlines(),
                                              "修复前", "修复后", lineterm="", n=1))
        if diff:
            out.append(CODE(diff, "diff"))
    return out


def _verification_blocks(state: RunState, best: ValidationReport | None) -> list[Block]:
    cfg = state.verify_config
    out = [H(2, "6. 验证机制"),
           T(["级别", "检查内容", "本次配置"], [
               ["1 静态检查", "语法、import 白名单、禁止危险调用、接口与 meta 一致性", "仅针对生成的算子"],
               ["2 沙箱执行", "独立子进程，限制超时与内存，清除含 KEY/TOKEN 的环境变量",
                f"超时 {_num(cfg.get('timeout_seconds'), 0)} s，内存 {_num(cfg.get('memory_mb'), 0)} MB"],
               ["3 契约检查", "不修改输入、行数一致、不引入 NaN/Inf、概率合法、能处理未见类别与缺失值",
                f"冒烟样本 {cfg.get('smoke_rows', '-')} 行"],
               ["4 指标检查", "阈值、种子间稳定性、疑似泄漏、耗时与内存",
                f"{cfg.get('cv_folds', '-')} 折 × 种子 {cfg.get('seeds', '-')}；std 上限 {cfg.get('max_metric_std', '-')}"],
           ])]
    if best is not None and best.resources:
        res = best.resources
        out.append(P(f"最优方案资源消耗：单次训练 {_fmt(res.get('fit_seconds'))} s，"
                     f"预测 {_num(res.get('predict_ms_per_1k'), 1)} ms / 千行，峰值内存 {_num(res.get('peak_rss_mb'), 0)} MB。"))
    if state.history:
        out.append(P(f"修复前的 {len(state.history)} 次验证结果保留在 state.json 的 history 中。"))
    return out


def _knowledge_blocks(state: RunState, kg: CapabilityKG | None) -> list[Block]:
    def title(lid: str) -> str:
        node = kg.g.nodes.get(lid) if kg is not None else None
        return f"{node['title']}（{node.get('error_code') or '-'}）" if node else lid

    return [H(2, "7. 知识沉淀"), UL([
        f"回写验证记录 ValidationRun：{len(state.kg_writes)} 条",
        f"注册到能力库的新算子：{'、'.join(state.registered) or '无'}",
        f"新提炼的经验：{'；'.join(title(x) for x in state.lessons_learned) or '无'}",
        f"本次提供的经验：{len(state.lessons_used)} 条（累计检索命中，不代表采纳）",
    ])]


def _log_blocks(state: RunState) -> list[Block]:
    timing = "，".join(f"{k} {v:.1f}s" for k, v in state.timings.items())
    blocks = [H(2, "附录 B. 运行记录"), P(f"各阶段耗时：{timing or '-'}"), CODE("\n".join(state.log))]
    if state.llm_calls:
        blocks += [H(3, "LLM 调用来源"), T(["阶段", "请求哈希", "来源", "本次耗时(s)", "录制耗时(s)", "tokens"], [
            [c["tag"], c["request_hash"], c["source"], _fmt(c.get("seconds")), _fmt(c.get("recorded_seconds")),
             str((c.get("usage") or {}).get("total_tokens", "未提供"))] for c in state.llm_calls])]
    return blocks


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def _md_cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", "<br>")


def to_markdown(blocks: list[Block]) -> str:
    out = []
    for b in blocks:
        if b.kind.startswith("h"):
            out.append("#" * int(b.kind[1]) + " " + b.text)
        elif b.kind == "p":
            out.append(b.text)
        elif b.kind == "ul":
            out.append("\n".join(f"- {i}" for i in b.items))
        elif b.kind == "table":
            lines = ["| " + " | ".join(map(_md_cell, b.header)) + " |", "|" + " --- |" * len(b.header)]
            lines += ["| " + " | ".join(map(_md_cell, row)) + " |" for row in b.rows]
            out.append("\n".join(lines))
        elif b.kind == "code":
            out.append(f"```{b.lang}\n{b.text}\n```")
    return "\n\n".join(out) + "\n"


def _inline(text: str) -> str:
    """HTML 转义后，再把 **粗体** 与 `代码` 还原为标签。"""
    s = html.escape(str(text))
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", s)


CSS = """
:root { --fg:#1f2328; --muted:#59636e; --bg:#ffffff; --line:#d1d9e0; --code:#f6f8fa; --accent:#0969da; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6edf3; --muted:#9198a1; --bg:#0d1117; --line:#3d444d; --code:#151b23; --accent:#4493f8; }
}
body { font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; color:var(--fg);
       background:var(--bg); max-width: 1100px; margin: 0 auto; padding: 24px 16px; line-height: 1.6; }
h1 { border-bottom: 2px solid var(--line); padding-bottom: 8px; }
h2 { border-bottom: 1px solid var(--line); padding-bottom: 4px; margin-top: 32px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; display: block; overflow-x: auto; }
th, td { border: 1px solid var(--line); padding: 6px 10px; text-align: left; vertical-align: top; }
th { background: var(--code); }
pre { background: var(--code); padding: 12px; overflow-x: auto; border-radius: 6px; font-size: 13px; }
code { font-family: ui-monospace, "SFMono-Regular", Consolas, monospace; }
p code, li code, td code { background: var(--code); padding: 1px 4px; border-radius: 4px; }
"""


def to_html(blocks: list[Block], title: str = "AlgoForge 验证报告") -> str:
    out = []
    for b in blocks:
        if b.kind.startswith("h"):
            out.append(f"<{b.kind}>{_inline(b.text)}</{b.kind}>")
        elif b.kind == "p":
            out.append(f"<p>{_inline(b.text)}</p>")
        elif b.kind == "ul":
            out.append("<ul>" + "".join(f"<li>{_inline(i)}</li>" for i in b.items) + "</ul>")
        elif b.kind == "table":
            head = "".join(f"<th>{_inline(h)}</th>" for h in b.header)
            body = "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>" for row in b.rows)
            out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
        elif b.kind == "code":
            out.append(f"<pre><code>{html.escape(b.text)}</code></pre>")
    return (f"<!doctype html>\n<html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head>\n<body>\n"
            + "\n".join(out) + "\n</body></html>\n")


def write_report(state: RunState, out_dir: Path, *, kg: CapabilityKG | None = None) -> dict[str, str]:
    """在 out_dir 下写出 report.md 与 report.html，返回文件路径。"""
    out_dir = Path(out_dir)
    code_path = out_dir / "pipeline.py"
    code = code_path.read_text(encoding="utf-8") if code_path.exists() else None
    blocks = build_blocks(state, pipeline_code=code, kg=kg)
    md, page = out_dir / "report.md", out_dir / "report.html"
    md.write_text(to_markdown(blocks), encoding="utf-8")
    page.write_text(to_html(blocks), encoding="utf-8")
    return {"report_md": str(md), "report_html": str(page)}
