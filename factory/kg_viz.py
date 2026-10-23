"""能力知识图谱可视化：用 pyvis 生成可交互的 HTML 网络图。

提供三种视图（数据本身不变，只是展示方式不同）：

- overview（概览，默认）：只画 场景 / 指标 / 算子 / 经验 四类主干节点。
  适用条件、依赖、出处折叠进算子的悬停信息；大量 ValidationRun 汇总成"算子 → 场景"的一条边
  （线越粗，验证次数越多）；只关联到条件的经验，连到它所属的场景。
- full（完整）：所有节点与边，可按节点类型过滤。
- focus（聚焦）：以某个节点为中心，只看其 N 跳邻域。

所有视图都不在边上常驻文字，关系名称在鼠标悬停时显示。
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx
from pyvis.network import Network

from factory.kg import NODE_TYPES, CapabilityKG

COLORS = {
    "Scenario": "#e8590c",
    "Operator": "#1971c2",
    "Metric": "#2f9e44",
    "Condition": "#7048e8",
    "Dependency": "#868e96",
    "ValidationRun": "#f08c00",
    "Lesson": "#c2255c",
    "Source": "#495057",
}
OPERATOR_COLORS = {"builtin": "#1971c2", "synthesized": "#0ca678", "reference": "#a5d8ff"}
SHAPES = {"Scenario": "star", "Operator": "dot", "Lesson": "diamond", "ValidationRun": "square", "Source": "triangle"}
REL_ZH = {
    "SUITABLE_WHEN": "适用于", "DEPENDS_ON": "依赖", "EVALUATED_BY": "评价指标", "VALIDATED_BY": "验证记录",
    "RUN_ON": "所属场景", "LEARNED_FROM": "经验来源", "APPLIES_TO": "经验适用于", "EXTRACTED_FROM": "抽取自",
    "DERIVED_FROM": "版本派生自", "VALIDATED_ON": "验证于", "LESSON_OF": "经验所属场景",
}
EDGE_COLORS = {"VALIDATED_ON": "#91a7ff", "APPLIES_TO": "#faa2c1", "LESSON_OF": "#faa2c1", "DERIVED_FROM": "#343a40",
               "EVALUATED_BY": "#8ce99a"}
TOOLTIP_FIELDS = ("description", "title", "trigger", "fix", "error_code", "origin", "executable", "metrics", "passed",
                  "note", "constraints", "path", "hits", "conditions", "dependencies", "sources", "history")
VIEWS = ("overview", "full")


def _label(data: dict) -> str:
    t, key = data["type"], data["key"]
    if t == "Lesson":
        return data.get("title", key)[:14]
    if t == "ValidationRun":
        ms = data.get("metrics", {})
        first = next(iter(ms.items()), None)
        return (f"{first[0]}={first[1]:.2f}" if first else "run") + ("" if data.get("passed") else " ✗")
    if t == "Source":
        return key.split("/")[-1][:24]
    return key


def _tooltip(nid: str, data: dict) -> str:
    lines = [nid]
    for f in TOOLTIP_FIELDS:
        if f in data and data[f] not in (None, "", [], {}):
            v = data[f]
            v = "、".join(map(str, v)) if isinstance(v, list) else json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else str(v)
            lines.append(f"{f}: {v[:240]}")
    return "\n".join(lines)


def _operator_color(data: dict) -> str:
    if not data.get("executable", True):
        return OPERATOR_COLORS["reference"]
    return OPERATOR_COLORS["synthesized" if data.get("origin") == "synthesized" else "builtin"]


# ---------------------------------------------------------------------------
# 视图
# ---------------------------------------------------------------------------

def subgraph(kg: CapabilityKG, *, types: list[str] | None = None, focus: str | None = None, radius: int = 1) -> nx.MultiDiGraph:
    """完整视图 / 聚焦视图：原图的子图。"""
    g = kg.g
    if focus:
        if focus not in g:
            raise KeyError(f"节点不存在：{focus}")
        g = nx.ego_graph(g.to_undirected(as_view=True), focus, radius=radius)
        g = kg.g.subgraph(g.nodes)
    if types:
        g = g.subgraph([n for n, d in g.nodes(data=True) if d["type"] in types or n == focus])
    return g


def overview_graph(kg: CapabilityKG) -> nx.MultiDiGraph:
    """概览视图：保留主干节点，把细节折叠为节点属性或汇总边。"""
    g = kg.g
    out = nx.MultiDiGraph()
    keep = {"Scenario", "Metric", "Operator", "Lesson"}
    for n, d in g.nodes(data=True):
        if d["type"] in keep:
            out.add_node(n, **d)

    def key_of(nid: str) -> str:
        return g.nodes[nid]["key"]

    for n, d in kg.nodes("Operator"):
        attrs = out.nodes[n]
        attrs["conditions"] = sorted(key_of(c) for c in kg.neighbors(n, "SUITABLE_WHEN"))
        attrs["dependencies"] = sorted(key_of(c) for c in kg.neighbors(n, "DEPENDS_ON"))
        attrs["sources"] = sorted(key_of(c) for c in kg.neighbors(n, "EXTRACTED_FROM"))
        # 验证记录汇总为 算子 → 场景
        per_scenario: dict[str, list[dict]] = defaultdict(list)
        for rid in kg.neighbors(n, "VALIDATED_BY"):
            for sid in kg.neighbors(rid, "RUN_ON"):
                per_scenario[sid].append(g.nodes[rid])
        history = []
        for sid, runs in per_scenario.items():
            passed = sum(bool(r.get("passed")) for r in runs)
            out.add_edge(n, sid, rel="VALIDATED_ON", runs=len(runs), passed=passed,
                         detail=f"{len(runs)} 次验证，通过 {passed} 次")
            history.append(f"{key_of(sid)}：{len(runs)} 次验证，通过 {passed} 次")
        attrs["history"] = history
        for parent in kg.neighbors(n, "DERIVED_FROM"):
            out.add_edge(n, parent, rel="DERIVED_FROM")

    for sid, d in kg.nodes("Scenario"):
        for mid in kg.neighbors(sid, "EVALUATED_BY"):
            edge = next(e for e in g.get_edge_data(sid, mid).values() if e["rel"] == "EVALUATED_BY")
            out.add_edge(sid, mid, rel="EVALUATED_BY", detail=f"阈值 {edge.get('threshold')}" if edge.get("threshold") else "")

    for lid, d in kg.nodes("Lesson"):
        targets = kg.neighbors(lid, "APPLIES_TO")
        out.nodes[lid]["conditions"] = sorted(key_of(t) for t in targets if g.nodes[t]["type"] == "Condition")
        main = [t for t in targets if g.nodes[t]["type"] in ("Operator", "Scenario")]
        for t in main:
            out.add_edge(lid, t, rel="APPLIES_TO")
        if not main:  # 只关联到条件的经验：连到它来源所属的场景
            for scenario in _lesson_scenarios(kg, lid):
                out.add_edge(lid, scenario, rel="LESSON_OF")
    # 没有任何关联的指标与经验（如从旧代码抽出、但没有场景使用的指标函数）在概览中省略，完整视图中仍可见
    out.remove_nodes_from([n for n, d in out.nodes(data=True) if d["type"] in ("Metric", "Lesson") and out.degree(n) == 0])
    return out


def _lesson_scenarios(kg: CapabilityKG, lid: str) -> list[str]:
    scenarios = set()
    for src in kg.neighbors(lid, "LEARNED_FROM"):
        if kg.g.nodes[src]["type"] == "ValidationRun":
            scenarios |= set(kg.neighbors(src, "RUN_ON"))
        else:
            scenarios |= {u for u, _, e in kg.g.in_edges(src, data=True)
                          if e["rel"] == "EXTRACTED_FROM" and kg.g.nodes[u]["type"] == "Scenario"}
    return sorted(scenarios)


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def build_network(kg: CapabilityKG, *, view: str = "overview", types: list[str] | None = None, focus: str | None = None,
                  radius: int = 1, height: str = "720px") -> Network:
    if view not in VIEWS:
        raise ValueError(f"未知视图 {view!r}，可选 {VIEWS}")
    g = subgraph(kg, types=types, focus=focus, radius=radius) if (focus or view == "full") else overview_graph(kg)
    net = Network(height=height, width="100%", directed=True, cdn_resources="remote", bgcolor="#ffffff")
    degree = dict(g.degree())
    for nid, data in g.nodes(data=True):
        t = data["type"]
        color = _operator_color(data) if t == "Operator" else COLORS.get(t, "#adb5bd")
        size = 12 + 2 * min(degree.get(nid, 0), 12) + (12 if t == "Scenario" else 0) + (10 if nid == focus else 0)
        # 注意不要传 group：pyvis 在节点带 group 时会丢弃 color，vis-network 改用默认调色板，与图例对不上
        net.add_node(nid, label=_label(data), title=_tooltip(nid, data), shape=SHAPES.get(t, "dot"),
                     color=color, size=size, borderWidth=3 if nid == focus else 1, font={"size": 13})
    for u, v, data in g.edges(data=True):
        rel = data["rel"]
        title = REL_ZH.get(rel, rel) + (f"：{data['detail']}" if data.get("detail") else "")
        net.add_edge(u, v, title=title, arrows="to", color=EDGE_COLORS.get(rel, "#dee2e6"),
                     width=1 + min(data.get("runs", 0), 6) if rel == "VALIDATED_ON" else 1,
                     dashes=rel == "DERIVED_FROM")
    net.set_options(json.dumps({
        "layout": {"randomSeed": 7},  # 布局可复现
        "physics": {"solver": "forceAtlas2Based",
                    "forceAtlas2Based": {"gravitationalConstant": -90, "springLength": 110, "avoidOverlap": 0.5},
                    "stabilization": {"iterations": 400}},
        "interaction": {"hover": True, "navigationButtons": True, "tooltipDelay": 120},
        "edges": {"smooth": {"type": "continuous"}},
    }))
    return net


GLYPHS = {"dot": "●", "star": "★", "diamond": "◆", "square": "■", "triangle": "▲"}


def _item(color: str, text: str, node_type: str = "Operator") -> str:
    """图例项：符号形状与图中该类节点一致（星形 = 场景，菱形 = 经验……）。"""
    glyph = GLYPHS[SHAPES.get(node_type, "dot")]
    return f'<span style="margin-right:14px;white-space:nowrap"><span style="color:{color};font-size:15px">{glyph}</span> {text}</span>'


def _edge_item(color: str, text: str, dashed: bool = False) -> str:
    line = f"border-top:3px {'dashed' if dashed else 'solid'} {color}"
    return (f'<span style="margin-right:14px;white-space:nowrap"><span style="display:inline-block;width:22px;'
            f'vertical-align:middle;{line}"></span> {text}</span>')


def legend_html(view: str = "overview", links: dict[str, str] | None = None) -> str:
    """图例：颜色与形状和图中一致。links 为 {名称: 相对路径}，用于在概览与完整视图之间跳转。"""
    if view == "overview":
        nodes = [_item(COLORS["Scenario"], "场景", "Scenario"), _item(COLORS["Metric"], "评价指标", "Metric"),
                 _item(OPERATOR_COLORS["builtin"], "内置算子"), _item(OPERATOR_COLORS["synthesized"], "生成并注册的算子"),
                 _item(OPERATOR_COLORS["reference"], "历史实现（未注册，可参考生成）"), _item(COLORS["Lesson"], "经验", "Lesson")]
        edges = [_edge_item(EDGE_COLORS["VALIDATED_ON"], "在该场景验证过（越粗次数越多）"),
                 _edge_item(EDGE_COLORS["APPLIES_TO"], "经验适用于"), _edge_item(EDGE_COLORS["EVALUATED_BY"], "场景的评价指标"),
                 _edge_item(EDGE_COLORS["DERIVED_FROM"], "新版本派生自", dashed=True)]
        note = ("概览只画主干：ValidationRun 汇总成蓝线，悬停算子可看各场景的验证次数；"
                "Condition、Dependency、Source 折叠进算子的悬停信息。所有节点与边见完整视图。")
    else:
        nodes = [_item(COLORS[t] if t != "Operator" else OPERATOR_COLORS["builtin"],
                       t + ("（内置）" if t == "Operator" else ""), t) for t in NODE_TYPES]
        nodes[1:1] = [_item(OPERATOR_COLORS["synthesized"], "Operator（生成）"),
                      _item(OPERATOR_COLORS["reference"], "Operator（历史实现）")]
        nodes.insert(1, nodes.pop(NODE_TYPES.index("Operator") + 2))  # 三种算子排在一起
        edges = []
        note = "完整视图：全部 8 类节点、9 类关系；关系名称在鼠标悬停于边上时显示。"
    nav = " | ".join(f'<a href="{href}">{name}</a>' for name, href in (links or {}).items())
    return (f'<div style="font:13px sans-serif;margin:6px 8px;line-height:1.9">'
            + (f'<div style="margin-bottom:2px"><b>视图：</b>{nav}</div>' if nav else "")
            + "".join(nodes) + ("<br>" + "".join(edges) if edges else "")
            + f'<div style="color:#666">{note}</div></div>')


def render_html(kg: CapabilityKG, out: Path | None = None, *, links: dict[str, str] | None = None, **kwargs) -> str:
    """返回完整 HTML（含图例）；给出 out 时同时写入文件。"""
    page = build_network(kg, **kwargs).generate_html(notebook=False)
    # pyvis 模板里有一个指向本地 node_modules 的无效脚本引用，去掉以免浏览器报 404
    page = re.sub(r'<script[^>]*src="\.\./node_modules[^"]*"[^>]*></script>', "", page)
    view = "full" if kwargs.get("focus") else kwargs.get("view", "overview")
    page = page.replace("<body>", "<body>" + legend_html(view, links), 1)
    if out is not None:
        Path(out).write_text(page, encoding="utf-8")
    return page


def render_views(kg: CapabilityKG, out: Path) -> tuple[Path, Path]:
    """同时写出概览（out）与完整视图（<out>_full.html），页面顶部互相链接。"""
    out = Path(out)
    full = out.with_name(out.stem + "_full.html")
    links = {"概览": out.name, "完整": full.name}
    render_html(kg, out, view="overview", links=links)
    render_html(kg, full, view="full", links=links)
    return out, full
