"""能力知识图谱可视化：用 pyvis 生成可交互的 HTML 网络图。

- 按节点类型着色、按度数设定大小，悬停显示属性；
- 可只显示部分节点类型，或以某个节点为中心只看其 N 跳邻域（例如某个算子的来源、条件、验证记录与经验）。
"""

from __future__ import annotations

import json
import re
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
SHAPES = {"Scenario": "star", "Operator": "dot", "Lesson": "diamond", "ValidationRun": "square", "Source": "triangle"}
TOOLTIP_FIELDS = ("description", "title", "trigger", "fix", "error_code", "origin", "executable", "metrics", "passed",
                  "note", "constraints", "path", "hits")


def _label(data: dict) -> str:
    t, key = data["type"], data["key"]
    if t == "Lesson":
        return data.get("title", key)[:16]
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
            v = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
            lines.append(f"{f}: {v[:200]}")
    return "\n".join(lines)


def subgraph(kg: CapabilityKG, *, types: list[str] | None = None, focus: str | None = None, radius: int = 1) -> nx.MultiDiGraph:
    g = kg.g
    if focus:
        if focus not in g:
            raise KeyError(f"节点不存在：{focus}")
        g = nx.ego_graph(g.to_undirected(as_view=True), focus, radius=radius)
        g = kg.g.subgraph(g.nodes)
    if types:
        g = g.subgraph([n for n, d in g.nodes(data=True) if d["type"] in types])
    return g


def build_network(kg: CapabilityKG, *, types: list[str] | None = None, focus: str | None = None, radius: int = 1,
                  height: str = "720px") -> Network:
    g = subgraph(kg, types=types, focus=focus, radius=radius)
    net = Network(height=height, width="100%", directed=True, cdn_resources="remote", bgcolor="#ffffff")
    degree = dict(g.degree())
    for nid, data in g.nodes(data=True):
        t = data["type"]
        executable = t != "Operator" or data.get("executable", True)
        net.add_node(
            nid, label=_label(data), title=_tooltip(nid, data), group=t, shape=SHAPES.get(t, "dot"),
            color=COLORS.get(t, "#adb5bd") if executable else "#a5d8ff",
            size=12 + 3 * min(degree.get(nid, 0), 10) + (10 if nid == focus else 0),
            borderWidth=3 if nid == focus else 1,
        )
    for u, v, data in g.edges(data=True):
        net.add_edge(u, v, label=data["rel"], title=data["rel"], arrows="to", color="#ced4da", font={"size": 9})
    net.set_options(json.dumps({
        "physics": {"barnesHut": {"gravitationalConstant": -6000, "springLength": 140}, "stabilization": {"iterations": 200}},
        "interaction": {"hover": True, "navigationButtons": True},
        "edges": {"smooth": {"type": "continuous"}},
    }))
    return net


def legend_html() -> str:
    items = "".join(
        f'<span style="display:inline-block;margin-right:12px"><span style="display:inline-block;width:10px;height:10px;'
        f'border-radius:50%;background:{COLORS[t]};margin-right:4px"></span>{t}</span>'
        for t in NODE_TYPES
    )
    extra = ('<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#a5d8ff;'
             'margin-right:4px"></span>Operator（历史实现，暂不可执行）')
    return f'<div style="font:13px sans-serif;margin:6px 0">{items}{extra}</div>'


def render_html(kg: CapabilityKG, out: Path | None = None, **kwargs) -> str:
    """返回完整 HTML（含图例）；给出 out 时同时写入文件。"""
    page = build_network(kg, **kwargs).generate_html(notebook=False)
    # pyvis 模板里有一个指向本地 node_modules 的无效脚本引用，去掉以免浏览器报 404
    page = re.sub(r'<script[^>]*src="\.\./node_modules[^"]*"[^>]*></script>', "", page)
    page = page.replace("<body>", "<body>" + legend_html(), 1)
    if out is not None:
        Path(out).write_text(page, encoding="utf-8")
    return page
