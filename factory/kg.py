"""能力知识图谱（Capability KG）：存储、检索、导出。

底层使用 NetworkX MultiDiGraph。节点 id 形如 ``"<类型>:<键>"``，例如
``Operator:LightGBMModel@1``、``Condition:imbalanced``。完整 schema 见 ``docs/schema.md``。

检索打分 = 文本相关度(BM25+，小语料下 IDF 不会为负) × W_TEXT + 适用条件重合度 × W_COND + 历史验证分 × W_HIST，
三项都归一化到 [0, 1]，并在结果中保留各项分值，便于 Planner 解释"为什么选它"。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
from rank_bm25 import BM25Plus

from factory.operators.base import OperatorMeta, OperatorRegistry
from factory.schemas import DiagnosticCode, Modality, OperatorKind, canonical_metric
from factory.utils import ROOT, get_logger

log = get_logger(__name__)

DEFAULT_KG_PATH = ROOT / "kg" / "capability_kg.json"

# ---------------------------------------------------------------------------
# Schema：节点类型与边类型
# ---------------------------------------------------------------------------

NODE_TYPES = ("Scenario", "Operator", "Metric", "Condition", "Dependency", "ValidationRun", "Lesson", "Source")

EDGE_TYPES = {
    "SUITABLE_WHEN": ("Operator", "Condition"),        # 算子在该条件下适用
    "DEPENDS_ON": ("Operator", "Dependency"),          # 算子依赖的 Python 包
    "EVALUATED_BY": ("Scenario", "Metric"),            # 场景使用的评价指标（带 threshold、primary）
    "VALIDATED_BY": ("Operator", "ValidationRun"),     # 算子参与的验证记录（带 metric、value）
    "RUN_ON": ("ValidationRun", "Scenario"),           # 验证记录所属场景
    "LEARNED_FROM": ("Lesson", "Source|ValidationRun"),  # 经验的来源
    "APPLIES_TO": ("Lesson", "Operator|Condition|Scenario"),  # 经验适用的对象
    "EXTRACTED_FROM": ("Operator|ValidationRun|Scenario|Metric", "Source"),  # 抽取出处
    "DERIVED_FROM": ("Operator", "Operator"),          # 版本谱系：新版本 → 旧版本
}

# 受控的适用条件词表：抽取时优先使用这些标签，保证检索时能对上
CONDITION_TAGS: dict[str, str] = {
    "has_missing_values": "数据存在缺失值",
    "has_categorical_features": "存在类别型特征",
    "low_cardinality": "类别基数较低（适合 one-hot）",
    "high_cardinality": "类别基数较高（适合目标编码等）",
    "numeric_features": "存在数值型特征",
    "string_typed_numeric": "数值列被读成字符串（含空白等脏值）",
    "id_columns": "数据含客户 ID 等标识列",
    "linear_model": "下游为线性模型",
    "scaled_features": "要求特征已标准化",
    "nonlinear": "存在非线性关系",
    "mixed_features": "数值与类别特征混合",
    "large_tabular": "表格数据量较大",
    "imbalanced": "正负样本不平衡",
    "recall_first": "业务上召回优先",
    "interpretable": "要求结果可解释",
    "feature_importance": "需要输出特征重要性",
    "sparse_features": "高维稀疏特征",
    "text_input": "输入为文本",
    "time_tenure": "存在在网时长等时间长度字段",
}

# 文本检索、条件重合、历史表现三项的权重
W_TEXT, W_COND, W_HIST = 0.4, 0.3, 0.3

# 诊断码相同、且标题相似度不低于该值的经验视为重复（实测重复对约 0.63，其余各对不超过 0.13）
LESSON_DUP_THRESHOLD = 0.5


def node_id(ntype: str, key: str) -> str:
    return f"{ntype}:{key}"


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def tokenize(text: str) -> list[str]:
    """中英混合分词：英文按词（拆 CamelCase 与下划线），中文按单字 + 相邻二元组。"""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text).replace("_", " ").lower()
    tokens: list[str] = re.findall(r"[a-z0-9]+", text)
    for run in re.findall(r"[一-鿿]+", text):
        tokens.extend(run)
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def text_similarity(a: str, b: str) -> float:
    """标题相似度：词元集合的 Jaccard 系数（忽略单个汉字，避免"的""导致"等字造成误判）。"""
    ta = {t for t in tokenize(a) if len(t) > 1 or t.isascii()}
    tb = {t for t in tokenize(b) if len(t) > 1 or t.isascii()}
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


@dataclass
class OperatorHit:
    """一次算子检索结果，保留分项得分与证据，供 Planner 解释选择理由。"""

    key: str
    attrs: dict[str, Any]
    score: float
    breakdown: dict[str, float]
    evidence: list[str] = field(default_factory=list)


class CapabilityKG:
    """能力知识图谱的读写与检索接口。"""

    def __init__(self, graph: nx.MultiDiGraph | None = None):
        self.g = graph if graph is not None else nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # 基础写入
    # ------------------------------------------------------------------

    def add_node(self, ntype: str, key: str, **attrs: Any) -> str:
        if ntype not in NODE_TYPES:
            raise ValueError(f"未知节点类型 {ntype!r}")
        nid = node_id(ntype, key)
        if nid in self.g:
            self.g.nodes[nid].update(attrs)
        else:
            self.g.add_node(nid, type=ntype, key=key, **attrs)
        return nid

    def add_edge(self, src: str, rel: str, dst: str, **attrs: Any) -> None:
        if rel not in EDGE_TYPES:
            raise ValueError(f"未知边类型 {rel!r}")
        for nid in (src, dst):
            if nid not in self.g:
                raise KeyError(f"节点不存在：{nid}")
        # 同一对节点之间同类型的边只保留一条，属性更新
        for k, data in self.g.get_edge_data(src, dst, default={}).items():
            if data["rel"] == rel:
                self.g.edges[src, dst, k].update(attrs)
                return
        self.g.add_edge(src, dst, rel=rel, **attrs)

    def nodes(self, ntype: str | None = None) -> list[tuple[str, dict[str, Any]]]:
        return [(n, d) for n, d in self.g.nodes(data=True) if ntype is None or d["type"] == ntype]

    def neighbors(self, nid: str, rel: str, direction: str = "out") -> list[str]:
        edges = self.g.out_edges(nid, data=True) if direction == "out" else self.g.in_edges(nid, data=True)
        return [(v if direction == "out" else u) for u, v, d in edges if d["rel"] == rel]

    # ------------------------------------------------------------------
    # 领域写入：算子、场景、验证记录、经验、来源
    # ------------------------------------------------------------------

    def add_source(self, kind: str, path: str, title: str = "") -> str:
        return self.add_node("Source", path, kind=kind, path=path, title=title or path)

    def add_condition(self, tag: str) -> str:
        return self.add_node("Condition", tag, description=CONDITION_TAGS.get(tag, tag))

    def add_operator(self, meta: OperatorMeta, *, executable: bool, source: str | None = None, code: str = "") -> str:
        """写入算子节点及其条件、依赖、出处和版本谱系边。

        executable=True 表示已注册到 OperatorRegistry、可直接复用；
        False 表示从历史代码抽取的参考实现，需要 Synthesizer 改写后才能用。
        """
        nid = self.add_node(
            "Operator",
            meta.key,
            name=meta.name,
            version=meta.version,
            kind=meta.kind.value,
            modalities=[m.value for m in meta.modalities],
            description=meta.description,
            inputs=meta.inputs,
            outputs=meta.outputs,
            params=meta.params,
            origin=meta.origin,
            executable=executable,
            code=code,
        )
        for tag in meta.conditions:
            self.add_edge(nid, "SUITABLE_WHEN", self.add_condition(tag))
        for dep in meta.dependencies:
            self.add_edge(nid, "DEPENDS_ON", self.add_node("Dependency", dep, package=dep))
        if source:
            self.add_edge(nid, "EXTRACTED_FROM", source)
        if meta.parent and node_id("Operator", meta.parent) in self.g:
            self.add_edge(nid, "DERIVED_FROM", node_id("Operator", meta.parent))
        return nid

    def add_registry(self, registry: OperatorRegistry) -> int:
        """把注册表中的所有算子同步为可执行算子节点。"""
        for meta in registry.list():
            self.add_operator(meta, executable=True)
        return len(registry)

    def add_scenario(self, name: str, *, title: str, task_type: str, modality: str, **attrs: Any) -> str:
        return self.add_node("Scenario", name, title=title, task_type=task_type, modality=modality, **attrs)

    def add_metric(self, scenario: str, metric: str, threshold: float | None = None, primary: bool = False) -> str:
        metric = canonical_metric(metric)
        mid = self.add_node("Metric", metric, name=metric)
        self.add_edge(scenario, "EVALUATED_BY", mid, threshold=threshold, primary=primary)
        return mid

    def add_validation_run(
        self,
        *,
        scenario: str,
        operators: Iterable[str],
        metrics: dict[str, float],
        passed: bool,
        origin: str,
        note: str = "",
        source: str | None = None,
        run_id: str | None = None,
    ) -> str:
        """写入一次验证记录（历史实验或本系统的验证结果），并连到相关算子。"""
        operators = list(operators)
        run_id = run_id or short_hash(json.dumps([scenario, operators, metrics, note], sort_keys=True, ensure_ascii=False))
        rid = self.add_node(
            "ValidationRun",
            run_id,
            metrics=metrics,
            passed=passed,
            origin=origin,
            note=note,
            pipeline=operators,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self.add_edge(rid, "RUN_ON", scenario)
        for op in operators:
            self.add_edge(op, "VALIDATED_BY", rid, **metrics)
        if source:
            self.add_edge(rid, "EXTRACTED_FROM", source)
        return rid

    def add_lesson(
        self,
        *,
        title: str,
        trigger: str,
        fix: str,
        error_code: DiagnosticCode | None = None,
        applies_to: Iterable[str] = (),
        learned_from: str | None = None,
    ) -> str:
        """写入一条可复用经验。applies_to 为已存在的节点 id。

        与已有经验重复（诊断码相同、标题相似）时不新建节点，而是合并到已有经验：
        补充适用对象与来源，并在 aliases 中记录被合并的标题。返回实际写入的节点 id。
        """
        code = error_code.value if error_code else None
        lid = self.find_similar_lesson(title, code)
        if lid is None:
            lid = self.add_node("Lesson", short_hash(title), title=title, trigger=trigger, fix=fix, error_code=code, hits=0)
        elif title != self.g.nodes[lid]["title"]:
            aliases = self.g.nodes[lid].setdefault("aliases", [])
            if title not in aliases:
                aliases.append(title)
        for target in applies_to:
            if target in self.g:
                self.add_edge(lid, "APPLIES_TO", target)
        if learned_from:
            self.add_edge(lid, "LEARNED_FROM", learned_from)
        return lid

    # ------------------------------------------------------------------
    # 去重与整理
    # ------------------------------------------------------------------

    def find_similar_lesson(self, title: str, error_code: str | None) -> str | None:
        best, best_sim = None, LESSON_DUP_THRESHOLD
        for nid, d in self.nodes("Lesson"):
            if d.get("error_code") != error_code:
                continue
            sim = max(text_similarity(title, t) for t in [d["title"], *d.get("aliases", [])])
            if sim >= best_sim:
                best, best_sim = nid, sim
        return best

    def merge_nodes(self, keep: str, drop: str) -> None:
        """把 drop 的所有边转移到 keep，然后删除 drop。"""
        for u, _, d in list(self.g.in_edges(drop, data=True)):
            if u != keep:
                self.add_edge(u, d["rel"], keep, **{k: v for k, v in d.items() if k != "rel"})
        for _, v, d in list(self.g.out_edges(drop, data=True)):
            if v != keep:
                self.add_edge(keep, d["rel"], v, **{k: v2 for k, v2 in d.items() if k != "rel"})
        kd, dd = self.g.nodes[keep], self.g.nodes[drop]
        if dd["type"] == "Lesson":
            kd["aliases"] = list(dict.fromkeys([*kd.get("aliases", []), dd["title"], *dd.get("aliases", [])]))
            kd["hits"] = kd.get("hits", 0) + dd.get("hits", 0)
        for k, v in dd.items():  # 补充 keep 缺少的属性（如抽取得到的指标说明）
            kd.setdefault(k, v)
        self.g.remove_node(drop)

    def tidy(self) -> list[str]:
        """整理已有图谱：合并同一指标的不同名字、合并重复经验。返回整理记录。"""
        log = []
        for mid, d in list(self.nodes("Metric")):
            canon = canonical_metric(d["name"], d.get("params"))
            if canon != d["key"]:
                target = node_id("Metric", canon)
                if target not in self.g:
                    self.add_node("Metric", canon, name=canon)
                self.merge_nodes(target, mid)
                self.g.nodes[target]["name"] = canon
                log.append(f"指标 {d['key']} → {canon}")
        lessons = sorted(self.nodes("Lesson"), key=lambda x: x[0])
        for i, (a, da) in enumerate(lessons):
            if a not in self.g:
                continue
            for b, db in lessons[i + 1 :]:
                if b in self.g and da.get("error_code") == db.get("error_code") \
                        and text_similarity(da["title"], db["title"]) >= LESSON_DUP_THRESHOLD:
                    self.merge_nodes(a, b)
                    log.append(f"经验「{db['title']}」并入「{da['title']}」")
        return log

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def operator_conditions(self, nid: str) -> set[str]:
        return {self.g.nodes[c]["key"] for c in self.neighbors(nid, "SUITABLE_WHEN")}

    def historical_score(self, nid: str, scenario: str | None = None, metric: str | None = None) -> tuple[float, int]:
        """算子在（指定场景的）通过的验证记录上的平均指标值，返回 (分数, 记录数)。"""
        values = []
        for rid in self.neighbors(nid, "VALIDATED_BY"):
            run = self.g.nodes[rid]
            if not run.get("passed"):
                continue
            if scenario and node_id("Scenario", scenario) not in self.neighbors(rid, "RUN_ON"):
                continue
            ms = run.get("metrics", {})
            if metric and metric in ms:
                values.append(ms[metric])
            elif ms:
                values.append(sum(ms.values()) / len(ms))
        return (sum(values) / len(values), len(values)) if values else (0.0, 0)

    def metric_evidence(self, nid: str, scenario: str, metric: str) -> list[float]:
        """算子在某场景上所有可信验证记录中的指标值，供方案搜索估计先验。

        与 ``historical_score`` 不同，这里**包含未通过的验证**（本系统的失败结果同样说明该算子表现如何）；
        但抽取到的历史实验只取有效的（``passed`` 为 False 表示被作废，如数据泄漏）。
        """
        return [r["value"] for r in self.metric_records(nid, scenario, metric)]

    def metric_records(self, nid: str, scenario: str, metric: str) -> list[dict]:
        """与 metric_evidence 相同的可信证据，保留可追溯的记录 ID。"""
        values = []
        for rid in self.neighbors(nid, "VALIDATED_BY"):
            run = self.g.nodes[rid]
            if run.get("origin") != "factory" and not run.get("passed"):
                continue
            if node_id("Scenario", scenario) not in self.neighbors(rid, "RUN_ON"):
                continue
            value = run.get("metrics", {}).get(metric)
            if isinstance(value, (int, float)):
                values.append({"id": rid, "metric": metric, "value": float(value), "passed": run.get("passed")})
        return values

    def search_operators(
        self,
        query: str,
        *,
        kind: OperatorKind | None = None,
        modality: Modality | None = None,
        conditions: Iterable[str] = (),
        scenario: str | None = None,
        metric: str | None = None,
        executable_only: bool = False,
        top_k: int = 5,
    ) -> list[OperatorHit]:
        """结构化过滤 → BM25 文本匹配 → 条件重合与历史验证分加权排序。"""
        cands = []
        for nid, d in self.nodes("Operator"):
            if kind and d["kind"] != kind.value:
                continue
            if modality and modality.value not in d.get("modalities", []):
                continue
            if executable_only and not d.get("executable"):
                continue
            cands.append((nid, d))
        if not cands:
            return []

        docs = [tokenize(f"{d['name']} {d['description']} {' '.join(self.operator_conditions(n))}") for n, d in cands]
        q = tokenize(query)
        raw = BM25Plus(docs).get_scores(q) if q else [0.0] * len(cands)
        top = max(max(raw), 1e-9)
        wanted = set(conditions)

        hits = []
        for (nid, d), bm in zip(cands, raw):
            conds = self.operator_conditions(nid)
            cond = len(conds & wanted) / len(wanted) if wanted else 0.0
            hist, n_runs = self.historical_score(nid, scenario, metric)
            text = max(bm, 0.0) / top
            score = W_TEXT * text + W_COND * cond + W_HIST * hist
            evidence = []
            if conds & wanted:
                evidence.append("匹配条件：" + "、".join(sorted(conds & wanted)))
            if n_runs:
                evidence.append(f"历史验证 {n_runs} 次，平均 {metric or '指标'} {hist:.3f}")
            hits.append(
                OperatorHit(
                    key=d["key"],
                    attrs=d,
                    score=round(score, 4),
                    breakdown={"text": round(text, 3), "condition": round(cond, 3), "history": round(hist, 3)},
                    evidence=evidence,
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def lesson_modalities(self, lid: str) -> set[str]:
        """经验适用的数据模态：取其适用算子与场景的模态并集；没有这类对象时为空（视为通用经验）。"""
        out: set[str] = set()
        for t in self.neighbors(lid, "APPLIES_TO"):
            d = self.g.nodes[t]
            if d["type"] == "Operator":
                out |= set(d.get("modalities", []))
            elif d["type"] == "Scenario" and d.get("modality"):
                out.add(d["modality"])
        return out

    def search_lessons(
        self,
        query: str = "",
        *,
        error_code: DiagnosticCode | None = None,
        modality: Modality | None = None,
        conditions: Iterable[str] = (),
        top_k: int = 3,
    ) -> list[dict]:
        """结构化过滤（诊断码、模态）→ 文本相关度与条件命中加权排序。

        条件命中：经验的适用条件中至少一个当前成立。不用成立比例，因为 linear_model、scaled_features
        这类条件取决于将要选的方案，规划前不会出现在画像中。
        只按文本排序时，"不平衡""ID 列"等共有标签会让其他场景的经验挤掉真正相关的经验。
        """
        lessons = [
            (n, d) for n, d in self.nodes("Lesson")
            if (error_code is None or d.get("error_code") == error_code.value)
            and (modality is None or not (mods := self.lesson_modalities(n)) or modality.value in mods)
        ]
        if not lessons:
            return []
        wanted = set(conditions)
        q = tokenize(query)
        raw = BM25Plus([tokenize(f"{d['title']} {d['trigger']} {d['fix']}") for _, d in lessons]).get_scores(q) \
            if q else [0.0] * len(lessons)
        top = max(max(raw), 1e-9)
        scored = []
        for (n, d), bm in zip(lessons, raw):
            conds = {t.split(":", 1)[1] for t in self.neighbors(n, "APPLIES_TO") if t.startswith("Condition:")}
            cond = 1.0 if conds & wanted else 0.0
            text = max(bm, 0.0) / top
            score = 0.5 * text + 0.5 * cond if wanted else text
            scored.append((score, n, d))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [{"id": n, **d} for _, n, d in scored[:top_k]]

    # ------------------------------------------------------------------
    # 持久化与导出
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, dict[str, int]]:
        nodes: dict[str, int] = {}
        for _, d in self.g.nodes(data=True):
            nodes[d["type"]] = nodes.get(d["type"], 0) + 1
        edges: dict[str, int] = {}
        for _, _, d in self.g.edges(data=True):
            edges[d["rel"]] = edges.get(d["rel"], 0) + 1
        return {"nodes": dict(sorted(nodes.items())), "edges": dict(sorted(edges.items()))}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "nodes": [{"id": n, **d} for n, d in sorted(self.g.nodes(data=True))],
            "edges": [{"source": u, "target": v, **d} for u, v, d in self.g.edges(data=True)],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapabilityKG":
        g = nx.MultiDiGraph()
        for n in data["nodes"]:
            n = dict(n)
            g.add_node(n.pop("id"), **n)
        for e in data["edges"]:
            e = dict(e)
            g.add_edge(e.pop("source"), e.pop("target"), **e)
        return cls(g)

    def save(self, path: Path = DEFAULT_KG_PATH, *, graphml: bool = True) -> Path:
        """保存为 JSON（主存储）；可选同时导出 GraphML（复杂属性序列化为 JSON 字符串）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        if graphml:
            nx.write_graphml(_graphml_safe(self.g), path.with_suffix(".graphml"))
        return path

    @classmethod
    def load(cls, path: Path = DEFAULT_KG_PATH) -> "CapabilityKG":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _graphml_safe(g: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """GraphML 只支持标量属性：列表/字典转为 JSON 字符串，None 去掉。"""

    def conv(attrs: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for k, v in attrs.items():
            if v is None:
                continue
            out[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict, tuple)) else v
        return out

    h = nx.MultiDiGraph()
    for n, d in g.nodes(data=True):
        h.add_node(n, **conv(d))
    for u, v, d in g.edges(data=True):
        h.add_edge(u, v, **conv(d))
    return h
