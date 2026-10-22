"""AlgoForge Web 界面（Streamlit）。

启动：streamlit run app.py
- 运行：输入能力描述、选择数据与 LLM 模式，实时查看各 Agent 的进度，得到验证报告与生成代码；
- 能力库：浏览算子与经验，交互式查看能力知识图谱；
- 历史运行：查看以前的运行报告（含 examples/demo 中的演示结果）。
"""

from __future__ import annotations

import base64
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from factory.kg import DEFAULT_KG_PATH, NODE_TYPES, CapabilityKG
from factory.utils import ROOT

st.set_page_config(page_title="AlgoForge 算法能力工厂", page_icon="🏭", layout="wide")

EXAMPLES = {
    "客户流失（表格二分类）": ("预测30天内可能流失的客户，需可解释，召回优先", "data/churn.csv"),
    "工单分类（中文文本多分类）": ("把客服工单自动分到网络故障、账单争议、套餐变更、销户投诉四类，宏平均 F1 不低于 0.8", "data/tickets.csv"),
}
RUN_DIRS = [ROOT / "runs", ROOT / "runs" / "demo" / "runs", ROOT / "examples" / "demo" / "runs"]


@st.cache_data(show_spinner=False)
def load_kg_dict(path: str, mtime: float) -> dict:
    return CapabilityKG.load(Path(path)).to_dict()


def get_kg(path: Path) -> CapabilityKG:
    return CapabilityKG.from_dict(load_kg_dict(str(path), path.stat().st_mtime))


def embed_html(page: str, height: int) -> None:
    """以 data: URL 嵌入 HTML：报告和图谱含 LLM 生成的文本，不让它与应用同源。"""
    url = "data:text/html;charset=utf-8;base64," + base64.b64encode(page.encode("utf-8")).decode("ascii")
    st.iframe(url, height=height)


def list_runs() -> list[Path]:
    runs = [p for d in RUN_DIRS if d.exists() for p in d.iterdir() if (p / "state.json").exists()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🏭 AlgoForge")
    st.caption("能力抽取 → 能力复刻 → 自动验证 → 能力沉淀")
    kg_path = Path(st.text_input("知识图谱路径", str(DEFAULT_KG_PATH.relative_to(ROOT))))
    kg_path = kg_path if kg_path.is_absolute() else ROOT / kg_path
    if not kg_path.exists():
        st.error("知识图谱不存在，请先运行：python -m factory.cli extract")
        st.stop()
    stats = get_kg(kg_path).stats()["nodes"]
    st.metric("算子", stats.get("Operator", 0))
    c1, c2 = st.columns(2)
    c1.metric("经验", stats.get("Lesson", 0))
    c2.metric("验证记录", stats.get("ValidationRun", 0))

tab_run, tab_kg, tab_hist = st.tabs(["🚀 运行", "🧠 能力库与知识图谱", "📜 历史运行"])


def show_report(run_dir: Path, key: str) -> None:
    """展示一次运行的结果；key 用于区分同一页面上多次展示的组件。"""
    from factory.schemas import RunState

    state = RunState.model_validate_json((run_dir / "state.json").read_text(encoding="utf-8"))
    ok = state.status == "succeeded"
    (st.success if ok else st.warning)(f"状态：{state.status}　最优方案：{state.best_plan or '-'}")
    if state.reports and state.spec:
        rows = []
        for r in state.reports:
            rows.append({"方案": ("★ " if r.plan_name == state.best_plan else "") + r.plan_name,
                         "结果": "通过" if r.passed else "未通过", **r.metrics,
                         "诊断": "；".join(f"{d.code.value} {d.message}" for d in r.diagnostics)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    files = {p.name: p for p in run_dir.iterdir()}
    tabs = st.tabs(["验证报告", "生成的代码", "运行记录"])
    with tabs[0]:
        if "report.html" in files:
            embed_html(files["report.html"].read_text(encoding="utf-8"), height=900)
            st.download_button("下载 report.md", files["report.md"].read_bytes(), "report.md", key=f"{key}-md")
    with tabs[1]:
        if "pipeline.py" in files:
            code = files["pipeline.py"].read_text(encoding="utf-8")
            st.download_button("下载 pipeline.py", code, "pipeline.py", key=f"{key}-py")
            st.code(code, language="python")
        for name in sorted(n for n in files if n.endswith(".py") and n != "pipeline.py"):
            with st.expander(f"生成算子 {name}"):
                st.code(files[name].read_text(encoding="utf-8"), language="python")
    with tabs[2]:
        st.code("\n".join(state.log))


# ---------------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------------

with tab_run:
    example = st.selectbox("示例", list(EXAMPLES))
    request = st.text_area("能力描述", EXAMPLES[example][0], height=80)
    c1, c2, c3 = st.columns([2, 1, 1])
    upload = c1.file_uploader("上传 CSV（可选，不上传则用示例数据）", type="csv")
    data_path = c1.text_input("数据路径", EXAMPLES[example][1])
    target = c2.text_input("目标列（可选）", "")
    mode = c3.selectbox("LLM 模式", ["live", "replay", "mock", "record"],
                        help="live 调用真实接口；replay 回放录制结果（离线）；mock 为规则版 Agent")
    c1, c2, c3, c4 = st.columns(4)
    n_cand = c1.number_input("候选方案数", 1, 4, 2)
    max_replans = c2.number_input("最多重新规划", 0, 3, 1)
    allow_synth = c3.checkbox("允许生成新算子", True)
    writeback = c4.checkbox("回写知识图谱", False, help="勾选后验证结果与新算子会写入上面的知识图谱")

    if st.button("开始", type="primary"):
        from factory.llm import LLMError, make_llm
        from factory.workflow import Workflow, WorkflowConfig

        if upload is not None:
            tmp = Path(tempfile.mkdtemp(prefix="algoforge-upload-")) / upload.name
            tmp.write_bytes(upload.getvalue())
            data_path = str(tmp)
        work_kg = kg_path
        if not writeback:  # 不回写时在副本上运行，避免改动知识图谱
            work_kg = Path(tempfile.mkdtemp(prefix="algoforge-kg-")) / kg_path.name
            shutil.copy(kg_path, work_kg)
            if (kg_path.parent / "operators").exists():
                shutil.copytree(kg_path.parent / "operators", work_kg.parent / "operators")
        try:
            llm = make_llm(mode)
        except LLMError as exc:
            st.error(str(exc))
            st.stop()
        with st.status("运行中…", expanded=True) as status:
            wf = Workflow(llm, kg_path=work_kg, config=WorkflowConfig(
                n_candidates=int(n_cand), max_replans=int(max_replans), allow_synthesis=allow_synth, writeback=True),
                on_event=lambda msg: status.write(msg))
            state = wf.run(request, data_path, target=target or None)
            status.update(label=f"完成：{state.status}", state="complete" if state.status == "succeeded" else "error")
        load_kg_dict.clear()
        show_report(ROOT / Path(state.artifacts["state"]).parent, key="run")

# ---------------------------------------------------------------------------
# 能力库与知识图谱
# ---------------------------------------------------------------------------

with tab_kg:
    kg = get_kg(kg_path)
    sub_ops, sub_les, sub_graph = st.tabs(["算子", "经验", "图谱"])
    with sub_ops:
        ops = [{"算子": d["key"], "类型": d["kind"], "来源": d.get("origin"), "可执行": d.get("executable"),
                "描述": d["description"], "历史验证分": round(kg.historical_score(n)[0], 3), "验证次数": kg.historical_score(n)[1]}
               for n, d in kg.nodes("Operator")]
        st.dataframe(pd.DataFrame(ops).sort_values(["类型", "算子"]), hide_index=True, width="stretch")
        q = st.text_input("检索算子（结构化过滤 + BM25 + 条件 + 历史验证分）", "处理不平衡的可解释模型")
        if q:
            hits = kg.search_operators(q, top_k=8)
            st.dataframe(pd.DataFrame([{"算子": h.key, "得分": h.score, **h.breakdown, "证据": "；".join(h.evidence)}
                                       for h in hits]), hide_index=True, width="stretch")
    with sub_les:
        les = [{"标题": d["title"], "诊断码": d.get("error_code"), "触发": d["trigger"], "修复": d["fix"],
                "来源": d.get("origin", "extracted"), "命中次数": d.get("hits", 0)} for _, d in kg.nodes("Lesson")]
        st.dataframe(pd.DataFrame(les), hide_index=True, width="stretch")
    with sub_graph:
        from factory.kg_viz import render_html

        c1, c2, c3 = st.columns([2, 2, 1])
        types = c1.multiselect("节点类型", list(NODE_TYPES), default=[t for t in NODE_TYPES if t not in ("Source", "Dependency")])
        focus = c2.selectbox("聚焦节点（可选）", [""] + sorted(n for n, _ in kg.nodes()), index=0)
        radius = c3.number_input("邻域半径", 1, 3, 1)
        page = render_html(kg, types=types or None, focus=focus or None, radius=int(radius), height="680px")
        embed_html(page, height=720)

# ---------------------------------------------------------------------------
# 历史运行
# ---------------------------------------------------------------------------

with tab_hist:
    runs = list_runs()
    if not runs:
        st.info("还没有运行记录。")
    else:
        choice = st.selectbox("选择运行", runs, format_func=lambda p: str(p.relative_to(ROOT)))
        show_report(choice, key="hist")
