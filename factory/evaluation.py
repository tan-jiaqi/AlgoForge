"""真实调用验收。每个 job 在新进程中执行，record 与严格 replay 共用同一实验清单。"""

from __future__ import annotations

import ast
import difflib
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from factory.kg import DEFAULT_KG_PATH, CapabilityKG
from factory.llm import LLMSettings, TrialLLM, make_llm
from factory.profile import profile
from factory.schemas import MetricTarget, Modality, RunState, TaskSpec, TaskType
from factory.utils import ROOT
from factory.workflow import Workflow, WorkflowConfig, _Ctx, best_report
from factory.agents.planner import resolve_scenario

DEFAULT_EVIDENCE = ROOT / "examples" / "evaluation"
REQUEST = "预测30天内可能流失的客户，需可解释，召回优先，召回率不低于0.75"
# 搜索实验的门槛：首轮常见方案（平衡权重逻辑回归）约 0.778，略低于门槛，需要按诊断调参才能达到；
# 提高正类权重可达 0.85 左右。0.99 在正类占 27% 的数据上几乎只能靠全判正类达到，搜索无从改进。
SEARCH_THRESHOLD = 0.80
SEARCH_REQUEST = "预测客户流失，召回率不低于0.8；优先尝试可解释模型，未达标时探索不同模型或参数。"
PROBES = [
    REQUEST + "；将 total_charges 的空白字符串转换并填补为合理数值。",
    REQUEST + "；正确处理累计费用的空白字符串，并增加在网时长分桶特征。",
    REQUEST + "；增加累计费用与月费的比值特征，正确处理缺失和零分母。",
]


def task(request: str, threshold: float = 0.75) -> TaskSpec:
    return TaskSpec(description=request, task_type=TaskType.BINARY_CLASSIFICATION, modality=Modality.TABULAR,
                    target_column="churn", metrics=[MetricTarget(name="recall", threshold=threshold)],
                    constraints=["interpretable", "recall_first"])


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_state(out: Path, name: str) -> RunState:
    return RunState.model_validate_json((out / name / "runs" / name / "state.json").read_text(encoding="utf-8"))


def inject_nan(source: str) -> str:
    """在真实源码 transform 返回前重新数值转换，明确注入未填补 NaN；只用于验收。"""
    tree = ast.parse(source)
    method = next(n for c in tree.body if isinstance(c, ast.ClassDef)
                  for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "transform")
    returns = [n for n in method.body if isinstance(n, ast.Return) and isinstance(n.value, ast.Name)]
    if not returns:
        raise ValueError("injection requires a direct DataFrame return")
    ret = returns[-1]
    name = ret.value.id
    # 只使用验证器白名单中的 pandas 调用；恢复原始脏列，产生应由契约检查发现的新 NaN。
    code = ast.parse(f'{name}["total_charges"] = pd.to_numeric(X["total_charges"].astype(str).str.strip(), errors="coerce")').body
    # 在第一句复制前保留输入列，否则原 transform 已经填补会使注入无效。
    snapshot = ast.parse('_evaluation_raw = X["total_charges"].copy()').body[0]
    code[0].value.args[0] = ast.parse('_evaluation_raw.astype(str).str.strip()', mode="eval").body
    method.body.insert(0, snapshot)
    method.body[method.body.index(ret):method.body.index(ret)] = code
    if not any(isinstance(n, ast.Import) and any(a.name == "pandas" and a.asname == "pd" for a in n.names) for n in tree.body):
        tree.body.insert(0, ast.Import(names=[ast.alias(name="pandas", asname="pd")]))
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


class InjectedWorkflow(Workflow):
    """固定已经真实生成并通过完整验证的种子方案，只独立重复诊断与真实 LLM 修复。"""

    def __init__(self, *args, seed: RunState, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed

    def _plan(self, state: RunState, ctx: _Ctx) -> str:
        state.spec = self.seed.spec.model_copy(deep=True)
        ctx.scenario = resolve_scenario(self.kg, state.data_path)
        ctx.profile = profile(state.data_path, state.spec.target_column)
        ctx.planning = self.planner.retrieve(state.spec, ctx.profile, scenario=ctx.scenario, top_k=100)
        state.profile_tags = ctx.profile.tags
        state.retrieval = ctx.planning.to_prompt()
        state.candidates = [next(p for p in self.seed.candidates if p.name == self.seed.best_plan).model_copy(deep=True)]
        state.generated = {s.synth_spec.name: self.seed.generated[s.synth_spec.name].model_copy(update={"revision": 0})
                           for s in state.candidates[0].steps if s.action == "synthesize"}
        from factory.verify import load_verify_config
        state.verify_config = load_verify_config(state.spec.task_type).model_dump()
        state.note("[PLAN] 受控注入实验：固定种子运行的真实生成方案与源码，不重新规划或生成；本次独立调用真实 LLM 修复。")
        return "BUILD"

    def _build(self, state: RunState, ctx: _Ctx) -> str:
        plan = state.candidates[0]
        original_report = self.verifier.verify(plan, state.spec, state.data_path, state.generated)
        if not original_report.passed:
            raise ValueError("seed failed full validation before injection")
        candidates = [g for g in state.generated.values() if "total_charges" in g.source and "def transform" in g.source]
        if not candidates:
            raise ValueError("no generated total_charges transformer in seed")
        gen = candidates[0]
        faulty = inject_nan(gen.source)
        state.generated[gen.spec.name] = gen.model_copy(update={"source": faulty})
        digest = lambda s: hashlib.sha256(s.encode()).hexdigest()
        dump(self.runs_dir.parent / "injection.json", {
            "kind": "controlled_nan_injection", "seed_run": self.seed.run_id,
            "operator": gen.key, "original": gen.source, "injected": faulty,
            "original_sha256": digest(gen.source), "injected_sha256": digest(faulty),
            "patch": "\n".join(difflib.unified_diff(gen.source.splitlines(), faulty.splitlines(), "original", "injected", lineterm="")),
            "original_validation": original_report.model_dump(mode="json"),
        })
        state.note(f"[BUILD] 已验证原始源码后注入 NaN：{gen.key}；原始源码和补丁见 injection.json")
        return "VERIFY"


def search_intervention(wf: Workflow, state: RunState, out: Path) -> None:
    """固定真实子方案池与本次观测，仅移除 KG 历史证据，不再次请求模型。"""
    from factory import search
    results = []
    from factory.agents.planner import PlanSet
    from factory.llm import extract_json

    def plan_set(call):
        record = json.loads((wf.llm.directory / f"plan-{call['request_hash']}.json").read_text(encoding="utf-8"))
        try:
            return PlanSet.model_validate(extract_json(record["response"])).plans
        except ValueError:
            return []

    plan_calls = [c for c in state.llm_calls if c["tag"] == "plan"]
    for rnd in state.search:
        # Planner 的原始有效回复保留所有子方案（包括未入选者），不根据最终结果反推候选池。
        # 多轮搜索时按子方案名找到该轮的回复（重名方案在工作流中被加了 _r<轮次> 后缀）。
        names = {re.sub(rf"_r{rnd.round}$", "", c.plan) for c in rnd.candidates}
        plans = next((ps for ps in map(plan_set, reversed(plan_calls)) if {p.name for p in ps} == names), None)
        if plans is None:
            continue
        children = [wf.planner.bind(p, state.spec) for p in plans[:rnd.width * max(wf.cfg.beam_expand, 1)]]
        parent_names = set(rnd.parents)
        parent_state = state.model_copy(update={"reports": [r for r in state.reports if r.plan_name in parent_names]})
        observed = search.run_observations(parent_state)
        seen = {search.signature(p) for p in state.candidates if p.name in parent_names}
        arms = {}
        for mode, scenario in (("full", resolve_scenario(wf.kg, state.data_path)), ("without_history", None)):
            kept, records = search.select(children, seen=seen, width=rnd.width, spec=state.spec,
                                          kg=wf.kg, scenario=scenario, observed=observed)
            arms[mode] = {"selected": [p.name for p in kept], "candidates": [r.model_dump() for r in records]}
        results.append({"round": rnd.round, "arms": arms,
                        "selection_changed": set(arms["full"]["selected"]) != set(arms["without_history"]["selected"])})
    dump(out / "intervention.json", results)


def worker(out: Path, evidence: Path, mode: str, job: dict) -> None:
    name = job["name"]
    directory = out / name
    kg_path = directory / "kg" / "capability_kg.json"
    kg_path.parent.mkdir(parents=True, exist_ok=True)
    if job["group"] == "reuse":
        shutil.copytree(out / job["seed"] / "kg", kg_path.parent, dirs_exist_ok=True)
    else:
        shutil.copy(DEFAULT_KG_PATH, kg_path)
    request = job["request"]
    cfg = WorkflowConfig(n_candidates=1, max_replans=0, max_repairs=3, retrieval_k=100,
                         knowledge_mode=job.get("knowledge_mode", "full"), task_spec=task(request))
    if job["group"] in {"probe", "repair_natural", "reuse"}:
        cfg.n_candidates = 3
    if job["group"] == "search":
        cfg.task_spec = task(request, SEARCH_THRESHOLD)
        cfg.max_replans, cfg.beam_width, cfg.beam_expand = 2, 2, 2
        # 必须保持搜索时的 KG 快照不被本次结果污染。
        cfg.writeback = False
    settings = LLMSettings.load()
    settings = settings.model_copy(update={k: job[k] for k in ("model", "temperature") if k in job})
    llm = make_llm(mode, settings=settings, cassette_dir=evidence / "cassettes" / name)
    if job.get("nonce"):
        llm = TrialLLM(llm, job["nonce"])
    kwargs = {"kg_path": kg_path, "runs_dir": directory / "runs", "config": cfg,
              "on_event": lambda message: print(message, flush=True)}
    wf = InjectedWorkflow(llm, seed=read_state(out, job["seed"]), **kwargs) if job["group"] == "repair_injected" else Workflow(llm, **kwargs)
    state = wf.run(request, "data/churn.csv", target="churn", run_id=name)
    if job["group"] == "search":
        # Curator 即使 no-writeback 也会改变内存图，恢复运行前的历史。
        wf.kg = CapabilityKG.load(kg_path)
        search_intervention(wf, state, directory)


def usable_seed(state: RunState) -> bool:
    plan = next((p for p in state.candidates if p.name == state.best_plan), None)
    return state.status == "succeeded" and bool(plan) and any(
        s.action == "synthesize" and "total_charges" in state.generated[s.synth_spec.name].source
        and "def transform" in state.generated[s.synth_spec.name].source for s in plan.steps)


def complete_repair(state: RunState) -> bool:
    """只有已修复算子进入通过的最优方案并注册，才算完整的修复沉淀链。"""
    return state.status == "succeeded" and any(r.fixed is True and r.operator in state.registered for r in state.repairs)


def run_evaluation(mode: str, out: Path, evidence: Path = DEFAULT_EVIDENCE, repeats: int = 3, *, resume: bool = False,
                   temperature: float | None = None) -> dict:
    if mode not in {"record", "replay"}:
        raise ValueError("evaluation supports record/replay only; no mock fallback")
    out, evidence = Path(out).resolve(), Path(evidence).resolve()
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if out == evidence or out in evidence.parents or evidence in out.parents:
        raise ValueError("output and evidence directories must be separate and must not contain one another")
    if resume and mode != "record":
        raise ValueError("resume is only supported for an interrupted record run")
    if out.exists() and any(out.iterdir()) and not resume:
        raise ValueError("output directory must be empty; use a new directory to preserve independent trials")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "repeats": repeats, "data": "data/churn.csv", "jobs": [],
                "environment": environment()}
    results = []
    if resume:
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        results = json.loads((out / "results.json").read_text(encoding="utf-8"))
        repeats = manifest["repeats"]
    settings = LLMSettings.load()
    if temperature is not None:
        # 低温度下相同提示词几乎总得到相同回复，重复运行就不再是独立样本；录制时可调高。
        settings = settings.model_copy(update={"temperature": temperature})

    def execute(job):
        if any(r["name"] == job["name"] for r in results):
            return read_state(out, job["name"]) if (out / job["name"] / "runs" / job["name"] / "state.json").exists() else None
        existing = next((j for j in manifest["jobs"] if j["name"] == job["name"]), None)
        job = existing or {"model": settings.model, "temperature": settings.temperature, **job}
        if not any(j["name"] == job["name"] for j in manifest["jobs"]):
            manifest["jobs"].append(job)
        dump(out / "manifest.json", manifest)
        if mode == "record":
            dump(evidence / "manifest.json", manifest)
        directory = out / job["name"]
        directory.mkdir(parents=True, exist_ok=True)
        dump(directory / "job.json", job)
        with (directory / "execution.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run([sys.executable, "-m", "factory.evaluation", str(out), str(evidence), mode, job["name"]],
                                  cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            state = read_state(out, job["name"])
            row = measurement(job, state)
        except (OSError, ValueError):
            state = None
            row = {**job, "status": "worker_error", "calls": 0}
        row["returncode"] = proc.returncode
        results.append(row)
        dump(out / "results.json", results)
        print(f"{job['name']}: {row['status']}", flush=True)
        return state

    if mode == "replay":
        recorded = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
        for job in recorded["jobs"]:
            execute(job)
    else:
        if (evidence / "manifest.json").exists() and not resume:
            raise ValueError("recording namespace already exists; choose a new --evidence directory")
        for i in range(repeats):
            for condition in ("full", "no_lessons", "catalog"):
                execute({"name": f"kg_{condition}_{i+1}", "group": "kg", "trial": i+1,
                         "knowledge_mode": condition, "request": REQUEST})
        probes = []
        for i, request in enumerate(PROBES, 1):
            job = {"name": f"probe_{i}", "group": "probe", "trial": i, "request": request}
            probes.append((job, execute(job)))
        natural = next(((j, s) for j, s in probes if s and s.repairs), None)
        repaired = []
        if natural:
            for i in range(repeats):
                job = {"name": f"repair_natural_{i+1}", "group": "repair_natural", "trial": i+1, "request": natural[0]["request"]}
                repaired.append((job, execute(job)))
        if not any(s and complete_repair(s) for _, s in repaired):
            seed = next(((j, s) for j, s in probes if s and usable_seed(s)), None)
            if seed:
                for i in range(repeats):
                    job = {"name": f"repair_injected_{i+1}", "group": "repair_injected", "trial": i+1,
                           "request": seed[0]["request"], "seed": seed[0]["name"], "nonce": f"injected-{i+1}"}
                    repaired.append((job, execute(job)))
        for j, s in repaired:
            if s and s.status == "succeeded" and s.repairs:
                execute({"name": j["name"] + "_reuse", "group": "reuse", "trial": j["trial"], "seed": j["name"],
                         "nonce": f"reuse-{j['name']}",
                         "request": "预测下个月可能流失的客户，要能解释原因，尽量不漏掉流失客户，召回率不低于0.75"})
        for i in range(repeats):
            execute({"name": f"search_{i+1}", "group": "search", "trial": i+1, "request": SEARCH_REQUEST,
                     "nonce": f"search-{i+1}"})
        dump(evidence / "manifest.json", manifest)
    results = finalize(out)
    if mode == "record":
        export_evidence(out, evidence)
    return evaluation_outcome(results)


def evaluation_outcome(results: list[dict]) -> dict:
    """预期的指标失败允许保留；执行错误或缺失修复复用证据使 CLI 验收失败。"""
    parents = {r["name"]: r for r in results if r.get("complete_repair_chain")}
    chains = [r for r in results if r["group"] == "reuse" and r["status"] == "succeeded"
              and set(r.get("best_reused", [])) & set(parents.get(r.get("seed"), {}).get("repaired_registered", []))]
    return {"jobs": len(results), "worker_errors": sum(r["status"] == "worker_error" or r.get("returncode", 0) != 0 for r in results),
            "execution_errors": sum(bool(r.get("execution_error")) for r in results),
            "repair_reuse_chains": len(chains), "required_evidence_missing": not chains or not any(r.get("search_rounds") for r in results)}


def environment() -> dict:
    return {"python": sys.version, "packages": {name: importlib.metadata.version(name) for name in
            ("numpy", "pandas", "scikit-learn", "lightgbm", "pydantic", "networkx", "openai")},
            "data_sha256": hashlib.sha256((ROOT / "data/churn.csv").read_bytes()).hexdigest(),
            "base_kg_sha256": hashlib.sha256(DEFAULT_KG_PATH.read_bytes()).hexdigest()}


def finalize(out: Path) -> list[dict]:
    """从原始状态生成统计，不改变请求、响应、源码或验证结果。"""
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    old = {r["name"]: r for r in json.loads((out / "results.json").read_text(encoding="utf-8"))}
    rows = []
    for job in manifest["jobs"]:
        try:
            state = read_state(out, job["name"])
            rows.append({**measurement(job, state), "returncode": old.get(job["name"], {}).get("returncode", 0)})
        except (OSError, ValueError):
            rows.append(old.get(job["name"], {**job, "status": "worker_error", "calls": 0, "returncode": 1}))
    manifest.setdefault("environment", environment())
    dump(out / "manifest.json", manifest)
    dump(out / "results.json", rows)
    write_summary(out, rows)
    return rows


def export_evidence(out: Path, evidence: Path) -> None:
    """导出可提交的原始证据并修正产物链接；录制响应不作改写。"""
    from factory.report import write_report
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for job in manifest["jobs"]:
        name = job["name"]
        shutil.copytree(out / name, evidence / name, dirs_exist_ok=True)
        state_file = evidence / name / "runs" / name / "state.json"
        if state_file.exists():
            state = RunState.model_validate_json(state_file.read_text(encoding="utf-8"))
            directory = state_file.parent
            def relative(path):
                try:
                    return str(path.relative_to(ROOT))
                except ValueError:
                    return str(path)
            state.artifacts = {k: relative(directory / Path(v).name) for k, v in state.artifacts.items()}
            state_file.write_text(state.model_dump_json(indent=2), encoding="utf-8")
            write_report(state, directory, kg=CapabilityKG.load(evidence / name / "kg" / "capability_kg.json"))
    for filename in ("manifest.json", "results.json", "SUMMARY.md", "SUMMARY.html"):
        shutil.copy(out / filename, evidence / filename)


def measurement(job: dict, state: RunState) -> dict:
    report = best_report(state)
    return {**job, "status": state.status, "repairs": len(state.repairs),
            "execution_error": any(c["source"] in {"missing_recording", "remote_error"} for c in state.llm_calls)
                               or any("] 出错：" in entry for entry in state.log),
            "original_diagnostic_resolved": sum(r.fixed is True for r in state.repairs),
            "complete_repair_chain": complete_repair(state),
            "repaired_registered": sorted({r.operator for r in state.repairs if r.fixed is True and r.operator in state.registered}),
            "registered": state.registered, "lessons_learned": state.lessons_learned,
            "reused": sorted({s.operator_ref for p in state.candidates for s in p.steps if s.action == "reuse"}),
            "best_reused": sorted({s.operator_ref for p in state.candidates if p.name == state.best_plan
                                    for s in p.steps if s.action == "reuse"}),
            "best_plan": state.best_plan, "metrics": report.metrics if report else {},
            "selected_steps": [s.model_dump(mode="json") for p in state.candidates if p.name == state.best_plan for s in p.steps],
            "calls": len(state.llm_calls), "remote_calls": sum(c["source"] == "remote" for c in state.llm_calls),
            "seconds": sum(state.timings.values()),
            "tokens": sum((c.get("usage") or {}).get("total_tokens", 0) for c in state.llm_calls),
            "recorded_llm_seconds": sum(c.get("recorded_seconds") or c.get("seconds", 0) for c in state.llm_calls),
            "search_rounds": len(state.search), "log": state.log if state.status == "failed" else []}


def write_summary(out: Path, results: list[dict]) -> None:
    from factory.report import H, P, T, to_html, to_markdown
    blocks = [H(1, "真实 LLM 机制验收"),
              P("每组默认三次独立进程运行，固定 data/churn.csv、TaskSpec 和 5 折 × 3 种子验证。样本量小，比例只描述本次观测。失败和未触发分支全部保留；注入与自然缺陷分别统计。"),
              T(["运行", "类型", "状态", "修复", "recall", "调用数", "远端调用", "tokens", "耗时(s)"], [
                  [r["name"], r["group"], r["status"], str(r.get("repairs", "-")),
                   str(round(r.get("metrics", {}).get("recall", 0), 4)), str(r["calls"]),
                   str(r.get("remote_calls", 0)), str(r.get("tokens", 0)), str(round(r.get("seconds", 0), 1))] for r in results]),
              P("完整证据：manifest.json、results.json；每个运行目录含 execution.log、runs/<运行名>/state.json、报告和源码。注入运行另外保留 injection.json；搜索运行保留 intervention.json。"),
              P("KG 引用通过范围检查不代表语义落实。KG 消融可能不改变规划或指标；搜索干预仅比较固定子方案池的打分和入选集合，不能证明搜索收益。回放重做训练、验证和知识回写，不请求远端模型。")]
    for group in dict.fromkeys(r.get("knowledge_mode") if r["group"] == "kg" else r["group"] for r in results):
        rows = [r for r in results if (r.get("knowledge_mode") if r["group"] == "kg" else r["group"]) == group]
        blocks.append(P(f"{group}：完整方案通过 {sum(r['status'] == 'succeeded' for r in rows)}/{len(rows)}；"
                        f"触发修复 {sum(bool(r.get('repairs')) for r in rows)}/{len(rows)}；"
                        f"修复算子通过且注册 {sum(bool(r.get('complete_repair_chain')) for r in rows)}/{len(rows)}。"))
    paired = []
    for trial in sorted({r["trial"] for r in results if r["group"] == "kg"}):
        arms = {r["knowledge_mode"]: r for r in results if r["group"] == "kg" and r["trial"] == trial}
        if set(arms) != {"full", "no_lessons", "catalog"}:
            continue
        def behavior(row):
            return [(s["kind"], s["action"], s["operator_ref"], s["params"],
                     tuple((s["synth_spec"] or {}).get(k) for k in ("name", "kind", "reference")))
                    for s in row.get("selected_steps", [])]
        paired.append([str(trial), str(behavior(arms["full"]) != behavior(arms["no_lessons"])),
                       str(behavior(arms["full"]) != behavior(arms["catalog"]))])
    if paired:
        blocks += [H(2, "KG 配对观察"), T(["重复", "去掉 Lesson 后最优候选结构/参数变化", "仅目录时最优候选结构/参数变化"], paired),
                   P("比较忽略方案名称、理由和证据引用。包含生成规格的结构差异仍可能来自模型随机性；三次观察不建立普遍因果结论，也不代表所有业务约束已得到执行。")]
    interventions = []
    for row in results:
        file = out / row["name"] / "intervention.json"
        if file.exists():
            for item in json.loads(file.read_text(encoding="utf-8")):
                interventions.append([row["name"], str(item["round"]), ", ".join(item["arms"]["full"]["selected"]),
                                      ", ".join(item["arms"]["without_history"]["selected"]), str(item["selection_changed"])])
    if interventions:
        blocks += [H(2, "固定搜索子方案池的历史证据干预"),
                   T(["运行", "轮次", "有历史时入选", "无历史时入选", "选择变化"], interventions),
                   P("干预保持候选源码/结构、本次观测、阈值和宽度一致。评分变化但选择不变时，只能证明历史证据参与计算；未执行被剪枝方案，不能据此推断搜索提高了最终指标。")]
    (out / "SUMMARY.md").write_text(to_markdown(blocks), encoding="utf-8")
    (out / "SUMMARY.html").write_text(to_html(blocks), encoding="utf-8")


if __name__ == "__main__":
    root, recordings, llm_mode, job_name = sys.argv[1:]
    job_data = json.loads((Path(root) / job_name / "job.json").read_text(encoding="utf-8"))
    worker(Path(root), Path(recordings), llm_mode, job_data)
