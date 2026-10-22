"""Synthesizer Agent：为缺失的步骤生成算子代码，并根据诊断修复。

- synthesize：SynthSpec + 任务/数据摘要 + 参考实现（KG 中抽取的历史代码）+ 相关经验 → 算子源码；
- repair：上一版源码 + 结构化诊断 + 同诊断码的经验 → 修复后的源码，只改出错的那个算子。

生成的代码不在本进程执行，交给 Verifier 先静态检查、再在沙箱子进程中运行。
"""

from __future__ import annotations

import re

from factory.agents.planner import _wrap
from factory.kg import CapabilityKG, node_id
from factory.llm import LLM
from factory.profile import DataProfile
from factory.schemas import DIAGNOSTIC_DESCRIPTIONS, Diagnostic, GeneratedOperator, SynthSpec, TaskSpec

CONTRACT = """算子代码规范：
1. 只输出一个 ```python 代码块；只定义一个类，类名必须是 {name}；
2. 类继承 (TransformerMixin, Operator)，模型类继承 (ClassifierMixin, Operator)；
   导入：from factory.operators.base import Operator, OperatorMeta；from factory.schemas import OperatorKind, Modality；
3. 类属性 meta = OperatorMeta(name="{name}", kind=OperatorKind.{kind}, version={version}, origin="synthesized",
   description=中文一句话, conditions=[适用条件标签], dependencies=[pip 包名], params={{与 __init__ 默认值一致}}{parent})；
4. __init__ 只把参数原样保存为同名属性（sklearn 约定），不做计算；fit 学到的状态用下划线结尾命名并 return self；
5. transform 必须先 X = X.copy()，不能修改输入；输入输出都是 pandas DataFrame，行数不变；
6. 本算子新产生的缺失值/无穷值要在本算子内处理；预测时可能出现未见类别和缺失值，不能报错；
7. 模型类：fit 后设置 classes_；predict_proba 返回 shape (n, n_classes) 的 numpy 数组，predict 返回 shape (n,) 的 numpy 数组；
8. 只能导入 numpy、pandas、sklearn、scipy、lightgbm、math、re 等，禁止 os/sys/subprocess/socket/open/eval/exec。"""


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip() + "\n"


class Synthesizer:
    def __init__(self, llm: LLM, kg: CapabilityKG):
        self.llm = llm
        self.kg = kg

    def _contract(self, spec: SynthSpec, version: int) -> str:
        parent = f', parent="{spec.reference}"' if spec.reference else ""
        return CONTRACT.format(name=spec.name, kind=spec.kind.name, version=version, parent=parent)

    def _reference_code(self, spec: SynthSpec) -> str | None:
        if not spec.reference:
            return None
        node = self.kg.g.nodes.get(node_id("Operator", spec.reference))
        return node.get("code") if node else None

    def synthesize(
        self, spec: SynthSpec, task: TaskSpec, profile: DataProfile, *, version: int, lessons: list[dict]
    ) -> GeneratedOperator:
        payload = {
            "synth_spec": spec.model_dump(mode="json"),
            "task": {"description": task.description, "target": task.target_column, "constraints": task.constraints},
            "data_profile": profile.summary(),
            "reference_code": self._reference_code(spec),
            "lessons": [{k: les[k] for k in ("title", "trigger", "fix")} for les in lessons],
        }
        system = (
            "你是算法能力工厂的 Synthesizer，负责按规格编写可复用的算法算子。"
            "有 reference_code 时，把它改写为符合规范的算子类；lessons 是历史踩坑经验，务必遵守。\n"
            + self._contract(spec, version)
        )
        text = self.llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": _wrap(payload)}], tag="synthesize"
        )
        return GeneratedOperator(spec=spec, source=extract_code(text), version=version)

    def repair(self, gen: GeneratedOperator, diag: Diagnostic, *, lessons: list[dict]) -> GeneratedOperator:
        payload = {
            "synth_spec": gen.spec.model_dump(mode="json"),
            "previous_code": gen.source,
            "diagnostic": {
                **diag.model_dump(mode="json"),
                "meaning": DIAGNOSTIC_DESCRIPTIONS.get(diag.code, ""),
            },
            "lessons": [{k: les[k] for k in ("title", "trigger", "fix")} for les in lessons],
        }
        system = (
            "你是算法能力工厂的 Synthesizer。previous_code 未通过验证，diagnostic 是验证器给出的结构化诊断，"
            "lessons 是同类问题的历史修复经验。只修复导致该诊断的问题，保持类名、meta 与接口不变，输出完整代码。\n"
            + self._contract(gen.spec, gen.version)
        )
        text = self.llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": _wrap(payload)}], tag="repair"
        )
        return gen.model_copy(update={"source": extract_code(text), "revision": gen.revision + 1})
