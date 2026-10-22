"""第一级：静态检查（不执行代码）。

- 语法：能否解析；
- 安全：只允许白名单内的 import，禁止 eval/exec/open 等危险调用和 dunder 属性访问；
- 接口：必须定义继承 Operator 的类，带 ``meta``，实现 fit 与 transform / predict。
"""

from __future__ import annotations

import ast

from factory.schemas import Diagnostic, DiagnosticCode, OperatorKind

ALLOWED_MODULES = {
    "__future__", "math", "re", "typing", "dataclasses", "collections", "itertools", "functools",
    "numpy", "pandas", "scipy", "sklearn", "lightgbm",
}
ALLOWED_FACTORY = ("factory.operators", "factory.schemas")
FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "open", "input", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "breakpoint", "exit", "quit",
}
SAFE_DUNDERS = {"__init__", "__name__", "__class__"}


def _diag(code: DiagnosticCode, message: str, operator: str | None, hint: str = "") -> Diagnostic:
    return Diagnostic(code=code, stage="static", message=message, operator=operator, hint=hint)


def _module_allowed(name: str) -> bool:
    return name.split(".")[0] in ALLOWED_MODULES or name.startswith(ALLOWED_FACTORY)


def check_source(source: str, *, class_name: str | None = None, kind: OperatorKind | None = None) -> list[Diagnostic]:
    """对生成的算子源码做静态检查，返回诊断列表（为空表示通过）。"""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [_diag(DiagnosticCode.SYNTAX_ERROR, f"line {exc.lineno}: {exc.msg}", class_name)]

    diags: list[Diagnostic] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad = [a.name for a in node.names if not _module_allowed(a.name)]
        elif isinstance(node, ast.ImportFrom):
            bad = [] if node.level == 0 and node.module and _module_allowed(node.module) else [node.module or "."]
        else:
            bad = []
        if bad:
            diags.append(_diag(DiagnosticCode.UNSAFE_CODE, f"禁止导入：{bad}", class_name,
                               f"只允许导入 {sorted(ALLOWED_MODULES)} 与 factory.operators"))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            diags.append(_diag(DiagnosticCode.UNSAFE_CODE, f"禁止调用 {node.func.id}()", class_name))
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr not in SAFE_DUNDERS:
            diags.append(_diag(DiagnosticCode.UNSAFE_CODE, f"禁止访问属性 {node.attr}", class_name))
    if diags:
        return diags

    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if class_name:
        classes = [c for c in classes if c.name == class_name]
    if not classes:
        return [_diag(DiagnosticCode.BAD_INTERFACE, f"未找到类 {class_name or '<Operator 子类>'}", class_name)]
    cls = classes[0]
    attrs = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    attrs |= {t.id for n in cls.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    attrs |= {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    need = ["meta", "fit"] + (["predict", "predict_proba"] if kind == OperatorKind.MODEL else ["transform"])
    missing = [a for a in need if a not in attrs]
    if missing:
        diags.append(_diag(DiagnosticCode.BAD_INTERFACE, f"{cls.name} 缺少 {missing}", cls.name,
                           "算子需继承 Operator，定义 meta = OperatorMeta(...) 并实现 sklearn 接口"))
    return diags
