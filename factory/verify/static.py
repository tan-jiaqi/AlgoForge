"""第一级：静态检查（不执行代码）。

- 语法：能否解析；
- 安全：只允许白名单内的 import；禁止 eval/exec/open 等危险函数（调用或取用都不行）、双下划线名字与属性；
  禁止 pandas / numpy / sklearn 等库自带的文件读写、网络下载和字符串求值入口（如 ``pd.read_csv``、
  ``np.load``、``df.to_pickle``、``df.query``）；factory 内部只允许导入算子基类与枚举；
- 接口：必须定义继承 Operator 的类，带 ``meta``，实现 fit 与 transform / predict。

按名字拦截总有遗漏，运行时还有第二道防线：沙箱子进程内的审计钩子（guard.py）按行为拦截。
"""

from __future__ import annotations

import ast
import re

from factory.schemas import Diagnostic, DiagnosticCode, OperatorKind

ALLOWED_MODULES = {
    "__future__", "math", "re", "typing", "dataclasses", "collections", "itertools", "functools",
    "numpy", "pandas", "scipy", "sklearn", "lightgbm",
}
#: factory 内部只允许导入这些名字（例如 factory.operators.synthesized 能执行任意源码，不能放行）
ALLOWED_FACTORY = {
    "factory.operators": {"Operator", "OperatorMeta"},
    "factory.operators.base": {"Operator", "OperatorMeta"},
    "factory.schemas": {"OperatorKind", "Modality"},
}
FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "open", "input", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "breakpoint", "exit", "quit",
}
SAFE_DUNDERS = {"__init__", "__name__", "__class__"}
#: 不允许出现的属性名 / 导入名 / 模块路径中的一段：文件读写、网络、字符串求值、可通往 os 等模块的入口
FORBIDDEN_ATTRS = {
    # numpy
    "load", "save", "savez", "savez_compressed", "savetxt", "loadtxt", "genfromtxt", "fromfile", "tofile",
    "fromregex", "memmap", "open_memmap", "DataSource", "npyio", "ctypeslib", "f2py", "distutils",
    # pandas
    "HDFStore", "ExcelWriter", "ExcelFile", "eval", "query", "io",
    # sklearn / scipy / lightgbm
    "datasets", "externals", "save_model", "load_model", "model_from_string", "dump", "dumps", "loads",
    # typing：会对字符串求值
    "get_type_hints", "ForwardRef", "_evaluate", "make_dataclass",
    # 不应经由其他模块的属性间接拿到的标准库模块
    "os", "sys", "subprocess", "shutil", "pathlib", "builtins", "importlib", "socket", "urllib", "pickle",
    "ctypes", "glob", "tempfile", "platform", "signal", "threading", "multiprocessing", "inspect", "gc",
    "marshal", "runpy", "pty", "requests", "http", "ftplib", "webbrowser", "system", "popen",
}
#: 按前缀 / 模式拦截的属性名：pandas 的 read_* / to_<文件格式>，sklearn 的 fetch_* 与 svmlight 读写
FORBIDDEN_ATTR_PATTERN = re.compile(
    r"^(read_\w+|fetch_\w+|(load|dump)_svmlight\w*|"
    r"to_(csv|pickle|parquet|json|excel|html|sql|hdf|feather|stata|clipboard|markdown|latex|xml|orc|gbq))$"
)
#: 会读取文件的关键字参数（如 lightgbm.Booster(model_file=...)）
FORBIDDEN_KEYWORDS = {"model_file", "filepath_or_buffer", "path_or_buf", "fname"}
#: 格式化字符串中的属性访问，如 "{0.__class__.__init__.__globals__}".format(x)
FORMAT_DUNDER = re.compile(r"\{[^{}]*(\.|\[)\s*_")


def _diag(code: DiagnosticCode, message: str, operator: str | None, hint: str = "") -> Diagnostic:
    return Diagnostic(code=code, stage="static", message=message, operator=operator, hint=hint)


def _attr_forbidden(name: str) -> bool:
    return name in FORBIDDEN_ATTRS or bool(FORBIDDEN_ATTR_PATTERN.match(name))


def _import_problem(module: str, names: list[str] | None = None) -> str | None:
    """返回导入不被允许的原因；names 为 from ... import 的名字（None 表示 import module）。"""
    parts = module.split(".")
    if parts[0] == "factory":
        allowed = ALLOWED_FACTORY.get(module)
        if names is None or allowed is None:
            return f"factory 内部只允许 {', '.join(f'from {m} import ...' for m in ALLOWED_FACTORY)}"
        bad = [n for n in names if n not in allowed]
        return f"{module} 只允许导入 {sorted(allowed)}，不允许 {bad}" if bad else None
    if parts[0] not in ALLOWED_MODULES:
        return f"只允许导入 {sorted(ALLOWED_MODULES)} 与 factory 的算子基类"
    bad = [p for p in parts[1:] + (names or []) if p == "*" or _attr_forbidden(p)]
    return f"不允许导入 {bad}（文件读写、网络或通配导入）" if bad else None


def check_source(source: str, *, class_name: str | None = None, kind: OperatorKind | None = None) -> list[Diagnostic]:
    """对生成的算子源码做静态检查，返回诊断列表（为空表示通过）。"""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [_diag(DiagnosticCode.SYNTAX_ERROR, f"line {exc.lineno}: {exc.msg}", class_name)]

    diags: list[Diagnostic] = []

    def bad(message: str, hint: str = "") -> None:
        diags.append(_diag(DiagnosticCode.UNSAFE_CODE, message, class_name, hint))

    io_hint = "算子只能处理传入的 DataFrame / 数组：不要读写文件、访问网络或对字符串求值"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if (why := _import_problem(alias.name)):
                    bad(f"禁止导入 {alias.name}：{why}")
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                bad("禁止相对导入")
            elif (why := _import_problem(node.module, [a.name for a in node.names])):
                bad(f"禁止导入 from {node.module}：{why}")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_CALLS:
                bad(f"禁止使用 {node.id}", f"沙箱不允许 {sorted(FORBIDDEN_CALLS)}；访问可选属性请用 try/except AttributeError")
            elif node.id.startswith("__") and node.id not in SAFE_DUNDERS:
                bad(f"禁止使用名字 {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr not in SAFE_DUNDERS:
                bad(f"禁止访问属性 {node.attr}", "不允许访问 __class__/__init__/__name__ 以外的双下划线属性")
            elif _attr_forbidden(node.attr):
                bad(f"禁止访问属性 {node.attr}", io_hint)
        elif isinstance(node, ast.keyword) and node.arg in FORBIDDEN_KEYWORDS:
            bad(f"禁止使用参数 {node.arg}=", io_hint)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and FORMAT_DUNDER.search(node.value):
            bad("禁止在格式化字符串中访问属性", "格式化字符串里不要写 {x.attr} / {x[...]} 形式的字段")
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
    mismatch = _params_mismatch(cls)
    if mismatch:
        diags.append(_diag(DiagnosticCode.BAD_INTERFACE, f"meta.params 与 __init__ 默认值不一致：{mismatch}", cls.name,
                           "实例化时 meta.params 会覆盖 __init__ 默认值，两处必须一致"))
    return diags


def _params_mismatch(cls: ast.ClassDef) -> list[str]:
    """比较 meta = OperatorMeta(params={...}) 字面量与 __init__ 的默认值（只比较能字面求值的部分）。"""
    meta_params: dict | None = None
    for node in cls.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "meta" for t in node.targets):
            call = node.value
            for kw in getattr(call, "keywords", []):
                if kw.arg == "params":
                    try:
                        meta_params = ast.literal_eval(kw.value)
                    except ValueError:
                        return []
    init = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
    if not meta_params or init is None:
        return []
    args = init.args.args[-len(init.args.defaults):] if init.args.defaults else []
    defaults = {}
    for a, d in zip(args, init.args.defaults):
        try:
            defaults[a.arg] = ast.literal_eval(d)
        except ValueError:
            pass
    return [f"{k}: meta={v!r}, __init__={defaults[k]!r}" for k, v in meta_params.items() if k in defaults and defaults[k] != v]
