"""沙箱的运行时防线：在子进程内用审计钩子（PEP 578 ``sys.addaudithook``）拦截越权操作。

静态检查只能按名字拦截，总有漏网的写法（例如 pandas / numpy 自带的文件读写函数）；
这一层在**运行时**按"做了什么"拦截，与写法无关：

- 文件：只允许读 Python 安装目录、``factory/`` 包、本次的数据文件和沙箱临时目录，
  只允许在临时目录内写。仓库根目录下的 ``.env`` 等文件一律不可读；
- 网络：禁止一切 socket 操作（连接、解析域名、建监听等）；
- 进程：禁止创建子进程、执行外部命令、``fork``、向其他进程发信号；
- 动态库：禁止加载新的动态库（``ctypes.dlopen``）。

被拦截的操作抛出 ``SandboxViolation``（``PermissionError`` 的子类）。如果调用栈中有生成的
算子代码（文件名以 ``<generated`` 开头），还会记入 ``VIOLATIONS``：即使生成代码自己捕获了异常，
验证结果也会被判为 ``E_UNSAFE`` 并定位到该算子。只由第三方库自身触发的拦截（如 loky 调用
``lscpu`` 探测核数）不记为违规，库会自行回退。

说明：审计钩子是进程内的防线，配合静态检查与进程级资源限制使用；它不能代替容器 / gVisor
等操作系统级隔离（README 的扩展方向中有说明）。
"""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path

from factory.utils import ROOT


class SandboxViolation(PermissionError):
    """沙箱拦截的越权操作。"""


#: 由生成代码触发的越权记录：{"operator": 类名或 None, "event": 审计事件, "detail": 说明}
VIOLATIONS: list[dict] = []

# 一律拒绝的事件（按前缀匹配）
_DENY_PREFIXES = (
    "socket.", "subprocess.", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty",
    "os.kill", "os.killpg", "os.startfile", "pty.", "ctypes.dlopen", "webbrowser.", "urllib.", "http.client.",
    "ftplib.", "smtplib.", "poplib.", "imaplib.", "nntplib.", "telnetlib.", "winreg.", "msvcrt.", "_winapi.",
)
# 只读的路径事件：参数 0 为路径
_READ_EVENTS = {"os.listdir", "os.scandir", "glob.glob", "glob.glob/2"}
# 会改动文件系统的事件：这些参数位置上的路径都必须在临时目录内
_WRITE_EVENTS = {
    "os.remove": (0,), "os.rmdir": (0,), "os.mkdir": (0,), "os.rename": (0, 1), "os.chmod": (0,), "os.chown": (0,),
    "os.link": (0, 1), "os.symlink": (0, 1), "os.truncate": (0,), "os.utime": (0,), "os.chflags": (0,),
    "os.lchmod": (0,), "os.lchflags": (0,), "shutil.rmtree": (0,), "shutil.copyfile": (0, 1),
    "shutil.copymode": (0, 1), "shutil.copystat": (0, 1), "shutil.copytree": (0, 1), "shutil.move": (0, 1),
    "shutil.chown": (0,), "shutil.make_archive": (0,), "shutil.unpack_archive": (0, 1),
}
# 第三方库运行时会读取的系统信息（CPU 核数、cgroup 配额、内存），只读
_SYSTEM_READ = ("/proc/self", "/proc/cpuinfo", "/proc/meminfo", "/proc/stat", "/sys/fs/cgroup", "/sys/devices/system",
                "/dev/null", "/dev/urandom", "/etc/localtime", "/usr/share/zoneinfo")

_state: dict = {"installed": False, "read": (), "write": (), "files": frozenset()}


def _norm(path) -> str:
    return os.path.normcase(os.path.realpath(os.fsdecode(path)))


def _under(path: str, roots: tuple[str, ...]) -> bool:
    return any(path == r or path.startswith(r.rstrip(os.sep) + os.sep) for r in roots)


def _culprit() -> tuple[bool, str | None]:
    """调用栈中是否有生成的算子代码；有则返回其所在类名（模块级代码为 None）。"""
    frame = sys._getframe(2)
    while frame is not None:
        code = frame.f_code
        if code.co_filename.startswith("<generated"):
            qual = getattr(code, "co_qualname", code.co_name)
            return True, (None if qual == "<module>" else qual.split(".")[0])
        frame = frame.f_back
    return False, None


def _deny(event: str, detail: str) -> None:
    generated, operator = _culprit()
    if generated:
        VIOLATIONS.append({"operator": operator, "event": event, "detail": detail})
    raise SandboxViolation(f"沙箱禁止 {event}：{detail}")


def _check_read(event: str, path) -> None:
    if isinstance(path, int) or path is None:  # 已打开的文件描述符
        return
    p = _norm(path)
    if p in _state["files"] or _under(p, _state["read"]) or _under(p, _state["write"]):
        return
    _deny(event, f"读取 {os.fsdecode(path)}")


def _check_write(event: str, path) -> None:
    if isinstance(path, int) or path is None:
        return
    if not _under(_norm(path), _state["write"]):
        _deny(event, f"写入 {os.fsdecode(path)}")


def _hook(event: str, args: tuple) -> None:
    if event == "open":
        path, mode = args[0], args[1] or "r"
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        writing = any(c in str(mode) for c in "wax+") or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        (_check_write if writing else _check_read)(event, path)
    elif event in _READ_EVENTS:
        _check_read(event, args[0] if args else None)
    elif event in _WRITE_EVENTS:
        for i in _WRITE_EVENTS[event]:
            if i < len(args):
                _check_write(event, args[i])
    elif event.startswith(_DENY_PREFIXES):
        _deny(event, f"{event}（算子不允许访问网络、启动进程或加载动态库）")


def prewarm() -> None:
    """安装钩子前先导入重型依赖、填好它们的缓存，避免它们在钩子下做不必要的探测。"""
    import lightgbm  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import scipy  # noqa: F401
    import sklearn.ensemble  # noqa: F401
    import sklearn.linear_model  # noqa: F401

    try:  # loky 第一次探测物理核数时会调用 lscpu，结果有缓存
        from joblib.externals.loky import cpu_count

        cpu_count(only_physical_cores=True)
    except Exception:  # noqa: BLE001 - 探测失败只影响并行度
        pass
    import factory.operators  # noqa: F401
    import factory.verify.checks  # noqa: F401


def install(*, workdir: str | Path, readable_files: list[str | Path] = ()) -> None:
    """安装审计钩子（每个进程只能装一次，装上后无法卸载）。"""
    if _state["installed"]:
        return
    prefixes = {sys.prefix, sys.base_prefix, sys.exec_prefix, *site.getsitepackages()}
    try:
        prefixes.add(site.getusersitepackages())
    except AttributeError:
        pass
    read = [*prefixes, *(p for p in sys.path if p and os.path.isdir(p) and not _is_repo_root(p)),
            str(ROOT / "factory"), *(_SYSTEM_READ if os.name != "nt" else ())]
    _state.update(
        installed=True,
        read=tuple(_norm(p) for p in read),
        write=(_norm(workdir),),
        files=frozenset(_norm(f) for f in readable_files),
    )
    sys.addaudithook(_hook)


def _is_repo_root(path: str) -> bool:
    """PYTHONPATH 指向仓库根目录：不能因此放行根目录下的 .env、kg/ 等文件，只放行 factory/。"""
    return _norm(path) == _norm(ROOT)
