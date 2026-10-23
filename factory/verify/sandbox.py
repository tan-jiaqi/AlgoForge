"""沙箱：在独立子进程中执行检查，限制超时与内存，并清除环境变量中的密钥。

说明：这是进程级隔离（资源限制 + 干净环境 + 临时工作目录），不是安全容器；
真正的第一道防线是 static.py 的白名单检查。生产环境应换成容器或 gVisor 等方案。

跨平台的内存限制：
- Linux / macOS：子进程启动前用 ``resource.setrlimit(RLIMIT_AS)`` 设置地址空间上限（由内核强制）；
- Windows：没有 ``resource`` 模块，也不支持 ``preexec_fn``，改由父进程的看门狗线程用 psutil
  轮询子进程（含其子进程）的内存，超限即终止。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from factory.schemas import Diagnostic, DiagnosticCode
from factory.utils import ROOT, get_logger

try:  # 仅 POSIX 提供
    import resource
except ImportError:  # Windows
    resource = None

try:
    import psutil
except ImportError:
    psutil = None

log = get_logger(__name__)

SECRET_PATTERN = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)

#: 是否用 RLIMIT_AS 限制内存；为 False 时使用 psutil 看门狗（Windows，或测试中强制切换）
USE_RLIMIT = resource is not None
WATCH_INTERVAL = 0.2  # 看门狗轮询间隔（秒）


def child_env(n_threads: int) -> dict[str, str]:
    """子进程环境：去掉所有疑似密钥的变量，限制线程数，统一使用 UTF-8 输出。"""
    env = {k: v for k, v in os.environ.items() if not SECRET_PATTERN.search(k)}
    env.update(
        PYTHONPATH=str(ROOT),
        PYTHONHASHSEED="0",
        PYTHONIOENCODING="utf-8",
        OMP_NUM_THREADS=str(n_threads),
        OPENBLAS_NUM_THREADS=str(n_threads),
        MKL_NUM_THREADS=str(n_threads),
        LOKY_MAX_CPU_COUNT=str(n_threads),
        MALLOC_ARENA_MAX="2",
    )
    return env


def _rlimit(memory_mb: int):
    def limit() -> None:
        mem = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))

    return limit


def _watch_memory(proc: subprocess.Popen, memory_mb: int, exceeded: threading.Event, stop: threading.Event) -> None:
    """看门狗：子进程树的常驻内存超过上限时终止它，并置位 exceeded。"""
    limit = memory_mb * 1024 * 1024
    try:
        root = psutil.Process(proc.pid)
        while proc.poll() is None and not stop.is_set():
            try:
                rss = sum(p.memory_info().rss for p in [root, *root.children(recursive=True)])
            except psutil.NoSuchProcess:
                return
            if rss > limit:
                exceeded.set()
                for p in root.children(recursive=True):
                    p.kill()
                proc.kill()
                return
            stop.wait(WATCH_INTERVAL)
    except psutil.NoSuchProcess:
        return


def run_job(job: dict, *, timeout: float, memory_mb: int, n_threads: int) -> tuple[dict | None, Diagnostic | None]:
    """返回 (检查结果, 沙箱级错误)，两者恰有一个非空。"""
    popen_kw: dict = {}
    watchdog = False
    if USE_RLIMIT:
        popen_kw["preexec_fn"] = _rlimit(memory_mb)
    elif psutil is not None:
        watchdog = True
    else:
        log.warning("未安装 psutil，当前平台无法限制沙箱内存，只限制超时")

    # Windows 上子进程被终止后文件可能仍被占用，清理失败不影响结果
    with tempfile.TemporaryDirectory(prefix="algoforge-sandbox-", ignore_cleanup_errors=True) as tmp:
        job_path, out_path = Path(tmp) / "job.json", Path(tmp) / "result.json"
        job_path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-m", "factory.verify", str(job_path), str(out_path)],
            cwd=tmp,
            env=child_env(n_threads),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **popen_kw,
        )
        exceeded, stop = threading.Event(), threading.Event()
        if watchdog:
            threading.Thread(target=_watch_memory, args=(proc, memory_mb, exceeded, stop), daemon=True).start()
        try:
            _, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return None, Diagnostic(code=DiagnosticCode.TIMEOUT, stage="sandbox",
                                    message=f"执行超过 {timeout:.0f} 秒被终止", hint="降低复杂度或检查死循环")
        finally:
            stop.set()  # 让看门狗线程退出

        if out_path.exists() and proc.returncode == 0:
            return json.loads(out_path.read_text(encoding="utf-8")), None
        tail = (stderr or "").strip().splitlines()[-5:]
        # POSIX 上被信号杀死时 returncode < 0；Windows 上以看门狗是否触发为准
        killed_for_memory = exceeded.is_set() or (USE_RLIMIT and proc.returncode < 0)
        if killed_for_memory or "MemoryError" in (stderr or ""):
            return None, Diagnostic(code=DiagnosticCode.TIMEOUT, stage="sandbox",
                                    message=f"子进程异常退出（code {proc.returncode}），可能超出内存上限 {memory_mb} MB",
                                    hint="\n".join(tail))
        return None, Diagnostic(code=DiagnosticCode.RUNTIME_ERROR, stage="sandbox",
                                message="子进程出错：" + (tail[-1] if tail else f"exit {proc.returncode}"),
                                hint="\n".join(tail))

