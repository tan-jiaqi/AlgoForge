"""沙箱：在独立子进程中执行检查，限制超时与内存，并清除环境变量中的密钥。

说明：这是进程级隔离（资源限制 + 干净环境 + 临时工作目录），不是安全容器；
真正的第一道防线是 static.py 的白名单检查。生产环境应换成容器或 gVisor 等方案。
"""

from __future__ import annotations

import json
import os
import re
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

from factory.schemas import Diagnostic, DiagnosticCode
from factory.utils import ROOT

SECRET_PATTERN = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)


def child_env(n_threads: int) -> dict[str, str]:
    """子进程环境：去掉所有疑似密钥的变量，限制线程数。"""
    env = {k: v for k, v in os.environ.items() if not SECRET_PATTERN.search(k)}
    env.update(
        PYTHONPATH=str(ROOT),
        PYTHONHASHSEED="0",
        OMP_NUM_THREADS=str(n_threads),
        OPENBLAS_NUM_THREADS=str(n_threads),
        MKL_NUM_THREADS=str(n_threads),
        LOKY_MAX_CPU_COUNT=str(n_threads),
        MALLOC_ARENA_MAX="2",
    )
    return env


def run_job(job: dict, *, timeout: float, memory_mb: int, n_threads: int) -> tuple[dict | None, Diagnostic | None]:
    """返回 (检查结果, 沙箱级错误)，两者恰有一个非空。"""

    def limit() -> None:
        mem = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))

    with tempfile.TemporaryDirectory(prefix="algoforge-sandbox-") as tmp:
        job_path, out_path = Path(tmp) / "job.json", Path(tmp) / "result.json"
        job_path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "factory.verify", str(job_path), str(out_path)],
                cwd=tmp,
                env=child_env(n_threads),
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=limit,
            )
        except subprocess.TimeoutExpired:
            return None, Diagnostic(code=DiagnosticCode.TIMEOUT, stage="sandbox",
                                    message=f"执行超过 {timeout:.0f} 秒被终止", hint="降低复杂度或检查死循环")
        if out_path.exists():
            return json.loads(out_path.read_text(encoding="utf-8")), None
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        if "MemoryError" in (proc.stderr or "") or proc.returncode < 0:
            return None, Diagnostic(code=DiagnosticCode.TIMEOUT, stage="sandbox",
                                    message=f"子进程异常退出（code {proc.returncode}），可能超出内存上限 {memory_mb} MB",
                                    hint="\n".join(tail))
        return None, Diagnostic(code=DiagnosticCode.RUNTIME_ERROR, stage="sandbox",
                                message="子进程出错：" + (tail[-1] if tail else f"exit {proc.returncode}"),
                                hint="\n".join(tail))
