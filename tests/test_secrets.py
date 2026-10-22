"""防止提交密钥：扫描所有会被 git 跟踪的文件。"""

import os
import re
import subprocess

from dotenv import dotenv_values

from factory.utils import ROOT

KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9]{20,}")


def _candidate_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return [f for f in out.splitlines() if f]


def test_env_file_is_ignored():
    files = _candidate_files()
    assert ".env" not in files
    res = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=ROOT)
    assert res.returncode == 0, ".env must be git-ignored"


def test_no_api_key_in_tracked_files():
    real_key = dotenv_values(ROOT / ".env").get("LLM_API_KEY") or os.getenv("LLM_API_KEY")
    leaks = []
    for rel in _candidate_files():
        try:
            text = (ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        if KEY_PATTERN.search(text) or (real_key and real_key in text):
            leaks.append(rel)
    assert not leaks, f"possible API key found in: {leaks}"
