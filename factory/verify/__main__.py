"""沙箱子进程入口：python -m factory.verify <job.json> <result.json>"""

import json
import sys

from factory.verify.checks import run_checks

job_path, out_path = sys.argv[1], sys.argv[2]
with open(job_path, encoding="utf-8") as f:
    job = json.load(f)
result = run_checks(job)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False)
