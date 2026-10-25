"""沙箱子进程入口：python -m factory.verify <job.json> <result.json>

读入任务后先预热依赖，再安装审计钩子（见 guard.py），之后才执行生成的算子代码。
"""

import json
import os
import sys

from factory.verify import guard

job_path, out_path = sys.argv[1], sys.argv[2]
with open(job_path, encoding="utf-8") as f:
    job = json.load(f)

guard.prewarm()
guard.install(workdir=os.getcwd(), readable_files=[job["data_path"]])

from factory.verify.checks import run_checks  # noqa: E402 - 预热时已导入，这里只是取名字

result = run_checks(job)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False)
