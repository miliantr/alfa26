"""Run worker jobs in subprocesses with a hard timeout and a memory guard."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from . import log
from .io_utils import read_json, write_json
from .resources import kill_tree, rss_tree

L = log.get("executor")
PKG_ROOT = Path(__file__).resolve().parents[1]


def run_job(job: dict, job_dir: Path, timeout: float, mem_limit: int, n_jobs: int,
            module: str = "agent.worker") -> dict:
    job_dir.mkdir(parents=True, exist_ok=True)
    job = {**job, "out_dir": str(job_dir), "n_jobs": n_jobs}
    write_json(job_dir / "job.json", job)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PKG_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[v] = str(n_jobs)
    # The worker never needs the LLM credentials.
    for v in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
        env.pop(v, None)
    t0 = time.time()
    with open(job_dir / "worker.log", "w", encoding="utf-8") as logf:
        proc = subprocess.Popen([sys.executable, "-m", module, str(job_dir / "job.json")],
                                cwd=str(PKG_ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT)
        status, peak = None, 0
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            el = time.time() - t0
            rss = rss_tree(proc.pid)
            peak = max(peak, rss)
            if el > timeout:
                status = "timeout"
            elif mem_limit and rss > mem_limit:
                status = "oom"
            if status:
                L.warning("job %s killed: %s (%.0fs, %.2f GB)", job_dir.name, status, el, rss / 1024 ** 3)
                kill_tree(proc.pid)
                break
            time.sleep(0.25 if el < 5 else 1.0)
    wall = time.time() - t0
    res = read_json(job_dir / "result.json", default=None)
    if status:
        res = {"status": status, "error": f"{status} after {wall:.0f}s"}
    elif res is None:
        res = {"status": "error", "error": f"worker exited with code {proc.returncode} without result"}
    res["wall_seconds"] = round(wall, 2)
    res["peak_rss_gb"] = round(peak / 1024 ** 3, 3)
    return res
