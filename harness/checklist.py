"""Submission-readiness checklist.

    python -m harness.checklist                      # all checks, default task 2
    python -m harness.checklist --task data/task1_energy/task.md --skip-kill
    python -m harness.checklist --rehearsals runs/m10_task1 runs/m10_task2

Checks
  1. imports ⊆ requirements intersection          (pytest tests/test_imports.py)
  2. no network except the LLM endpoint            (static test + an offline Docker run)
  3. ZIP root contains solution.py                 (build + inspect)
  4. submission exists when the LLM is down        (unreachable endpoint + fake key)
  5. submission exists when killed after minute 5  (hard kill at two arbitrary moments)
  6. peak RAM < 28 GB and 7. finish within 3 h     (from dress-rehearsal reports)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run_local(args: list[str], env_extra: dict | None = None) -> dict:
    out = REPO / "runs" / f"checklist_{abs(hash(tuple(args))) % 10 ** 8}"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", **(env_extra or {})}
    subprocess.run([PY, str(REPO / "run_local.py"), *args, "--out", str(out), "--baselines", ""],
                   cwd=str(REPO), env=env, capture_output=True, text=True)
    try:
        return json.loads((out / "report.json").read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="data/task2_attacks/task.md")
    ap.add_argument("--skip-kill", action="store_true")
    ap.add_argument("--skip-docker", action="store_true")
    ap.add_argument("--rehearsals", nargs="*", default=[])
    args = ap.parse_args()
    results: dict = {}

    r = subprocess.run([PY, "-m", "pytest", "-q", "tests/test_imports.py"], cwd=str(REPO),
                       capture_output=True, text=True)
    results["1_imports_within_requirements"] = r.returncode == 0
    results["2a_no_network_libs_outside_llm_client"] = "passed" in r.stdout and r.returncode == 0

    sh = subprocess.run(["bash", str(REPO / "make_submission.sh")], cwd=str(REPO), capture_output=True, text=True)
    zips = sorted((REPO / "dist").glob("submission_*.zip"), key=lambda p: p.stat().st_mtime)
    if zips:
        names = zipfile.ZipFile(zips[-1]).namelist()
        results["3_zip_root_has_solution_py"] = "solution.py" in names
        results["3_zip"] = {"path": str(zips[-1]), "files": len(names),
                            "forbidden": [n for n in names if n.endswith(("pyproject.toml", "uv.lock", ".csv"))]}
    else:
        results["3_zip_root_has_solution_py"] = False
        results["3_zip"] = sh.stderr[-500:]

    down = _run_local(["--task", args.task, "--time-budget", "240", "--llm", "real"],
                      {"OPENAI_API_KEY": "sk-invalid", "OPENAI_MODEL": "qwen-test",
                       "OPENAI_BASE_URL": "http://127.0.0.1:9/v1"})
    results["4_submission_when_llm_down"] = bool(down.get("submission_exists")) and not down.get("submission_errors")
    results["4_detail"] = {k: down.get(k) for k in ("exit_code", "wall_s", "submission_errors", "agent_summary")
                           if k != "agent_summary"}
    if down.get("agent_summary"):
        results["4_detail"]["llm_usage"] = down["agent_summary"].get("llm_usage")

    if not args.skip_kill:
        kills = {}
        for t in sorted(random.Random(7).sample(range(305, 900), 2)):
            rep = _run_local(["--task", args.task, "--time-budget", "10800", "--kill-at", str(t), "--llm", "mock"])
            kills[t] = {"killed": rep.get("killed"), "valid": bool(rep.get("submission_exists")) and not rep.get("submission_errors")}
        results["5_submission_after_kill"] = all(v["valid"] for v in kills.values())
        results["5_detail"] = kills

    if not args.skip_docker:
        rep = _run_local(["--task", args.task, "--time-budget", "300", "--llm", "off", "--docker", "--memory", "3500m"])
        results["2b_offline_docker_run_valid"] = bool(rep.get("submission_exists")) and not rep.get("submission_errors")
        results["2b_detail"] = {k: rep.get(k) for k in ("exit_code", "wall_s", "peak_rss_gb", "submission_errors")}

    peaks, walls = {}, {}
    for d in args.rehearsals:
        try:
            rep = json.loads((Path(d) / "report.json").read_text(encoding="utf-8"))
            peaks[d], walls[d] = rep.get("peak_rss_gb"), rep.get("wall_s")
        except Exception:
            pass
    if peaks:
        results["6_peak_ram_below_28gb"] = all((v or 0) < 28 for v in peaks.values())
        results["7_finished_within_3h"] = all((v or 1e9) <= 10800 for v in walls.values())
        results["6_7_detail"] = {"peak_rss_gb": peaks, "wall_s": walls}

    out = REPO / "runs" / "checklist.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    flags = [v for k, v in results.items() if isinstance(v, bool)]
    return 0 if all(flags) else 1


if __name__ == "__main__":
    sys.exit(main())
