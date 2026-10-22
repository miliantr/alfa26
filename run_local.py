"""Run solution.py exactly like the platform would, locally or in the offline Docker image.

Examples
--------
Plain platform-like run on a task (no scoring possible: test labels are hidden)::

    python run_local.py --task data/task1_energy/task.md --time-budget 900

Offline-scorable run: holdout carved from train.csv, scored, baseline proxies computed::

    python run_local.py --task data/task2_attacks/task.md --holdout --time-budget 900

Same inside the offline container (network disabled, 8 CPUs)::

    python run_local.py --task data/task1_energy/task.md --holdout --docker
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import agent_metrics as metrics  # noqa: E402
from agent_io_utils import write_json  # noqa: E402
from agent_submission import SubmissionTemplate, check_file  # noqa: E402
from agent_task_parser import parse_task  # noqa: E402

DOCKER_IMAGE = "alfa-automl-platform:py312"


def _copy_task(task_md: Path, dest: Path) -> Path:
    src = task_md.parent
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("agent_run", "submission.csv", ".*"))
    return dest / task_md.name


class PeakRSS(threading.Thread):
    def __init__(self, pid: int):
        super().__init__(daemon=True)
        self.pid, self.peak, self.stop = pid, 0, False

    def run(self):
        from agent_resources import rss_tree

        while not self.stop:
            self.peak = max(self.peak, rss_tree(self.pid))
            time.sleep(1.0)


def _env(args) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENAI_", "AGENT_"))}
    env["PYTHONIOENCODING"] = "utf-8"
    env["AGENT_TIME_BUDGET"] = str(args.time_budget)
    env["AGENT_LLM_MODE"] = args.llm
    if args.llm == "real":
        for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
            if os.environ.get(k):
                env[k] = os.environ[k]
    for kv in args.env or []:
        k, _, v = kv.partition("=")
        env[k] = v
    return env


def _stage_solution(sol_dir: Path, dest: Path) -> Path:
    """Copy what the platform would: from the repo the files that ship in the ZIP, from an
    unzipped ZIP only its root-level .py files and requirements.txt (subfolders are dropped)."""
    from harness.ship import zip_members

    if (sol_dir / ".git").exists() or (sol_dir / ".venv").exists():
        files = zip_members(sol_dir)
    else:
        files = {p.name: p for p in sol_dir.glob("*.py")}
        if (sol_dir / "requirements.txt").exists():
            files["requirements.txt"] = sol_dir / "requirements.txt"
    dest.mkdir(parents=True, exist_ok=True)
    for name, src in files.items():
        shutil.copy2(src, dest / name)
    return dest


def _docker_peak_mb(name: str) -> float:
    try:
        out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", name],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        val = out.split("/")[0].strip()
        num = float("".join(ch for ch in val if ch.isdigit() or ch == "."))
        unit = val.lower()
        return num * (1024 if "gib" in unit else 1 if "mib" in unit else 1 / 1024)
    except Exception:
        return 0.0


def _launch_docker(args, task_md: Path, run_root: Path, log_f) -> tuple[int | None, float, int, bool]:
    """docker create + cp avoids host bind mounts (Docker Desktop file-sharing settings).
    Platform layout: the solution files are copied into the task directory and run from
    there as `python3 solution.py task.md`, as a non-root user."""
    env = _env(args)
    name = f"alfa_run_{int(time.time())}"
    sol = _stage_solution(Path(args.solution_dir).resolve(), run_root / "solution_stage")
    timeout_s = int(args.kill_at or args.time_budget)
    inner = f"timeout -s KILL {timeout_s} python3 solution.py {task_md.name}"
    shell = f"chown -R runner:runner /task && exec su -p runner -s /bin/sh -c '{inner}'"
    cmd = ["docker", "create", "--name", name, "--network", "none", f"--cpus={args.cpus}",
           f"--memory={args.memory}", "--memory-swap", args.memory, "--user", "0", "-w", "/task",
           "-e", "HOME=/tmp"]
    for k in ("AGENT_TIME_BUDGET", "AGENT_LLM_MODE", "PYTHONIOENCODING"):
        cmd += ["-e", f"{k}={env[k]}"]
    for kv in args.env or []:
        cmd += ["-e", kv]
    cmd += [args.image, "sh", "-c", shell]
    print("launch:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, capture_output=True)
    subprocess.run(["docker", "cp", f"{task_md.parent}{os.sep}.", f"{name}:/task"], check=True, capture_output=True)
    subprocess.run(["docker", "cp", f"{sol}{os.sep}.", f"{name}:/task"], check=True, capture_output=True)
    t0 = time.time()
    proc = subprocess.Popen(["docker", "start", "-a", name], stdout=log_f, stderr=subprocess.STDOUT)
    peak_mb, killed = 0.0, False
    while proc.poll() is None:
        peak_mb = max(peak_mb, _docker_peak_mb(name))
        if time.time() - t0 > timeout_s + 60:
            subprocess.run(["docker", "kill", name], capture_output=True)
            killed = True
        time.sleep(2)
    wall = time.time() - t0
    rc = proc.returncode
    killed = killed or rc == 137
    subprocess.run(["docker", "cp", f"{name}:/task/.", str(task_md.parent)], capture_output=True)
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    return rc, wall, int(peak_mb * 1024 ** 2), killed


def _launch(args, task_md: Path, run_root: Path) -> tuple[int | None, float, int, bool]:
    env = _env(args)
    kill_at = args.kill_at or args.time_budget
    log_f = open(run_root / "solution_stdout.log", "w", encoding="utf-8")
    if args.docker:
        try:
            return _launch_docker(args, task_md, run_root, log_f)
        finally:
            log_f.close()
    sol_dir = Path(args.solution_dir).resolve()
    cmd = [sys.executable, str(sol_dir / "solution.py"), str(task_md)]
    print("launch:", " ".join(cmd), flush=True)
    t0 = time.time()
    # cwd is deliberately NOT the task dir: the solution must resolve paths itself.
    proc = subprocess.Popen(cmd, cwd=str(run_root), env=env, stdout=log_f, stderr=subprocess.STDOUT)
    mon = PeakRSS(proc.pid)
    mon.start()
    killed = False
    try:
        rc = proc.wait(timeout=kill_at)
    except subprocess.TimeoutExpired:
        killed = True
        from agent_resources import kill_tree

        kill_tree(proc.pid)
        rc = None
    mon.stop = True
    log_f.close()
    return rc, time.time() - t0, mon.peak, killed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="path to task.md")
    ap.add_argument("--holdout", action="store_true", help="carve a scorable holdout from train.csv")
    ap.add_argument("--split", default="auto", help="auto | random | stratified | time:<column>")
    ap.add_argument("--frac", type=float, default=0.25, help="holdout fraction (random splits)")
    ap.add_argument("--max-train-rows", type=int, default=None, help="subsample train for quick runs")
    ap.add_argument("--time-budget", type=float, default=10800, help="seconds (platform: 10800)")
    ap.add_argument("--kill-at", type=float, default=None, help="hard-kill earlier than the budget")
    ap.add_argument("--llm", default="mock", choices=["off", "mock", "real"])
    ap.add_argument("--docker", action="store_true", help="run inside the offline platform image")
    ap.add_argument("--image", default=DOCKER_IMAGE)
    ap.add_argument("--cpus", default="8")
    ap.add_argument("--memory", default="12g", help="container memory limit (platform: 32g)")
    ap.add_argument("--solution-dir", default=str(REPO), help="dir containing solution.py (e.g. unzipped ZIP)")
    ap.add_argument("--baselines", default="linear,rf", help="baseline proxies to score ('' to skip)")
    ap.add_argument("--out", default=None, help="run directory (default runs/<ts>_<task>)")
    ap.add_argument("--env", action="append", help="extra KEY=VALUE for the solution")
    args = ap.parse_args()

    task_md = Path(args.task).resolve()
    name = task_md.parent.name + ("_holdout" if args.holdout else "")
    run_root = Path(args.out).resolve() if args.out else REPO / "runs" / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}"
    run_root.mkdir(parents=True, exist_ok=True)
    task_copy = _copy_task(task_md, run_root / "task")
    report: dict = {"task": str(task_md), "args": vars(args)}

    labels_path = run_root / "holdout_labels.csv"
    if args.holdout:
        from harness.holdout import make_holdout

        report["holdout"] = make_holdout(task_copy, labels_path, split=args.split, frac=args.frac,
                                         max_train_rows=args.max_train_rows)
        print("holdout:", report["holdout"], flush=True)

    rc, wall, peak, killed = _launch(args, task_copy, run_root)
    report.update(exit_code=rc, wall_s=round(wall, 1), peak_rss_gb=round(peak / 1024 ** 3, 2), killed=killed)

    spec = parse_task(task_copy)
    tpl = SubmissionTemplate.from_spec(spec)
    out_path = Path(spec.output_paths[0])
    report["submission_path"] = str(out_path)
    report["submission_exists"] = out_path.exists()
    report["submission_errors"] = check_file(out_path, tpl) if out_path.exists() else ["missing"]

    if args.holdout and out_path.exists() and not report["submission_errors"]:
        from harness.baseline_proxy import run_proxy
        from harness.holdout import test_like_ids
        from harness.scoring import score_submission

        like_ids, like_info = test_like_ids(task_md, task_copy)
        n_hold = report["holdout"]["holdout_rows"]
        subsets = {"all": None}
        if like_info and len(like_ids) < n_hold:
            subsets["test_like"] = like_ids
            report["test_like_subset"] = {"rows": len(like_ids), "excluded_unseen": like_info}
        report["score"] = {k: score_submission(out_path, labels_path, spec, ids) for k, ids in subsets.items()}
        report["baselines"] = {}
        for kind in [k for k in args.baselines.split(",") if k]:
            t0 = time.time()
            try:
                bp = run_proxy(spec, kind, run_root / f"baseline_{kind}.csv")
                entry = {"seconds": round(time.time() - t0, 1)}
                for k, ids in subsets.items():
                    b = score_submission(bp, labels_path, spec, ids)["value"]
                    entry[k] = {"value": b, "normalized": metrics.normalized_score(
                        spec.metric, report["score"][k]["value"], b)}
                report["baselines"][kind] = entry
            except Exception as e:  # noqa: BLE001
                report["baselines"][kind] = {"error": repr(e)}
        for k in subsets:
            norms = [v[k]["normalized"] for v in report["baselines"].values() if k in v]
            report[f"normalized_conservative_{k}"] = min(norms) if norms else None
        # Score the per-phase snapshots: fallback (quick) vs core (zoo) vs full agent.
        snaps = sorted((task_copy.parent / "agent_run").glob("*/snapshots/*.csv"))
        report["phase_scores"] = {}
        for sp in snaps:
            entry = {}
            for k, ids in subsets.items():
                try:
                    v = score_submission(sp, labels_path, spec, ids)["value"]
                except Exception:  # noqa: BLE001
                    continue
                bvals = [b[k]["value"] for b in report["baselines"].values() if k in b]
                entry[k] = {"value": v, "normalized_vs_best_baseline": min(
                    (metrics.normalized_score(spec.metric, v, b) for b in bvals), default=None)}
            report["phase_scores"][sp.stem] = entry

    agent_runs = sorted((task_copy.parent / "agent_run").glob("*"))
    if agent_runs:
        report["agent_run_dir"] = str(agent_runs[-1])
        summ = agent_runs[-1] / "summary.json"
        if summ.exists():
            report["agent_summary"] = json.loads(summ.read_text(encoding="utf-8"))
    write_json(run_root / "report.json", report)
    keys = ("exit_code", "wall_s", "peak_rss_gb", "killed", "submission_exists", "submission_errors",
            "score", "baselines", "test_like_subset", "normalized_conservative_all",
            "normalized_conservative_test_like", "phase_scores")
    print(json.dumps({k: report.get(k) for k in keys}, indent=2, ensure_ascii=False, default=str))
    print("report:", run_root / "report.json")
    return 0 if report["submission_exists"] and not report["submission_errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
