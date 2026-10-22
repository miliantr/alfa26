"""Validate and score whatever submission an interrupted holdout run left on disk.

    python -m harness.score_partial runs/m10c_task1_energy [data/task1_energy/task.md]

Writes <run>/partial_report.json with the same fields run_local.py reports for a finished
run (validity, holdout score on all / test-like rows, baseline proxies, normalized score,
per-phase snapshot scores).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import agent_metrics as metrics  # noqa: E402
from agent_submission import SubmissionTemplate, check_file  # noqa: E402
from agent_task_parser import parse_task  # noqa: E402
from harness.baseline_proxy import run_proxy  # noqa: E402
from harness.holdout import test_like_ids  # noqa: E402
from harness.scoring import score_submission  # noqa: E402


def main(run_root: str, original_md: str | None = None) -> dict:
    root = Path(run_root).resolve()
    task_copy = root / "task" / "task.md"
    if original_md is None:
        name = root.name.split("_", 1)[1]
        original_md = str(REPO / "data" / name / "task.md")
    spec = parse_task(task_copy)
    out = Path(spec.output_paths[0])
    rep: dict = {"submission_exists": out.exists()}
    rep["submission_errors"] = check_file(out, SubmissionTemplate.from_spec(spec)) if out.exists() else ["missing"]
    labels = root / "holdout_labels.csv"
    like_ids, like_info = test_like_ids(Path(original_md), task_copy)
    subsets = {"all": None}
    if like_info:
        subsets["test_like"] = like_ids
    if not rep["submission_errors"]:
        rep["score"] = {k: score_submission(out, labels, spec, ids)["value"] for k, ids in subsets.items()}
    rep["baselines"] = {}
    for kind in ("linear", "rf"):
        bp = run_proxy(spec, kind, root / f"baseline_{kind}.csv")
        rep["baselines"][kind] = {k: score_submission(bp, labels, spec, ids)["value"] for k, ids in subsets.items()}
    m = metrics.REGISTRY[spec.metric]
    best = {k: (max if m.greater_is_better else min)(b[k] for b in rep["baselines"].values()) for k in subsets}
    if "score" in rep:
        rep["normalized_vs_best_baseline"] = {k: metrics.normalized_score(spec.metric, rep["score"][k], best[k])
                                             for k in subsets}
    rep["phase_scores"] = {}
    for sp in sorted((task_copy.parent / "agent_run").glob("*/snapshots/*.csv")):
        try:
            rep["phase_scores"][sp.stem] = {k: score_submission(sp, labels, spec, ids)["value"] for k, ids in subsets.items()}
        except Exception as e:  # noqa: BLE001
            rep["phase_scores"][sp.stem] = {"error": repr(e)}
    rep["metric"] = spec.metric
    (root / "partial_report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=2, default=str))
    return rep


if __name__ == "__main__":
    main(*sys.argv[1:])
