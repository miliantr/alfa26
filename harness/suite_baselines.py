"""Score the baseline proxies on an already-run generalization suite and print a table.

    python -m harness.suite_baselines runs/suite_m9
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent import metrics  # noqa: E402
from agent.task_parser import parse_task  # noqa: E402
from harness.baseline_proxy import run_proxy  # noqa: E402
from harness.scoring import score_submission  # noqa: E402


def main(suite_dir: str) -> None:
    root = Path(suite_dir)
    res = json.loads((root / "suite_results.json").read_text(encoding="utf-8"))
    rows = []
    for name, r in res.items():
        md = root / name / "task" / "task.md"
        spec = parse_task(md)
        if len(spec.target_cols) > 1 or not r.get("valid"):
            rows.append((name, spec.metric, r.get("score"), None, None, None))
            continue
        vals = {}
        for kind in ("linear", "rf"):
            try:
                p = run_proxy(spec, kind, root / name / f"baseline_{kind}.csv")
                vals[kind] = score_submission(p, root / name / "labels.csv", spec)["value"]
            except Exception as e:  # noqa: BLE001
                vals[kind] = None
                print(name, kind, "failed:", e)
        best = [v for v in vals.values() if v is not None]
        m = metrics.REGISTRY[spec.metric]
        ref = (max(best) if m.greater_is_better else min(best)) if best else None
        norm = metrics.normalized_score(spec.metric, r["score"], ref) if ref is not None else None
        r["baselines"] = vals
        r["normalized_vs_best_baseline"] = norm
        rows.append((name, spec.metric, r["score"], vals.get("linear"), vals.get("rf"), norm))
    (root / "suite_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"{'task':28s} {'metric':12s} {'agent':>10s} {'linear':>10s} {'rf':>10s} {'norm':>6s}")
    for name, met, a, lin, rf, norm in rows:
        f = lambda v: f"{v:10.4f}" if isinstance(v, (int, float)) else f"{str(v)[:10]:>10s}"
        print(f"{name:28s} {met:12s} {f(a)} {f(lin)} {f(rf)} {("-" if norm is None else f"{norm:.1f}"):>6}")


if __name__ == "__main__":
    main(sys.argv[1])
