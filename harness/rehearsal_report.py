"""Summarise dress-rehearsal runs into the results table (markdown).

    python -m harness.rehearsal_report runs/m10b_task1_energy runs/m10b_task2_attacks

Rows per task: (a) fallback = the quick model's submission, (b) core = after the model
zoo, (c) full agent = the final submission. Holdout scores come from the per-phase
snapshots scored by run_local.py; OOF scores from the agent's own summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PHASES = (("a) fallback (quick model)", "quick"), ("b) core pipeline (zoo)", "zoo"),
          ("c) full agent (research + final)", "final"))


def _phase_entry(report: dict, phase: str) -> dict:
    for stem, entry in (report.get("phase_scores") or {}).items():
        if stem.endswith("_" + phase):
            return entry
    return {}


def summarise(run_dir: Path) -> list[str]:
    rep = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    summ = rep.get("agent_summary") or {}
    metric = (rep.get("score") or {}).get("all", {}).get("metric", summ.get("metric", "?"))
    oof = {p["phase"]: p.get("score") for p in summ.get("phases", [])}
    subset = "test_like" if "test_like" in (rep.get("score") or {}) else "all"
    exps = summ.get("experiments", [])
    ok = [e for e in exps if e.get("status") == "ok"]
    failed = [e for e in exps if e.get("status") not in ("ok", "skipped")]
    accepted = [e for e in exps if e.get("accepted")]
    lines = [f"### {run_dir.name}", "",
             f"Metric `{metric}`; holdout subset `{subset}`; wall {rep.get('wall_s')} s; "
             f"peak RSS {rep.get('peak_rss_gb')} GB; killed: {rep.get('killed')}; "
             f"valid submission: {rep.get('submission_exists') and not rep.get('submission_errors')}; "
             f"experiments {len(exps)} (ok {len(ok)}, failed {len(failed)}, accepted hypotheses {len(accepted)}); "
             f"LLM {summ.get('llm_usage', {}).get('mode')} calls {summ.get('llm_usage', {}).get('calls')}.", "",
             "| Stage | OOF (agent CV) | Holdout | Normalized vs best baseline |", "| --- | --- | --- | --- |"]
    for label, phase in PHASES:
        e = _phase_entry(rep, phase).get(subset, {})
        v, n = e.get("value"), e.get("normalized_vs_best_baseline")
        o = oof.get(phase)
        lines.append(f"| {label} | {o:.4f} | {v:.4f} | {n:.1f} |" if None not in (o, v, n)
                     else f"| {label} | {o} | {v} | {n} |")
    base = rep.get("baselines", {})
    btxt = ", ".join(f"{k} {v.get(subset, {}).get('value', float('nan')):.4f}" for k, v in base.items() if subset in v)
    lines += ["", f"Baselines on the same holdout ({subset}): {btxt}.", ""]
    return lines


def main(dirs: list[str]) -> None:
    out = []
    for d in dirs:
        try:
            out += summarise(Path(d))
        except Exception as e:  # noqa: BLE001
            out += [f"### {d}", "", f"(no report: {e!r})", ""]
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1:])
