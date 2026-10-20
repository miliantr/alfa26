"""Score a submission file against holdout labels with the task metric."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agent import metrics
from agent.task_parser import TaskSpec


def score_submission(sub_path: Path, labels_path: Path, spec: TaskSpec, ids: set | None = None) -> dict:
    sub = pd.read_csv(sub_path, dtype={spec.id_col: str})
    lab = pd.read_csv(labels_path, dtype={spec.id_col: str})
    if ids is not None:
        lab = lab[lab[spec.id_col].isin(ids)]
    target = spec.target_cols[0]
    merged = lab.merge(sub, on=spec.id_col, how="left", suffixes=("_true", ""))
    out: dict = {"metric": spec.metric, "n": len(merged)}
    true_col = f"{target}_true" if f"{target}_true" in merged.columns else target
    if spec.task_type == metrics.REGRESSION:
        y = merged[true_col].astype(float).to_numpy()
        p = merged[target].astype(float).to_numpy()
        out["value"] = metrics.evaluate(spec.metric, y, p)
        for extra in ("rmse", "mae", "r2"):
            out[extra] = metrics.evaluate(extra, y, p)
        return out
    classes = spec.classes
    enc = {c: i for i, c in enumerate(classes)}
    y = merged[true_col].astype(str).map(enc).fillna(-1).astype(int).to_numpy()
    if spec.output_kind == "label":
        lab_pred = merged[target].astype(str).map(enc).fillna(-1).astype(int).to_numpy()
        proba = np.zeros((len(y), len(classes)))
        ok = lab_pred >= 0
        proba[np.where(ok)[0], lab_pred[ok]] = 1.0
    elif spec.output_kind == "proba":
        pos = enc[spec.positive_label]
        p1 = merged[target].astype(float).to_numpy()
        proba = np.zeros((len(y), 2))
        proba[:, pos] = p1
        proba[:, 1 - pos] = 1 - p1
    else:
        proba = merged[classes].astype(float).to_numpy()
    out["value"] = metrics.evaluate(spec.metric, y, proba)
    for extra in ("accuracy", "f1_macro", "f1_weighted"):
        out[extra] = metrics.evaluate(extra, y, proba)
    return out
