"""Turn a task directory into an offline-scorable one.

The test labels are hidden, so we carve a holdout out of ``train.csv``: the copy's
``train.csv`` keeps the rest, ``test.csv`` and ``sample_submission.csv`` are rebuilt from
the holdout, and the true labels go to a file outside the task directory.

The split mimics the real train/test relation when it can: if the real test lies after
train on some ordinal column (e.g. ``Year_Factor``), the holdout is the last period.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from agent_io_utils import read_csv
from agent_task_parser import parse_task


def detect_time_column(train: pd.DataFrame, test: pd.DataFrame, exclude: set[str]) -> str | None:
    best = None
    for c in train.columns:
        if c in exclude or c not in test.columns:
            continue
        a = pd.to_numeric(train[c], errors="coerce")
        b = pd.to_numeric(test[c], errors="coerce")
        if a.notna().mean() < 0.99 or b.notna().mean() < 0.99:
            continue
        nun = a.nunique()
        if not 2 <= nun <= 1000:
            continue
        if b.min() >= a.max():  # test strictly at or beyond the end of train
            if best is None or nun < best[1]:
                best = (c, nun)
    return best[0] if best else None


def make_holdout(task_md: Path, labels_out: Path, split: str = "auto", frac: float = 0.25,
                 seed: int = 0, max_train_rows: int | None = None) -> dict:
    """Rewrite the task copy at ``task_md`` in place. Returns split metadata."""
    spec = parse_task(task_md)
    train = read_csv(Path(spec.train_path), dtype=str, keep_default_na=False)
    test = read_csv(Path(spec.test_path), dtype=str, keep_default_na=False, nrows=50000)
    targets = spec.target_cols
    id_col = spec.id_col
    exclude = set(targets) | ({id_col} if id_col else set())
    info: dict = {"split": split, "rows_before": len(train)}

    if split == "auto":
        tcol = detect_time_column(train, test, exclude)
        split = f"time:{tcol}" if tcol else ("stratified" if spec.task_type != "regression" else "random")
    info["split_resolved"] = split
    if split.startswith("time:"):
        col = split.split(":", 1)[1]
        v = pd.to_numeric(train[col], errors="coerce")
        last = v.max()
        mask = (v == last).to_numpy()
        if mask.mean() > 0.6 or mask.mean() < 0.05:  # last period too big/small: take a quantile
            mask = (v >= v.quantile(1 - frac)).to_numpy()
        hold_idx = np.where(mask)[0]
        keep_idx = np.where(~mask)[0]
        info["time_column"] = col
    else:
        strat = train[targets[0]] if split == "stratified" else None
        if strat is not None and strat.value_counts().min() < 2:
            strat = None
        keep_idx, hold_idx = train_test_split(np.arange(len(train)), test_size=frac,
                                              random_state=seed, stratify=strat)
        keep_idx, hold_idx = np.sort(keep_idx), np.sort(hold_idx)
    if max_train_rows and len(keep_idx) > max_train_rows:
        rng = np.random.default_rng(seed)
        keep_idx = np.sort(rng.choice(keep_idx, max_train_rows, replace=False))

    new_train = train.iloc[keep_idx]
    hold = train.iloc[hold_idx]
    if not id_col:
        raise ValueError("holdout harness needs an id column")
    new_test = hold.drop(columns=targets)
    # Keep the real test column order.
    real_test_cols = list(read_csv(Path(spec.test_path), nrows=0).columns)
    new_test = new_test[[c for c in real_test_cols if c in new_test.columns]]

    sample_cols = spec.submission_cols
    placeholder = "0" if spec.task_type == "regression" else (spec.classes[0] if spec.classes else "0")
    sample = pd.DataFrame({c: (hold[id_col].values if c == id_col else placeholder) for c in sample_cols})
    if spec.output_kind == "proba_wide":
        for c in spec.proba_columns:
            sample[c] = "0.0"
    labels = hold[[id_col] + targets]

    new_train.to_csv(spec.train_path, index=False)
    new_test.to_csv(spec.test_path, index=False)
    if spec.sample_path:
        sample.to_csv(spec.sample_path, index=False)
    labels_out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(labels_out, index=False)
    info.update(train_rows=len(new_train), holdout_rows=len(hold))
    return info


def test_like_ids(original_md: Path, copy_md: Path, max_card: int = 200) -> tuple[set, dict]:
    """Holdout ids whose categorical values were seen in the holdout's train part, for
    every low-cardinality column on which the *real* test has no unseen values.

    Mirrors the agent's eval mask: it estimates the score under the real test's
    conditions when the holdout itself introduces brand-new categories.
    """
    real = parse_task(original_md)
    cp = parse_task(copy_md)
    rtr = read_csv(Path(real.train_path), dtype=str, keep_default_na=False)
    rte = read_csv(Path(real.test_path), dtype=str, keep_default_na=False)
    tr = read_csv(Path(cp.train_path), dtype=str, keep_default_na=False)
    ho = read_csv(Path(cp.test_path), dtype=str, keep_default_na=False)
    keep = np.ones(len(ho), dtype=bool)
    info = {}
    for c in ho.columns:
        if c == cp.id_col or c not in tr.columns:
            continue
        nonempty = rtr[c][rtr[c] != ""]
        if len(nonempty) and pd.to_numeric(nonempty, errors="coerce").notna().mean() > 0.95:
            continue  # numeric column (empty cells are missing values)
        if rtr[c].nunique() > max_card:
            continue
        if (~rte[c].isin(set(rtr[c]))).mean() > 0.005:
            continue  # real test extrapolates on this column too
        unseen = ~ho[c].isin(set(tr[c])).to_numpy()
        if unseen.any():
            keep &= ~unseen
            info[c] = int(unseen.sum())
    return set(ho.loc[keep, cp.id_col]), info
