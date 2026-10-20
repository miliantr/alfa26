"""Submission template, validation and atomic writing.

The template fixes ids (exact strings from test.csv, in test.csv order), columns
(from sample_submission) and value kinds. Nothing is written unless it validates.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import log, metrics
from .io_utils import atomic_write_csv, read_csv
from .task_parser import TaskSpec

L = log.get("submission")


class SubmissionTemplate:
    def __init__(self, spec: TaskSpec, ids: list[str] | None, columns: list[str]):
        self.spec = spec
        self.ids = ids  # None when the task has no id column
        self.columns = columns
        self.n_rows = len(ids) if ids is not None else None
        self.value_cols = [c for c in columns if c != spec.id_col]

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_spec(cls, spec: TaskSpec, n_test: int | None = None) -> "SubmissionTemplate":
        ids = None
        if spec.id_col and spec.test_path:
            col = read_csv(Path(spec.test_path), usecols=[spec.id_col], dtype=str, keep_default_na=False)
            ids = col[spec.id_col].tolist()
        elif spec.sample_path:
            # No id in test: keep the sample's first column if it looks like an id.
            sample = read_csv(Path(spec.sample_path), dtype=str, keep_default_na=False)
            if spec.id_col and spec.id_col in sample.columns:
                ids = sample[spec.id_col].tolist()
        columns = list(spec.submission_cols) or ([spec.id_col] if spec.id_col else []) + spec.target_cols
        tpl = cls(spec, ids, columns)
        if tpl.n_rows is None and n_test is not None:
            tpl.n_rows = n_test
        tpl._check_sample()
        return tpl

    def _check_sample(self) -> None:
        if not self.spec.sample_path or self.ids is None or not self.spec.id_col:
            return
        try:
            s = read_csv(Path(self.spec.sample_path), usecols=[self.spec.id_col], dtype=str, keep_default_na=False)
        except Exception:
            return
        sid = s[self.spec.id_col].tolist()
        if set(sid) != set(self.ids):
            msg = f"sample ids differ from test ids (sample {len(sid)}, test {len(self.ids)}); using test.csv"
            self.spec.warnings.append(msg)
            L.warning(msg)
        elif sid != self.ids:
            L.info("sample order differs from test.csv order; following test.csv as required")

    # ------------------------------------------------------------------ build frames
    def frame(self, values: dict[str, np.ndarray | list]) -> pd.DataFrame:
        """``values`` maps every value column to an array of length n_rows."""
        data = {}
        if self.spec.id_col and self.spec.id_col in self.columns:
            data[self.spec.id_col] = self.ids
        for c in self.value_cols:
            data[c] = values[c]
        return pd.DataFrame(data, columns=self.columns)

    def format_labels(self, labels) -> list:
        """Write labels the way they appear in train (ints stay ints, etc.)."""
        dt = self.spec.label_dtype
        out = []
        for v in labels:
            s = str(v)
            if dt == "int":
                try:
                    out.append(int(float(s)))
                    continue
                except ValueError:
                    pass
            out.append(s)
        return out

    def from_predictions(self, pred: np.ndarray) -> pd.DataFrame:
        """Build the submission from one target's raw output.

        regression: values ``(n,)``; label: class strings ``(n,)``; proba: ``(n,)``
        probability of the positive class; proba_wide: ``(n, K)`` in ``spec.classes`` order.
        """
        kind = self.spec.output_kind
        col = self.value_cols
        if kind == "proba_wide":
            pred = np.asarray(pred, float)
            idx = {c: i for i, c in enumerate(self.spec.classes)}
            return self.frame({c: pred[:, idx[c]] if c in idx else np.zeros(len(pred)) for c in col})
        if kind == "label":
            return self.frame({col[0]: self.format_labels(pred)})
        return self.frame({col[0]: np.asarray(pred, float)})

    def constant(self, train_target: pd.Series | None) -> pd.DataFrame:
        """Prior-only prediction: median / mode / class frequencies."""
        n = self.n_rows or 0
        kind = self.spec.output_kind
        y = train_target.dropna() if train_target is not None else None
        if kind == "value":
            v = float(pd.to_numeric(y, errors="coerce").median()) if y is not None and len(y) else 0.0
            if not np.isfinite(v):
                v = 0.0
            return self.from_predictions(np.full(n, v))
        classes = self.spec.classes or ([str(y.mode().iloc[0])] if y is not None and len(y) else ["0"])
        freq = y.astype(str).value_counts(normalize=True) if y is not None and len(y) else pd.Series({classes[0]: 1.0})
        if kind == "label":
            return self.from_predictions([str(freq.idxmax())] * n)
        if kind == "proba":
            p = float(freq.get(self.spec.positive_label, 0.5))
            return self.from_predictions(np.full(n, p))
        pri = np.array([float(freq.get(c, 0.0)) for c in self.spec.classes])
        return self.from_predictions(np.tile(pri, (n, 1)))

    # --------------------------------------------------------------------- validation
    def validate(self, df: pd.DataFrame) -> list[str]:
        errs: list[str] = []
        if list(df.columns) != list(self.columns):
            errs.append(f"columns {list(df.columns)} != expected {self.columns}")
            return errs
        if self.n_rows is not None and len(df) != self.n_rows:
            errs.append(f"rows {len(df)} != expected {self.n_rows}")
        if self.spec.id_col and self.spec.id_col in df.columns and self.ids is not None:
            got = df[self.spec.id_col].astype(str).tolist()
            if got != self.ids:
                errs.append("ids differ from test.csv (values or order)")
            if len(set(got)) != len(got):
                errs.append("duplicate ids")
        kind = self.spec.output_kind
        for c in self.value_cols:
            s = df[c]
            if kind == "label":
                if s.isna().any():
                    errs.append(f"{c}: missing labels")
                allowed = set(self.spec.classes)
                bad = set(s.astype(str)) - allowed
                if allowed and bad:
                    errs.append(f"{c}: labels outside allowed set: {sorted(bad)[:5]}")
            else:
                v = pd.to_numeric(s, errors="coerce").to_numpy(float)
                if not np.all(np.isfinite(v)):
                    errs.append(f"{c}: {int((~np.isfinite(v)).sum())} non-finite values")
                if kind in ("proba", "proba_wide") and np.isfinite(v).all() and ((v < 0) | (v > 1)).any():
                    errs.append(f"{c}: probabilities outside [0, 1]")
        return errs

    # -------------------------------------------------------------------------- write
    def write(self, df: pd.DataFrame, tag: str = "") -> bool:
        errs = self.validate(df)
        if errs:
            L.error("submission %s rejected: %s", tag, "; ".join(errs))
            return False
        ok = False
        inputs = {Path(p).resolve() for p in (self.spec.train_path, self.spec.test_path,
                                              self.spec.sample_path, self.spec.dict_path,
                                              self.spec.task_md) if p}
        for p in self.spec.output_paths:
            if Path(p).resolve() in inputs:
                L.error("refusing to overwrite input file %s", p)
                continue
            try:
                atomic_write_csv(df, Path(p))
                ok = True
            except Exception as e:
                L.error("cannot write %s: %s", p, e)
        if ok:
            L.info("submission written (%s): %s", tag, ", ".join(self.spec.output_paths))
        return ok


def check_file(path: Path, template: SubmissionTemplate) -> list[str]:
    """Validate a submission file already on disk (used by tests and the harness)."""
    try:
        df = read_csv(Path(path), dtype=str, keep_default_na=False)
    except Exception as e:
        return [f"unreadable: {e}"]
    if template.spec.output_kind != "label":
        for c in template.value_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return template.validate(df)
