"""Last line of defence: make sure *some* valid submission file exists.

Tries the normal template path first (pandas); if even that fails, falls back to the
standard library and copies ``sample_submission.csv`` (right ids, columns and order; the
placeholder values score poorly but a file always beats no file).
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path


def _existing_ok(paths: list[Path], n_expected: int | None) -> bool:
    for p in paths:
        try:
            if p.exists() and p.stat().st_size > 0:
                if n_expected is None:
                    return True
                with open(p, encoding="utf-8", errors="replace") as f:
                    rows = sum(1 for _ in f) - 1
                if rows == n_expected:
                    return True
        except Exception:
            continue
    return False


def _find(task_dir: Path, word: str) -> Path | None:
    cands = [p for p in task_dir.rglob("*.csv") if word in p.name.lower() and "agent_run" not in p.parts]
    cands.sort(key=lambda p: len(p.parts))
    return cands[0] if cands else None


def _count_rows(path: Path | None) -> int | None:
    if not path:
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return max(0, sum(1 for _ in csv.reader(f)) - 1)
    except Exception:
        return None


def ensure_submission(task_md: Path, reason: str = "") -> None:
    task_md = Path(task_md)
    task_dir = task_md.parent
    default_out = task_dir / "submission.csv"
    sys.stderr.write(f"[emergency] checking submission ({reason[:300]})\n")
    # 1) Template-based constant prediction (validated).
    try:
        from .io_utils import read_csv
        from .submission import SubmissionTemplate
        from .task_parser import parse_task

        spec = parse_task(task_md)
        outs = [Path(p) for p in spec.output_paths] or [default_out]
        tpl = SubmissionTemplate.from_spec(spec)
        if _existing_ok(outs, tpl.n_rows):
            return
        y = None
        if spec.train_path and spec.target_cols:
            y = read_csv(Path(spec.train_path), usecols=spec.target_cols[:1], dtype=str)[spec.target_cols[0]]
        if tpl.write(tpl.constant(y), "emergency-constant"):
            return
    except BaseException as e:  # noqa: BLE001
        sys.stderr.write(f"[emergency] template path failed: {e!r}\n")
    # 2) Stdlib copy of the sample submission.
    try:
        test = _find(task_dir, "test")
        if _existing_ok([default_out], _count_rows(test)):
            return
        sample = _find(task_dir, "sample")
        if sample is None:
            return
        data = sample.read_bytes()
        fd, tmp = tempfile.mkstemp(dir=str(task_dir), prefix=".submission.", suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, default_out)
        sys.stderr.write("[emergency] copied sample submission\n")
    except BaseException as e:  # noqa: BLE001
        sys.stderr.write(f"[emergency] stdlib path failed: {e!r}\n")
