"""Orchestration of one task run."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

from . import log
from .config import load_settings
from .deadline import Deadline
from .io_utils import read_csv, write_json
from .submission import SubmissionTemplate
from .task_parser import parse_task


def make_run_dir(task_md: Path, configured: str | None) -> Path:
    cands = [Path(configured)] if configured else []
    cands.append(task_md.parent / "agent_run" / time.strftime("%Y%m%d_%H%M%S"))
    for c in cands:
        try:
            c.mkdir(parents=True, exist_ok=True)
            probe = c / ".probe"
            probe.write_text("ok")
            probe.unlink()
            return c
        except Exception:
            continue
    return Path(tempfile.mkdtemp(prefix="agent_run_"))


def run(task_md: Path) -> None:
    settings = load_settings()
    deadline = Deadline(settings.time_budget)
    task_md = Path(task_md).resolve()
    run_dir = make_run_dir(task_md, settings.run_dir)
    L = log.setup(run_dir)
    L.info("task=%s run_dir=%s budget=%.0fs llm=%s", task_md, run_dir, settings.time_budget, settings.llm_mode)

    spec = parse_task(task_md)
    L.info("spec: %s", spec.summary())
    for w in spec.warnings:
        L.warning("spec: %s", w)
    write_json(run_dir / "task_spec.json", spec.to_dict())
    tpl = SubmissionTemplate.from_spec(spec)
    y = read_csv(Path(spec.train_path), usecols=spec.target_cols[:1], dtype=str)[spec.target_cols[0]]
    tpl.write(tpl.constant(y), "constant")
    L.info("constant fallback written at %.1fs", deadline.elapsed())
