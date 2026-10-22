"""Logging: UTF-8 file in the run dir plus stderr (never crashes on encoding)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s %(levelname).1s [%(name)s] %(message)s"


def setup(run_dir: Path | None, level: int = logging.INFO) -> logging.Logger:
    root = logging.getLogger("agent")
    root.setLevel(level)
    root.propagate = False
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter(_FMT, "%H:%M:%S"))
    root.addHandler(sh)
    if run_dir is not None:
        try:
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(Path(run_dir) / "agent.log", encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FMT))
            root.addHandler(fh)
        except Exception:
            pass
    return root


def get(name: str) -> logging.Logger:
    return logging.getLogger(f"agent.{name}")
