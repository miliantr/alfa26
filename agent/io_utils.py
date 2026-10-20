"""File helpers: atomic writes, robust CSV reading, JSON with numpy support."""
from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().encode("utf-8")


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    atomic_write_bytes(path, df_to_csv_bytes(df))


class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (set, tuple)):
            return list(o)
        return str(o)


def _clean(o):
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def dumps(obj, indent: int | None = None) -> str:
    return json.dumps(_clean(obj), cls=_NpEncoder, ensure_ascii=False, indent=indent)


def write_json(path: Path, obj, indent: int = 2) -> None:
    atomic_write_text(path, dumps(obj, indent=indent))


def append_jsonl(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(dumps(obj) + "\n")


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def sniff_sep(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            head = f.read(65536)
        if head.count(",") >= max(1, head.count(";"), head.count("\t")):
            return ","
        return csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
    except Exception:
        return ","


def read_csv(path: Path, **kw) -> pd.DataFrame:
    """Read a CSV per the case spec (UTF-8, comma, dot decimal), tolerating BOM and
    other separators if the file clearly uses one."""
    path = Path(path)
    sep = kw.pop("sep", None) or sniff_sep(path)
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            return pd.read_csv(path, sep=sep, encoding=enc, low_memory=False, **kw)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise last_err  # type: ignore[misc]


def read_header(path: Path) -> list[str]:
    return list(read_csv(path, nrows=0).columns)
