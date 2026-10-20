"""Sandbox for LLM-generated feature-engineering code.

The code must define ``transform(df) -> DataFrame`` (applied to train and test
separately) or ``transform(train, test) -> (DataFrame, DataFrame)`` and return only
*new* columns. It never receives the target. Layers of protection:

1. static AST check: import whitelist (subset of requirements.txt), no dangerous builtins
   (open/exec/eval/__import__/...), no dunder attribute access, no file/network I/O
   methods;
2. separate ``python -I`` process with a minimal environment, restricted ``__builtins__``,
   sockets disabled, address-space/CPU rlimits where the OS supports them;
3. parent-side timeout and RSS guard;
4. output validation: same row counts, same columns on train and test, numeric or
   categorical dtypes, no all-NaN / constant / inf-exploding columns.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import log
from .io_utils import read_json, write_json
from .resources import kill_tree, rss_tree

L = log.get("sandbox")

ALLOWED_IMPORTS = {"numpy", "pandas", "math", "re", "datetime", "itertools", "functools", "collections",
                   "statistics", "string", "scipy", "sklearn", "operator", "decimal", "fractions"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
                   "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help",
                   "memoryview", "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib",
                   "builtins", "pickle", "ctypes", "multiprocessing", "threading", "signal", "requests",
                   "urllib", "http", "httpx", "openai", "__builtins__", "__loader__", "__spec__"}
FORBIDDEN_ATTRS = {"to_csv", "to_pickle", "to_parquet", "to_feather", "to_sql", "to_excel", "to_hdf",
                   "to_json", "to_html", "to_clipboard", "read_csv", "read_pickle", "read_parquet",
                   "read_sql", "read_json", "read_excel", "read_html", "read_feather", "read_hdf",
                   "read_clipboard", "system", "popen", "spawn", "fork", "remove", "unlink", "rmdir",
                   "mkdir", "chmod", "load", "loads", "save", "savez", "savetxt", "loadtxt", "fromfile",
                   "tofile", "eval", "query", "pipe_input", "open"}


def check_code(code: str) -> list[str]:
    """Static checks; returns a list of problems (empty = acceptable)."""
    problems: list[str] = []
    if len(code) > 20000:
        return ["code too long"]
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    has_transform = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "transform":
            has_transform = True
            if not 1 <= len(node.args.args) <= 2:
                problems.append("transform must take (df) or (train, test)")
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"import not allowed: {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module or node.module.split(".")[0] not in ALLOWED_IMPORTS:
                problems.append(f"import not allowed: {node.module}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"name not allowed: {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRS:
                problems.append(f"attribute not allowed: {node.attr}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append("global/nonlocal not allowed")
        elif isinstance(node, (ast.AsyncFunctionDef, ast.Await)):
            problems.append("async code not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value and len(node.value) < 40 and node.value.startswith("__"):
            problems.append("dunder strings not allowed")
    if not has_transform:
        problems.append("no function named transform")
    return sorted(set(problems))


# The runner executes inside the sandbox process (python -I). It is our code; the
# untrusted code only ever runs through exec() with restricted builtins.
_RUNNER = r'''
import sys, json, builtins, time
cfg = json.loads(open(sys.argv[1], encoding="utf-8").read())
try:
    import resource
    lim = int(cfg["mem_bytes"])
    resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    resource.setrlimit(resource.RLIMIT_CPU, (int(cfg["cpu_seconds"]), int(cfg["cpu_seconds"]) + 5))
except Exception:
    pass
import numpy as np, pandas as pd
import socket
def _blocked(*a, **k):
    raise PermissionError("network disabled in sandbox")
socket.socket = _blocked
socket.create_connection = _blocked
train = pd.read_pickle(cfg["train"]); test = pd.read_pickle(cfg["test"])
allowed = set(cfg["allowed"])
_real_import = builtins.__import__
def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0 or name.split(".")[0] not in allowed:
        raise ImportError(f"import of {name} not allowed")
    return _real_import(name, globals, locals, fromlist, level)
safe = {k: getattr(builtins, k) for k in (
    "abs all any bool dict enumerate filter float frozenset int isinstance issubclass len list map max min "
    "next pow range reversed round set slice sorted str sum tuple zip print divmod hash iter repr ValueError "
    "TypeError KeyError IndexError ZeroDivisionError Exception ArithmeticError RuntimeError NotImplementedError "
    "True False None object staticmethod classmethod property super type callable chr ord format bin hex oct"
).split() if hasattr(builtins, k)}
safe["__import__"] = _guarded_import
ns = {"__builtins__": safe, "np": np, "pd": pd, "__name__": "llm_fe"}
exec(compile(cfg["code"], "<llm_fe>", "exec"), ns)
fn = ns["transform"]
import inspect
nargs = len(inspect.signature(fn).parameters)
t0 = time.time()
if nargs == 2:
    a, b = fn(train.copy(), test.copy())
else:
    a, b = fn(train.copy()), fn(test.copy())
if isinstance(a, pd.Series): a = a.to_frame()
if isinstance(b, pd.Series): b = b.to_frame()
if not isinstance(a, pd.DataFrame) or not isinstance(b, pd.DataFrame):
    raise TypeError("transform must return DataFrame(s)")
pd.to_pickle({"train": a, "test": b}, cfg["out"])
json.dump({"status": "ok", "seconds": time.time() - t0, "cols": [str(c) for c in a.columns]}, open(cfg["result"], "w"))
'''


def run_feature_code(code: str, train: pd.DataFrame, test: pd.DataFrame, work_dir: Path, name: str,
                     timeout: float = 90.0, mem_bytes: int = 4 * 1024 ** 3) -> dict:
    """Run LLM code; on success the validated new columns are pickled to ``out_path``."""
    problems = check_code(code)
    if problems:
        return {"status": "rejected", "error": "; ".join(problems)}
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    inp_tr, inp_te = work_dir / "in_train.pkl", work_dir / "in_test.pkl"
    if not inp_tr.exists():
        train.to_pickle(inp_tr)
        test.to_pickle(inp_te)
    d = work_dir / name
    d.mkdir(parents=True, exist_ok=True)
    cfg = {"train": str(inp_tr), "test": str(inp_te), "code": code, "out": str(d / "raw.pkl"),
           "result": str(d / "result.json"), "allowed": sorted(ALLOWED_IMPORTS),
           "mem_bytes": int(mem_bytes), "cpu_seconds": int(timeout * 2)}
    write_json(d / "cfg.json", cfg)
    (d / "runner.py").write_text(_RUNNER, encoding="utf-8")
    env = {"PATH": os.environ.get("PATH", os.defpath), "PYTHONIOENCODING": "utf-8",
           "OMP_NUM_THREADS": "2", "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
    t0 = time.time()
    with open(d / "stderr.log", "w", encoding="utf-8") as errf:
        proc = subprocess.Popen([sys.executable, "-I", str(d / "runner.py"), str(d / "cfg.json")],
                                cwd=str(d), env=env, stdout=errf, stderr=subprocess.STDOUT)
        status = None
        while proc.poll() is None:
            if time.time() - t0 > timeout:
                status = "timeout"
            elif rss_tree(proc.pid) > mem_bytes:
                status = "oom"
            if status:
                kill_tree(proc.pid)
                break
            time.sleep(0.2)
    if status:
        return {"status": status, "error": f"{status} after {time.time() - t0:.0f}s"}
    res = read_json(d / "result.json")
    if not res or res.get("status") != "ok":
        err = (d / "stderr.log").read_text(encoding="utf-8", errors="replace")[-1500:]
        return {"status": "error", "error": err or "no result"}
    raw = pd.read_pickle(d / "raw.pkl")
    try:
        a, b = validate_output(raw["train"], raw["test"], train, test, prefix=f"llm_{name}_")
    except ValueError as e:
        return {"status": "invalid", "error": str(e)}
    out_path = d / "columns.pkl"
    pd.to_pickle({"train": a, "test": b}, out_path)
    return {"status": "ok", "path": str(out_path), "columns": list(a.columns),
            "seconds": round(time.time() - t0, 2)}


def validate_output(a: pd.DataFrame, b: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame,
                    prefix: str = "llm_", max_cols: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(a) != len(train) or len(b) != len(test):
        raise ValueError(f"row count changed ({len(a)}/{len(train)}, {len(b)}/{len(test)})")
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    new = [c for c in a.columns if c not in train.columns and c in b.columns]
    if not new:
        raise ValueError("no new columns (return only the new features, same names for train/test)")
    keep_a, keep_b = {}, {}
    for c in new[:max_cols]:
        sa, sb = a[c], b[c]
        name = (prefix + "".join(ch if ch.isalnum() else "_" for ch in str(c)))[:70]
        if sa.dtype == object or isinstance(sa.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(sa):
            va, vb = sa.astype(str), sb.astype(str)
            vocab = pd.Index(sorted(set(va.unique()) | set(vb.unique())))
            if len(vocab) <= 1 or len(vocab) > 0.5 * len(a) + 10:
                continue
            keep_a[name] = pd.Categorical(va, categories=vocab)
            keep_b[name] = pd.Categorical(vb, categories=vocab)
            continue
        va = pd.to_numeric(sa, errors="coerce").astype("float64")
        vb = pd.to_numeric(sb, errors="coerce").astype("float64")
        if np.isinf(va).mean() > 0.01 or np.isinf(vb).mean() > 0.01:
            continue  # inf explosion
        va, vb = va.replace([np.inf, -np.inf], np.nan), vb.replace([np.inf, -np.inf], np.nan)
        if va.isna().mean() > 0.95 or va.nunique() <= 1:
            continue
        if vb.isna().mean() > max(0.95, va.isna().mean() + 0.5):
            continue
        keep_a[name], keep_b[name] = va.astype("float32"), vb.astype("float32")
    if not keep_a:
        raise ValueError("all new columns were empty, constant, non-finite or unusable")
    return pd.DataFrame(keep_a), pd.DataFrame(keep_b)
