"""Task understanding: locate files, find id/target columns, infer task type, metric and
the expected submission format from task.md + data_dictionary + sample_submission + data.

Everything here is deterministic. The optional LLM reading of the description is merged
in by :func:`merge_llm_view`, where heuristics win on disagreement (and it is logged).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import log, metrics
from .io_utils import read_csv, read_header

L = log.get("task")

FILE_KINDS = ("train", "test", "sample", "dictionary")
_PATH_RE = re.compile(r"[`'\"(\[]?([\w\-./\\]+\.(?:csv|tsv|parquet|txt))[`'\")\]]?", re.IGNORECASE)
_OUTPUT_CTX = ("формат результата", "формат ответа", "результат", "сохран", "запиш", "output",
               "submission", "предсказан", "ответ")
_OUTPUT_NAMES = ("submission", "submit", "predict", "answer", "result", "output", "ответ", "прогноз")


@dataclass
class TaskSpec:
    task_md: str
    task_dir: str
    title: str = ""
    text: str = ""
    train_path: str | None = None
    test_path: str | None = None
    sample_path: str | None = None
    dict_path: str | None = None
    output_paths: list[str] = field(default_factory=list)
    id_col: str | None = None
    target_cols: list[str] = field(default_factory=list)
    submission_cols: list[str] = field(default_factory=list)
    task_type: str = metrics.REGRESSION  # of the (first) target
    target_types: dict = field(default_factory=dict)
    metric: str = "rmse"
    metric_source: str = "default"
    mentioned_metrics: list[str] = field(default_factory=list)
    output_kind: str = "value"  # value | label | proba | proba_wide
    proba_columns: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    positive_label: str | None = None
    label_dtype: str = "str"  # how labels are written: "int" | "float" | "str" | "bool"
    dictionary: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    llm_view: dict = field(default_factory=dict)

    def to_dict(self, with_text: bool = False) -> dict:
        d = asdict(self)
        if not with_text:
            d.pop("text", None)
        return d

    def summary(self) -> str:
        return (f"target={self.target_cols} type={self.task_type} metric={self.metric}"
                f"({self.metric_source}) output={self.output_kind} id={self.id_col} "
                f"classes={len(self.classes)}")


# ----------------------------------------------------------------------- file lookup
def _classify_name(name: str) -> str | None:
    n = name.lower()
    if "sample" in n or "template" in n or "шаблон" in n:
        return "sample"
    if "dictionary" in n or "dict" in n or "description" in n or "schema" in n:
        return "dictionary"
    if "train" in n:
        return "train"
    if "test" in n:
        return "test"
    return None


def locate_files(task_md: Path, text: str) -> tuple[dict, list[Path], list[str]]:
    task_dir = task_md.parent
    found: dict[str, Path] = {}
    warnings: list[str] = []
    outputs: list[Path] = []
    low = text.lower()
    for m in _PATH_RE.finditer(text):
        token = m.group(1).replace("\\", "/")
        p = (task_dir / token).resolve()
        kind = _classify_name(Path(token).name)
        if p.exists() and p.is_file() and kind in FILE_KINDS:
            found.setdefault(kind, p)
            continue
        # The output target is a submission-like name that is not an input, e.g.
        # "Файл `submission.csv` в каталоге этого Markdown-файла". Names that classify as
        # train/test/sample/dictionary are never outputs, even if they do not exist.
        name = Path(token).name.lower()
        ctx = low[max(0, m.start() - 200): m.end() + 80]
        if (kind is None and Path(token).suffix.lower() == ".csv"
                and any(w in name for w in _OUTPUT_NAMES) and any(c in ctx for c in _OUTPUT_CTX)):
            outputs.append((task_dir / token).resolve())
    # Fallback: recursive search for the standard names (shallowest wins).
    if len(found) < 4:
        cands: dict[str, list[Path]] = {k: [] for k in FILE_KINDS}
        for p in task_dir.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in (".csv", ".tsv", ".parquet"):
                continue
            if any(part.startswith(".") or part in ("agent_run", "__pycache__") for part in p.relative_to(task_dir).parts[:-1]):
                continue
            kind = _classify_name(p.name)
            if kind:
                cands[kind].append(p)
        for kind, ps in cands.items():
            if kind not in found and ps:
                ps.sort(key=lambda q: (len(q.relative_to(task_dir).parts), len(q.name)))
                found[kind] = ps[0]
                warnings.append(f"{kind} file found by search: {ps[0].relative_to(task_dir)}")
    # Avoid treating our own output as the sample.
    default_out = (task_dir / "submission.csv").resolve()
    if found.get("sample") == default_out:
        found.pop("sample")
    outs = []
    for o in outputs + [default_out]:
        if o not in outs and o not in found.values():
            outs.append(o)
    return found, outs, warnings


# ------------------------------------------------------------------ column inference
def _load_dictionary(path: Path | None) -> list[dict]:
    if not path:
        return []
    try:
        d = read_csv(path)
    except Exception as e:
        L.warning("cannot read dictionary %s: %s", path, e)
        return []
    d.columns = [str(c).strip().lower() for c in d.columns]
    if "feature" not in d.columns:
        # Use the first column as the feature name whatever it is called.
        d = d.rename(columns={d.columns[0]: "feature"})
    rows = []
    for r in d.to_dict("records"):
        rows.append({k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()})
    return rows


def _dict_role(rows: list[dict], role: str) -> list[str]:
    out = []
    for r in rows:
        v = str(r.get("role") or "").strip().lower()
        if v == role or (role == "target" and v in ("target", "label", "таргет", "целевая")):
            out.append(str(r["feature"]))
    return out


def _guess_id(train_cols, test_cols, sample_cols, dict_rows, test_head: pd.DataFrame | None) -> str | None:
    for c in _dict_role(dict_rows, "id"):
        if c in test_cols:
            return c
    if sample_cols and sample_cols[0] in test_cols:
        return sample_cols[0]
    for c in test_cols:
        if str(c).lower() in ("id", "row_id", "rowid", "index", "uid", "sample_id", "object_id"):
            return c
    if test_head is not None:
        for c in test_cols:
            lc = str(c).lower()
            if (lc.endswith("_id") or lc.startswith("id_")) and test_head[c].is_unique:
                if sample_cols and c in sample_cols:
                    return c
    return None


def _numeric(s: pd.Series) -> pd.Series | None:
    v = pd.to_numeric(s, errors="coerce")
    if v.notna().sum() >= max(1, int(0.99 * s.notna().sum())):
        return v
    return None


def _infer_target_type(y_raw: pd.Series, metric_hint: str | None, n_rows: int) -> str:
    """regression / binary / multiclass from the raw (string-read) target values."""
    y = y_raw.dropna()
    nun = y.nunique()
    m = metrics.REGISTRY.get(metric_hint) if metric_hint else None
    num = _numeric(y)
    if num is None:
        return metrics.BINARY if nun == 2 else metrics.MULTICLASS
    if m is not None and metrics.REGRESSION in m.task_types:
        return metrics.REGRESSION
    if nun == 2:
        return metrics.BINARY
    if m is not None and metrics.MULTICLASS in m.task_types:
        return metrics.MULTICLASS
    is_int = bool(np.all(np.isclose(num.dropna() % 1, 0)))
    if is_int and nun <= 30 and nun <= max(3, 0.01 * n_rows):
        return metrics.MULTICLASS
    return metrics.REGRESSION


def _label_dtype(values: pd.Series) -> str:
    v = values.dropna().astype(str)
    if v.empty:
        return "str"
    if v.str.lower().isin(["true", "false"]).all():
        return "bool"
    if v.str.fullmatch(r"[+-]?\d+").all():
        return "int"
    if pd.to_numeric(v, errors="coerce").notna().all():
        return "float"
    return "str"


_POS_NAMES = ("1", "1.0", "true", "yes", "да", "y", "positive", "pos", "attack", "fraud", "default", "churn")


def _positive_label(classes: list[str], counts: pd.Series, text: str) -> str:
    low = {c.lower(): c for c in classes}
    for n in _POS_NAMES:
        if n in low:
            return low[n]
    # "вероятность класса X" / "probability of X" in the description
    for c in classes:
        if re.search(rf"(вероятност\w*|probability)[^\n.]{{0,40}}\b{re.escape(c.lower())}\b", text.lower()):
            return c
    return str(counts.idxmin())  # minority class is the usual "positive"


def _output_kind(task_type, metric_name, sample: pd.DataFrame | None, id_col, targets, classes, text) -> tuple[str, list[str]]:
    if task_type == metrics.REGRESSION:
        return "value", []
    m = metrics.REGISTRY[metric_name]
    if sample is not None:
        value_cols = [c for c in sample.columns if c != id_col]
        cls_set = {str(c) for c in classes}
        if len(value_cols) > 1 and set(map(str, value_cols)) <= cls_set | {str(t) for t in targets} and not (set(value_cols) <= set(targets)):
            return "proba_wide", [str(c) for c in value_cols]
        if len(value_cols) >= 1 and value_cols[0] in targets:
            col = sample[value_cols[0]].dropna().astype(str)
            num = pd.to_numeric(col, errors="coerce")
            labels_like = col.isin(cls_set).mean() if len(col) else 0
            frac = num.notna().all() and len(num) and (num % 1 != 0).any() and num.between(0, 1).all()
            if frac and labels_like < 0.99:
                return "proba", []
            if m.needs == "proba" and task_type == metrics.BINARY:
                return "proba", []
            if labels_like >= 0.99:
                return "label", []
    if m.needs == "proba" and task_type == metrics.BINARY:
        return "proba", []
    if re.search(r"вероятност\w* (класса|принадлежност|того, что)|probabilit(y|ies) of", text.lower()) and task_type == metrics.BINARY:
        return "proba", []
    return "label", []


def fix_metric_for_task(metric_name: str | None, task_type: str) -> tuple[str, str | None]:
    """Make the metric compatible with the task type; returns (metric, note)."""
    if metric_name and metrics.compatible(metric_name, task_type):
        return metric_name, None
    remap = {
        (metrics.MULTICLASS, "f1"): "f1_macro",
        (metrics.BINARY, "f1_macro"): "f1_macro",
    }
    if metric_name and (task_type, metric_name) in remap:
        return remap[(task_type, metric_name)], f"metric {metric_name} remapped for {task_type}"
    d = metrics.default_metric(task_type)
    if metric_name:
        return d, f"metric {metric_name} incompatible with {task_type}; using {d}"
    return d, f"no metric found in description; using default {d}"


# ------------------------------------------------------------------------- main entry
def parse_task(task_md: Path) -> TaskSpec:
    task_md = Path(task_md).resolve()
    text = ""
    try:
        text = task_md.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        L.warning("cannot read task file %s: %s", task_md, e)
    spec = TaskSpec(task_md=str(task_md), task_dir=str(task_md.parent), text=text)
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    spec.title = first.lstrip("# ").strip()[:200]

    files, outs, warns = locate_files(task_md, text)
    spec.warnings += warns
    for k, attr in (("train", "train_path"), ("test", "test_path"), ("sample", "sample_path"), ("dictionary", "dict_path")):
        if k in files:
            setattr(spec, attr, str(files[k]))
    spec.output_paths = [str(p) for p in outs]
    if not spec.train_path or not spec.test_path:
        spec.warnings.append("train/test file not found")
        return spec

    train_cols = read_header(Path(spec.train_path))
    test_cols = read_header(Path(spec.test_path))
    sample = None
    if spec.sample_path:
        try:
            sample = read_csv(Path(spec.sample_path), dtype=str, keep_default_na=False, nrows=20000)
        except Exception as e:
            spec.warnings.append(f"cannot read sample submission: {e}")
    sample_cols = list(sample.columns) if sample is not None else []
    spec.dictionary = _load_dictionary(Path(spec.dict_path) if spec.dict_path else None)
    test_head = read_csv(Path(spec.test_path), nrows=5000)
    spec.id_col = _guess_id(train_cols, test_cols, sample_cols, spec.dictionary, test_head)

    # Targets: dictionary role -> columns missing from test -> sample columns in train.
    targets = [c for c in _dict_role(spec.dictionary, "target") if c in train_cols]
    if not targets:
        targets = [c for c in train_cols if c not in test_cols and c != spec.id_col]
    if not targets and sample_cols:
        targets = [c for c in sample_cols if c in train_cols and c != spec.id_col]
    if len(targets) > 1 and sample_cols:
        in_sample = [c for c in targets if c in sample_cols]
        if in_sample:
            targets = in_sample
    spec.target_cols = targets
    if not targets:
        spec.warnings.append("target column not identified")
        return spec

    metric_text, mentioned = metrics.extract_primary_metric(text)
    spec.mentioned_metrics = mentioned
    y_raw = read_csv(Path(spec.train_path), usecols=targets, dtype=str, keep_default_na=True)
    n_rows = len(y_raw)
    for t in targets:
        spec.target_types[t] = _infer_target_type(y_raw[t], metric_text, n_rows)
    spec.task_type = spec.target_types[targets[0]]

    metric_name, note = fix_metric_for_task(metric_text, spec.task_type)
    spec.metric = metric_name
    spec.metric_source = "text" if metric_text and not note else ("text-remapped" if metric_text else "default")
    if note:
        spec.warnings.append(note)

    if spec.task_type in metrics.CLASSIFICATION:
        t0 = targets[0]
        counts = y_raw[t0].dropna().value_counts()
        spec.label_dtype = _label_dtype(y_raw[t0])
        classes = list(counts.index.astype(str))
        if spec.label_dtype in ("int", "float"):
            classes.sort(key=lambda c: float(c))
        else:
            classes.sort()
        spec.classes = classes
        if spec.task_type == metrics.BINARY:
            spec.positive_label = _positive_label(classes, counts, text)
    spec.output_kind, spec.proba_columns = _output_kind(
        spec.task_type, spec.metric, sample, spec.id_col, targets, spec.classes, text)
    spec.submission_cols = sample_cols or ([spec.id_col] if spec.id_col else []) + targets
    if spec.output_kind == "proba_wide":
        missing = set(spec.proba_columns) - set(spec.classes)
        if missing:
            spec.warnings.append(f"sample columns not among classes: {sorted(missing)}")
    return spec


def sub_spec(spec: TaskSpec, target: str, out_path: Path) -> TaskSpec:
    """Single-target view of a multi-target task (own type, classes, output kind)."""
    import copy

    s = copy.deepcopy(spec)
    s.target_cols = [target]
    s.task_type = spec.target_types.get(target, spec.task_type)
    s.submission_cols = [c for c in ([spec.id_col] if spec.id_col else []) + [target]]
    s.output_paths = [str(out_path)]
    s.classes, s.positive_label, s.label_dtype = [], None, "str"
    y = read_csv(Path(spec.train_path), usecols=[target], dtype=str)[target]
    metric_name, _ = fix_metric_for_task(spec.metric, s.task_type)
    s.metric = metric_name
    if s.task_type in metrics.CLASSIFICATION:
        counts = y.dropna().value_counts()
        s.label_dtype = _label_dtype(y)
        cls = list(counts.index.astype(str))
        cls.sort(key=(lambda c: float(c)) if s.label_dtype in ("int", "float") else None)
        s.classes = cls
        if s.task_type == metrics.BINARY:
            s.positive_label = _positive_label(cls, counts, spec.text)
    sample = None
    if spec.sample_path:
        try:
            sample = read_csv(Path(spec.sample_path), dtype=str, keep_default_na=False, nrows=20000)
            sample = sample[[c for c in s.submission_cols if c in sample.columns]]
        except Exception:
            sample = None
    s.output_kind, s.proba_columns = _output_kind(s.task_type, s.metric, sample, s.id_col, [target], s.classes, spec.text)
    if s.output_kind == "proba_wide":  # wide probabilities are not defined per target
        s.output_kind = "proba" if s.task_type == metrics.BINARY else "label"
    return s


# --------------------------------------------------------------------- LLM cross-check
LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "task_type": {"type": "string", "enum": ["regression", "binary", "multiclass", "multilabel", "other"]},
        "target_columns": {"type": "array", "items": {"type": "string"}},
        "id_column": {"type": ["string", "null"]},
        "metric": {"type": "string"},
        "secondary_metrics": {"type": "array", "items": {"type": "string"}},
        "output_format": {"type": "string", "enum": ["value", "label", "proba", "proba_wide"]},
        "positive_label": {"type": ["string", "null"]},
        "time_column": {"type": ["string", "null"]},
        "group_column": {"type": ["string", "null"]},
        "special_requirements": {"type": "array", "items": {"type": "string"}},
        "domain_notes": {"type": "string"},
    },
    "required": ["task_type", "target_columns", "metric"],
}


PARSE_SYSTEM = (
    "You read machine-learning task descriptions (often in Russian) and extract a structured spec. "
    "Reply with JSON only, matching: {\"task_type\": regression|binary|multiclass|multilabel|other, "
    "\"target_columns\": [..], \"id_column\": str|null, \"metric\": str (the PRIMARY scoring metric, e.g. "
    "rmse, mae, rmsle, mape, r2, f1_macro, f1_weighted, f1, accuracy, roc_auc, pr_auc, logloss, qwk), "
    "\"secondary_metrics\": [..], \"output_format\": value|label|proba|proba_wide, \"positive_label\": str|null, "
    "\"time_column\": str|null (only if the test lies after train in time), \"group_column\": str|null, "
    "\"special_requirements\": [..], \"domain_notes\": str (<=60 words of domain knowledge useful for features)}"
)


def llm_parse_prompt(spec: TaskSpec, max_chars: int = 6000) -> str:
    cols = []
    try:
        cols = read_header(Path(spec.train_path)) if spec.train_path else []
    except Exception:
        pass
    sample_head = ""
    if spec.sample_path:
        try:
            sample_head = Path(spec.sample_path).read_text(encoding="utf-8", errors="replace").splitlines()[:3]
        except Exception:
            pass
    roles = {str(r.get("feature")): r.get("role") for r in spec.dictionary if r.get("role") not in (None, "feature")}
    return (f"TASK DESCRIPTION:\n{spec.text[:max_chars]}\n\nTRAIN COLUMNS ({len(cols)}): {cols[:80]}\n"
            f"DICTIONARY ROLES (non-feature): {roles}\nSAMPLE SUBMISSION HEAD: {sample_head}\n"
            f"target candidates: {spec.target_cols}\nReturn the JSON spec.")


def merge_llm_view(spec: TaskSpec, view: dict) -> TaskSpec:
    """Cross-check the LLM's reading against the heuristics. Heuristics win on conflict;
    the LLM only fills gaps (e.g. no metric found in the text)."""
    if not view:
        return spec
    spec.llm_view = view
    llm_metric = metrics.normalize_name(view.get("metric"))
    if llm_metric:
        llm_metric, _ = fix_metric_for_task(llm_metric, spec.task_type)
    if spec.metric_source == "default" and llm_metric:
        L.info("metric: text had none, using LLM's %s", llm_metric)
        spec.metric, spec.metric_source = llm_metric, "llm"
        spec.output_kind, spec.proba_columns = _refresh_output_kind(spec)
    elif llm_metric and llm_metric != spec.metric:
        msg = f"LLM metric {llm_metric} != heuristic {spec.metric}; keeping heuristic"
        L.warning(msg)
        spec.warnings.append(msg)
    lt = view.get("task_type")
    if lt and lt != spec.task_type and lt in (metrics.REGRESSION, metrics.BINARY, metrics.MULTICLASS):
        msg = f"LLM task_type {lt} != heuristic {spec.task_type}; keeping heuristic"
        L.warning(msg)
        spec.warnings.append(msg)
    ltargets = [c for c in (view.get("target_columns") or []) if isinstance(c, str)]
    if ltargets and set(ltargets) != set(spec.target_cols):
        spec.warnings.append(f"LLM targets {ltargets} != heuristic {spec.target_cols}; keeping heuristic")
    if spec.task_type == metrics.BINARY and view.get("positive_label") in spec.classes:
        if view["positive_label"] != spec.positive_label and spec.positive_label not in _POS_NAMES:
            spec.warnings.append(f"positive label from LLM: {view['positive_label']}")
            spec.positive_label = view["positive_label"]
    return spec


def _refresh_output_kind(spec: TaskSpec):
    sample = None
    if spec.sample_path:
        try:
            sample = read_csv(Path(spec.sample_path), dtype=str, keep_default_na=False, nrows=20000)
        except Exception:
            pass
    return _output_kind(spec.task_type, spec.metric, sample, spec.id_col, spec.target_cols, spec.classes, spec.text)
