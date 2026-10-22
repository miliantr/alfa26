"""Metric registry.

Every metric knows its direction, which prediction it consumes (continuous value, class
label or class probabilities) and which training objective suits it. Classification
metrics receive encoded integer labels ``0..K-1`` and an ``n x K`` probability matrix;
the evaluator turns probabilities into labels when the metric needs labels.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn import metrics as skm

REGRESSION = "regression"
BINARY = "binary"
MULTICLASS = "multiclass"
CLASSIFICATION = (BINARY, MULTICLASS)


@dataclass(frozen=True)
class Metric:
    name: str
    task_types: tuple[str, ...]
    greater_is_better: bool
    needs: str  # "value" | "label" | "proba"
    fn: Callable[[np.ndarray, np.ndarray], float]
    objective: str  # hint for model objective / post-processing
    bounded_one: bool = False  # perfect score is 1.0 (higher-is-better metrics)
    patterns: tuple[str, ...] = ()

    def better(self, a: float, b: float) -> bool:
        """True if score ``a`` is strictly better than ``b``."""
        if b is None or not np.isfinite(b):
            return a is not None and np.isfinite(a)
        if a is None or not np.isfinite(a):
            return False
        return a > b if self.greater_is_better else a < b

    def sign(self) -> float:
        """+1 if higher is better, -1 otherwise; ``sign * score`` is to be maximised."""
        return 1.0 if self.greater_is_better else -1.0


# ---------------------------------------------------------------- regression functions
def _rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def _mse(y, p):
    return float(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2))


def _mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def _medae(y, p):
    return float(np.median(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def _rmsle(y, p):
    y = np.clip(np.asarray(y, float), 0, None)
    p = np.clip(np.asarray(p, float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(p)) ** 2)))


def _msle(y, p):
    return _rmsle(y, p) ** 2


def _mape(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    denom = np.maximum(np.abs(y), np.finfo(float).eps)
    return float(np.mean(np.abs(y - p) / denom))


def _smape(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    denom = np.abs(y) + np.abs(p)
    out = np.where(denom == 0, 0.0, 2.0 * np.abs(y - p) / np.where(denom == 0, 1, denom))
    return float(np.mean(out))


def _wape(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(np.sum(np.abs(y - p)) / max(np.sum(np.abs(y)), np.finfo(float).eps))


def _r2(y, p):
    return float(skm.r2_score(y, p))


# ------------------------------------------------------------ classification functions
def _labels(p):
    p = np.asarray(p)
    return p.argmax(axis=1) if p.ndim == 2 else p


def _accuracy(y, lab):
    return float(skm.accuracy_score(y, lab))


def _bal_acc(y, lab):
    return float(skm.balanced_accuracy_score(y, lab))


def _f1_binary(y, lab):
    return float(skm.f1_score(y, lab, average="binary", pos_label=1, zero_division=0))


def _f1_macro(y, lab):
    return float(skm.f1_score(y, lab, average="macro", zero_division=0))


def _f1_weighted(y, lab):
    return float(skm.f1_score(y, lab, average="weighted", zero_division=0))


def _f1_micro(y, lab):
    return float(skm.f1_score(y, lab, average="micro", zero_division=0))


def _precision_macro(y, lab):
    return float(skm.precision_score(y, lab, average="macro", zero_division=0))


def _recall_macro(y, lab):
    return float(skm.recall_score(y, lab, average="macro", zero_division=0))


def _mcc(y, lab):
    return float(skm.matthews_corrcoef(y, lab))


def _kappa(y, lab):
    return float(skm.cohen_kappa_score(y, lab))


def _qwk(y, lab):
    return float(skm.cohen_kappa_score(y, lab, weights="quadratic"))


def _proba_matrix(p, k=None):
    p = np.asarray(p, float)
    if p.ndim == 1:
        p = np.column_stack([1 - p, p])
    return p


def _roc_auc(y, p):
    p = _proba_matrix(p)
    y = np.asarray(y)
    if p.shape[1] == 2:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(skm.roc_auc_score(y, p[:, 1]))
    present = np.unique(y)
    if len(present) < 2:
        return float("nan")
    # macro one-vs-rest over the classes present in y_true
    scores = []
    for k in present:
        yk = (y == k).astype(int)
        if 0 < yk.sum() < len(yk):
            scores.append(skm.roc_auc_score(yk, p[:, int(k)]))
    return float(np.mean(scores)) if scores else float("nan")


def _gini(y, p):
    return 2.0 * _roc_auc(y, p) - 1.0


def _pr_auc(y, p):
    p = _proba_matrix(p)
    y = np.asarray(y)
    if p.shape[1] == 2:
        return float(skm.average_precision_score(y, p[:, 1]))
    scores = []
    for k in np.unique(y):
        yk = (y == k).astype(int)
        scores.append(skm.average_precision_score(yk, p[:, int(k)]))
    return float(np.mean(scores))


def _logloss(y, p):
    p = _proba_matrix(p)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    p = p / p.sum(axis=1, keepdims=True)
    y = np.asarray(y, int)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def _brier(y, p):
    p = _proba_matrix(p)
    y = np.asarray(y, int)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1
    if p.shape[1] == 2:
        return float(np.mean((p[:, 1] - onehot[:, 1]) ** 2))
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


R, B, M = (REGRESSION,), (BINARY,), (MULTICLASS,)
BM = (BINARY, MULTICLASS)

# Patterns are matched case-insensitively against the task text; more specific metrics
# come first in REGISTRY order so that e.g. "RMSLE" wins over "RMSE".
REGISTRY: dict[str, Metric] = {}


def _reg(m: Metric) -> None:
    REGISTRY[m.name] = m


_reg(Metric("rmsle", R, False, "value", _rmsle, "log_l2",
            patterns=(r"\brmsle\b", r"root mean squared log", r"среднеквадратичн\w* логарифм")))
_reg(Metric("msle", R, False, "value", _msle, "log_l2",
            patterns=(r"\bmsle\b", r"mean squared log", r"mean_squared_log_error")))
_reg(Metric("rmse", R, False, "value", _rmse, "l2",
            patterns=(r"\brmse\b", r"root mean squared error", r"корень из средне\w* квадрат",
                      r"sqrt\(\s*mean\(\s*\(\s*y_true\s*-\s*y_pred\s*\)\s*\*\*\s*2",
                      r"squared\s*=\s*false", r"root_mean_squared_error")))
_reg(Metric("mse", R, False, "value", _mse, "l2",
            patterns=(r"\bmse\b", r"mean squared error", r"mean_squared_error",
                      r"среднеквадратичн\w* ошибк", r"средн\w* квадрат\w* ошибк")))
_reg(Metric("smape", R, False, "value", _smape, "l1",
            patterns=(r"\bsmape\b", r"symmetric mean absolute percentage")))
_reg(Metric("mape", R, False, "value", _mape, "mape",
            patterns=(r"\bmape\b", r"mean absolute percentage", r"mean_absolute_percentage_error",
                      r"средн\w* абсолютн\w* процентн")))
_reg(Metric("wape", R, False, "value", _wape, "l1",
            patterns=(r"\bwape\b", r"weighted absolute percentage")))
_reg(Metric("medae", R, False, "value", _medae, "l1",
            patterns=(r"median absolute error", r"median_absolute_error", r"\bmedae\b",
                      r"медианн\w* абсолютн")))
_reg(Metric("mae", R, False, "value", _mae, "l1",
            patterns=(r"\bmae\b", r"mean absolute error", r"mean_absolute_error",
                      r"средн\w* абсолютн\w* ошибк")))
_reg(Metric("r2", R, True, "value", _r2, "l2", bounded_one=True,
            patterns=(r"\br2\b", r"\br\^2\b", r"r²", r"r2_score", r"коэффициент\w* детерминац",
                      r"r-squared", r"r squared")))

_reg(Metric("f1_macro", BM, True, "label", _f1_macro, "f1_macro", bounded_one=True,
            patterns=(r"macro[\s\-_]*f1", r"f1[\s\-_]*macro", r"macro[\s\-_]*averaged f1",
                      r"average\s*=\s*['\"]macro['\"]", r"макро[\s\-]*f1", r"f1[\s\-]*макро")))
_reg(Metric("f1_weighted", BM, True, "label", _f1_weighted, "f1_weighted", bounded_one=True,
            patterns=(r"weighted[\s\-_]*f1", r"f1[\s\-_]*weighted", r"average\s*=\s*['\"]weighted['\"]",
                      r"взвешенн\w*[\s\-]*f1")))
_reg(Metric("f1_micro", BM, True, "label", _f1_micro, "f1_micro", bounded_one=True,
            patterns=(r"micro[\s\-_]*f1", r"f1[\s\-_]*micro", r"average\s*=\s*['\"]micro['\"]",
                      r"микро[\s\-]*f1")))
_reg(Metric("qwk", BM, True, "label", _qwk, "qwk", bounded_one=True,
            patterns=(r"quadratic[\s\-_]*weighted[\s\-_]*kappa", r"\bqwk\b",
                      r"weights\s*=\s*['\"]quadratic['\"]", r"квадратичн\w*\s+\w*\s*капп")))
_reg(Metric("kappa", BM, True, "label", _kappa, "kappa", bounded_one=True,
            patterns=(r"cohen'?s?[\s\-_]*kappa", r"cohen_kappa", r"\bkappa\b", r"каппа")))
_reg(Metric("mcc", BM, True, "label", _mcc, "mcc", bounded_one=True,
            patterns=(r"\bmcc\b", r"matthews", r"мэттьюс")))
_reg(Metric("balanced_accuracy", BM, True, "label", _bal_acc, "balanced", bounded_one=True,
            patterns=(r"balanced[\s\-_]*accuracy", r"сбалансированн\w* точност")))
_reg(Metric("pr_auc", BM, True, "proba", _pr_auc, "proba", bounded_one=True,
            patterns=(r"pr[\s\-_]*auc", r"average[\s\-_]*precision", r"auprc", r"auc[\s\-_]*pr",
                      r"precision[\s\-]*recall auc")))
_reg(Metric("gini", BM, True, "proba", _gini, "proba", bounded_one=True,
            patterns=(r"\bgini\b", r"джини")))
_reg(Metric("roc_auc", BM, True, "proba", _roc_auc, "proba", bounded_one=True,
            patterns=(r"roc[\s\-_]*auc", r"auc[\s\-_]*roc", r"roc_auc_score", r"\bauc\b",
                      r"площадь под roc")))
_reg(Metric("logloss", BM, False, "proba", _logloss, "logloss",
            patterns=(r"log[\s\-_]*loss", r"cross[\s\-_]*entropy", r"log_loss", r"кросс[\s\-]*энтроп",
                      r"логарифмическ\w* потер")))
_reg(Metric("brier", BM, False, "proba", _brier, "proba",
            patterns=(r"brier",)))
_reg(Metric("precision_macro", BM, True, "label", _precision_macro, "f1_macro", bounded_one=True,
            patterns=(r"macro[\s\-_]*precision", r"precision[\s\-_]*macro")))
_reg(Metric("recall_macro", BM, True, "label", _recall_macro, "balanced", bounded_one=True,
            patterns=(r"macro[\s\-_]*recall", r"recall[\s\-_]*macro")))
_reg(Metric("f1", B, True, "label", _f1_binary, "f1_binary", bounded_one=True,
            patterns=(r"\bf1[\s\-_]*score\b", r"\bf1\b", r"f-?мера", r"f-measure")))
_reg(Metric("accuracy", BM, True, "label", _accuracy, "accuracy", bounded_one=True,
            patterns=(r"\baccuracy\b", r"accuracy_score", r"доля правильн\w* ответ")))

ALIASES = {
    "root_mean_squared_error": "rmse", "neg_root_mean_squared_error": "rmse",
    "mean_squared_error": "mse", "mean_absolute_error": "mae", "l2": "mse", "l1": "mae",
    "mean_absolute_percentage_error": "mape", "r2_score": "r2", "r^2": "r2", "r²": "r2",
    "macro_f1": "f1_macro", "f1macro": "f1_macro", "f1-macro": "f1_macro", "macro f1": "f1_macro",
    "weighted_f1": "f1_weighted", "micro_f1": "f1_micro", "f1_score": "f1", "f1-score": "f1",
    "auc": "roc_auc", "roc-auc": "roc_auc", "rocauc": "roc_auc", "roc auc": "roc_auc",
    "auc_roc": "roc_auc", "average_precision": "pr_auc", "pr-auc": "pr_auc",
    "log_loss": "logloss", "cross_entropy": "logloss", "crossentropy": "logloss",
    "cohen_kappa": "kappa", "quadratic_weighted_kappa": "qwk", "acc": "accuracy",
    "balanced_acc": "balanced_accuracy", "matthews_corrcoef": "mcc",
}


def normalize_name(name: str | None) -> str | None:
    """Map a free-form metric name to a registry key (None if unknown)."""
    if not name:
        return None
    key = str(name).strip().lower().replace(" ", "_")
    key = key.replace("-", "_")
    if key in REGISTRY:
        return key
    for raw in (str(name).strip().lower(), key, key.replace("_", " "), key.replace("_", "-")):
        if raw in ALIASES:
            return ALIASES[raw]
    # Last resort: run the text patterns over the name itself.
    found = find_metric_mentions(str(name))
    return found[0][0] if found else None


def get(name: str) -> Metric:
    key = normalize_name(name)
    if key is None:
        raise KeyError(f"unknown metric: {name!r}")
    return REGISTRY[key]


def default_metric(task_type: str) -> str:
    """Fallback when the description names no metric we recognise."""
    return {REGRESSION: "rmse", BINARY: "roc_auc", MULTICLASS: "f1_macro"}.get(task_type, "rmse")


def compatible(metric_name: str, task_type: str) -> bool:
    return task_type in REGISTRY[metric_name].task_types


def find_metric_mentions(text: str) -> list[tuple[str, int]]:
    """All ``(metric, char_offset)`` mentions in ``text``, without overlapping spans."""
    low = text.lower()
    taken: list[tuple[int, int]] = []
    out: list[tuple[str, int]] = []
    for name, m in REGISTRY.items():
        for pat in m.patterns:
            for mt in re.finditer(pat, low):
                s, e = mt.span()
                if any(s < te and e > ts for ts, te in taken):
                    continue
                taken.append((s, e))
                out.append((name, s))
    out.sort(key=lambda t: t[1])
    return out


_PRIMARY_HINTS = ("основн", "главн", "метрика оценки", "метрика:", "метрика —", "метрика -",
                  "оценива", "primary", "evaluation metric", "evaluated", "scored", "score",
                  "учитывается в итогов", "лидерборд")
_SECONDARY_HINTS = ("дополнительн", "справочн", "также рассчитыва", "не учитыва", "additional",
                    "reference", "вместо", "а не ", "не отражают")


def extract_primary_metric(text: str) -> tuple[str | None, list[str]]:
    """Pick the metric that the description names as the one used for scoring.

    Returns ``(metric_or_None, all_mentioned_metrics)``. Each mention is scored by the
    words on its line: "основная метрика" / "метрика оценки" push it up, "дополнительно"
    / "справочные" push it down; ties go to the earliest mention.
    """
    mentions = find_metric_mentions(text)
    if not mentions:
        return None, []
    low = text.lower()

    def line_of(pos):
        return low.count("\n", 0, pos)

    # "f1_score(..., average='macro')" yields both "f1" and "f1_macro" on one line:
    # the averaged variant is the precise one.
    averaged = {line_of(p) for n, p in mentions if n in ("f1_macro", "f1_weighted", "f1_micro")}
    mentions = [(n, p) for n, p in mentions if not (n == "f1" and line_of(p) in averaged)]
    best, best_score, seen = None, -1e9, []
    for name, pos in mentions:
        if name not in seen:
            seen.append(name)
        line_start = low.rfind("\n", 0, pos) + 1
        line_end = low.find("\n", pos)
        line = low[line_start: line_end if line_end != -1 else len(low)]
        before = low[line_start:pos]
        score = 0.0
        if any(h in line for h in _PRIMARY_HINTS):
            score += 3
        if "метрик" in line or "metric" in line:
            score += 2
        if any(h in before for h in _SECONDARY_HINTS):
            score -= 5
        elif any(h in line for h in _SECONDARY_HINTS):
            score -= 2
        if "**" in line:
            score += 1
        # Section heading context: mentions under a "метрика" heading get a boost.
        head = low.rfind("\n#", 0, pos)
        if head != -1 and ("метрик" in low[head: low.find("\n", head + 1)] or "metric" in low[head: low.find("\n", head + 1)]):
            score += 1
        score -= pos / max(len(low), 1) * 0.5  # mild preference for earlier mentions
        if score > best_score:
            best, best_score = name, score
    return best, seen


def normalized_score(metric_name: str, value: float, baseline: float) -> float:
    """Case formula: ``100*max(0, 1 - M/B)`` for errors, ``100*max(0, (M-B)/(1-B))`` for
    bounded higher-is-better metrics. Other metrics use a relative-improvement analogue."""
    m = REGISTRY[metric_name]
    if value is None or baseline is None or not np.isfinite(value) or not np.isfinite(baseline):
        return 0.0
    if not m.greater_is_better:
        if baseline <= 0:
            return 0.0
        return 100.0 * max(0.0, 1.0 - value / baseline)
    if m.bounded_one:
        if baseline >= 1:
            return 0.0
        return 100.0 * max(0.0, (value - baseline) / (1.0 - baseline))
    return 100.0 * max(0.0, (value - baseline) / max(abs(baseline), 1e-12))


# ----------------------------------------------------------------------- evaluation
def apply_decision(proba: np.ndarray, weights: np.ndarray | None = None,
                   threshold: float | None = None) -> np.ndarray:
    """Turn class probabilities into labels.

    ``threshold`` (binary only) labels a row positive when ``p1 >= threshold``;
    ``weights`` rescale class probabilities before the argmax.
    """
    proba = _proba_matrix(proba)
    if threshold is not None and proba.shape[1] == 2:
        return (proba[:, 1] >= threshold).astype(int)
    if weights is not None:
        proba = proba * np.asarray(weights)[None, :]
    return proba.argmax(axis=1)


def evaluate(metric_name: str, y_true: np.ndarray, raw: np.ndarray, decision: dict | None = None) -> float:
    """Score raw model output: values for regression, ``n x K`` probabilities otherwise."""
    m = REGISTRY[metric_name]
    if m.needs == "value":
        return m.fn(y_true, raw)
    if m.needs == "proba":
        return m.fn(y_true, _proba_matrix(raw))
    decision = decision or {}
    lab = apply_decision(raw, decision.get("weights"), decision.get("threshold"))
    return m.fn(y_true, lab)
