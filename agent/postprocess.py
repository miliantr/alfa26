"""Metric-specific post-processing fitted on out-of-fold predictions only.

``fit(metric, task, y, raw)`` returns a JSON-able decision; ``apply(decision, raw)``
transforms raw output (values or class probabilities) into what gets scored/written:
regression -> clipped values, label metrics -> one-hot of the chosen class,
proba metrics -> (optionally temperature-scaled) probabilities.
"""
from __future__ import annotations

import numpy as np

from . import metrics

_FAST_LABEL = ("f1_macro", "f1_weighted", "f1_micro", "accuracy", "balanced_accuracy",
               "precision_macro", "recall_macro", "f1")


def _confusion(y: np.ndarray, lab: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(y.astype(int) * k + lab.astype(int), minlength=k * k).reshape(k, k)


def fast_label_score(name: str, y: np.ndarray, lab: np.ndarray, k: int) -> float:
    if name not in _FAST_LABEL:
        return metrics.REGISTRY[name].fn(y, lab)
    cm = _confusion(y, lab, k).astype(float)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    if name == "accuracy" or name == "f1_micro":
        return float(tp.sum() / max(cm.sum(), 1))
    prec = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    rec = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    if name == "f1":
        return float(f1[1]) if k > 1 else 0.0
    # sklearn's macro averages over labels present in y_true or y_pred
    present = (support > 0) | (predicted > 0)
    if name == "f1_macro":
        return float(f1[present].mean()) if present.any() else 0.0
    if name == "precision_macro":
        return float(prec[present].mean()) if present.any() else 0.0
    if name in ("recall_macro", "balanced_accuracy"):
        s = support > 0
        return float(rec[s].mean()) if s.any() else 0.0
    if name == "f1_weighted":
        return float((f1 * support).sum() / max(support.sum(), 1))
    raise KeyError(name)


def _onehot(lab: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((len(lab), k))
    out[np.arange(len(lab)), lab.astype(int)] = 1.0
    return out


def _proba(raw: np.ndarray) -> np.ndarray:
    p = np.asarray(raw, float)
    if p.ndim == 1:
        p = np.column_stack([1 - p, p])
    return p


# ------------------------------------------------------------------------- fitting
def fit(metric_name: str, task: str, y: np.ndarray, raw: np.ndarray, y_range: tuple | None = None,
        fast: bool = False) -> dict:
    m = metrics.REGISTRY[metric_name]
    if task == metrics.REGRESSION:
        return _fit_regression(metric_name, y, raw, y_range)
    P = _proba(raw)
    k = P.shape[1]
    if m.needs == "proba":
        if metric_name in ("logloss", "brier"):
            return _fit_temperature(metric_name, y, P)
        return {"kind": "proba"}
    if metric_name == "qwk" or metric_name == "kappa" and k > 2:
        return _fit_ordinal(metric_name, y, P)
    if k == 2:
        return _fit_threshold(metric_name, y, P)
    return _fit_class_weights(metric_name, y, P, rounds=2 if fast else 3)


def _score_labels(metric_name, y, lab, k):
    return fast_label_score(metric_name, y, lab, k)


def _fit_threshold(metric_name, y, P):
    p1 = P[:, 1]
    base = _score_labels(metric_name, y, (p1 >= 0.5).astype(int), 2)
    best_t, best = 0.5, base
    qs = np.unique(np.quantile(p1, np.linspace(0.01, 0.99, 197)))
    for t in qs:
        s = _score_labels(metric_name, y, (p1 >= t).astype(int), 2)
        if s > best + 1e-12:
            best, best_t = s, float(t)
    return {"kind": "threshold", "threshold": best_t, "gain": best - base}


def _fit_class_weights(metric_name, y, P, rounds=3):
    k = P.shape[1]
    w = np.ones(k)
    logP = np.log(np.clip(P, 1e-12, 1))
    base = best = _score_labels(metric_name, y, P.argmax(1), k)
    factors = np.array([0.4, 0.55, 0.7, 0.85, 1.2, 1.45, 1.8, 2.5, 3.5])
    for _ in range(rounds):
        improved = False
        for c in range(k):
            for f in factors:
                w2 = w.copy()
                w2[c] *= f
                lab = (logP + np.log(w2)[None, :]).argmax(1)
                s = _score_labels(metric_name, y, lab, k)
                if s > best + 1e-9:
                    best, w, improved = s, w2, True
        if not improved:
            break
    w = w / np.exp(np.mean(np.log(w)))
    return {"kind": "class_weights", "weights": w.tolist(), "gain": best - base}


def _fit_ordinal(metric_name, y, P):
    k = P.shape[1]
    e = P @ np.arange(k)
    cuts = np.arange(k - 1) + 0.5

    def score(c):
        return metrics.REGISTRY[metric_name].fn(y, np.searchsorted(np.sort(c), e))

    base = best = score(cuts)
    for _ in range(3):
        for i in range(k - 1):
            for d in np.linspace(-0.5, 0.5, 21):
                c2 = cuts.copy()
                c2[i] += d
                s = score(c2)
                if s > best + 1e-9:
                    best, cuts = s, c2
    return {"kind": "ordinal", "cuts": np.sort(cuts).tolist(), "gain": best - base}


def _fit_temperature(metric_name, y, P):
    logP = np.log(np.clip(P, 1e-12, 1))
    f = metrics.REGISTRY[metric_name].fn
    base = f(y, P)
    best_t, best = 1.0, base
    for t in np.concatenate([np.linspace(0.5, 0.95, 10), np.linspace(1.05, 2.5, 15)]):
        Z = logP / t
        Q = np.exp(Z - Z.max(1, keepdims=True))
        Q /= Q.sum(1, keepdims=True)
        s = f(y, Q)
        if s < best - 1e-9:
            best, best_t = s, float(t)
    return {"kind": "temperature", "temperature": best_t, "gain": base - best}


def _fit_regression(metric_name, y, raw, y_range):
    raw = np.asarray(raw, float)
    lo, hi = (float(np.min(y)), float(np.max(y))) if y_range is None else y_range
    f = metrics.REGISTRY[metric_name]
    base = f.fn(y, raw)
    clipped = f.fn(y, np.clip(raw, lo, hi))
    if not f.better(base, clipped):  # clipping is neutral or better: keep it (safe)
        return {"kind": "clip", "lo": lo, "hi": hi, "gain": abs(base - clipped)}
    return {"kind": "none"}


# ------------------------------------------------------------------------ applying
def apply(decision: dict | None, raw: np.ndarray) -> np.ndarray:
    d = decision or {}
    kind = d.get("kind", "none")
    if kind == "clip":
        return np.clip(np.asarray(raw, float), d["lo"], d["hi"])
    if kind in ("none",) and np.asarray(raw).ndim == 1:
        return np.asarray(raw, float)
    P = _proba(raw)
    k = P.shape[1]
    if kind == "threshold":
        return _onehot((P[:, 1] >= d["threshold"]).astype(int), 2)
    if kind == "class_weights":
        logP = np.log(np.clip(P, 1e-12, 1))
        return _onehot((logP + np.log(np.asarray(d["weights"]))[None, :]).argmax(1), k)
    if kind == "ordinal":
        e = P @ np.arange(k)
        return _onehot(np.clip(np.searchsorted(np.asarray(d["cuts"]), e), 0, k - 1), k)
    if kind == "temperature":
        Z = np.log(np.clip(P, 1e-12, 1)) / d["temperature"]
        Q = np.exp(Z - Z.max(1, keepdims=True))
        return Q / Q.sum(1, keepdims=True)
    if kind == "label":
        return _onehot(P.argmax(1), k)
    return P


def labels_from(decision: dict | None, raw: np.ndarray) -> np.ndarray:
    """Class indices after the decision rule (classification only)."""
    return apply(decision or {"kind": "label"}, raw).argmax(1)


def score(metric_name: str, y: np.ndarray, raw: np.ndarray, decision: dict | None) -> float:
    return metrics.evaluate(metric_name, y, apply(decision, raw))
