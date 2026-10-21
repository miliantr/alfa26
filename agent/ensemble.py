"""Ensembling of out-of-fold predictions (greedy forward selection with replacement).

Weights are fitted on OOF predictions over the shared folds. For label metrics
(macro F1 etc.) the greedy search optimises a smooth proxy — class-balanced log loss —
and the real metric is checked afterwards with its own decision rule.
"""
from __future__ import annotations

import numpy as np

from . import metrics, postprocess

_BALANCED = ("f1_macro", "balanced_accuracy", "recall_macro", "precision_macro", "mcc", "kappa", "qwk")


def proxy_loss(metric_name: str, task: str, y: np.ndarray, k: int):
    """Return f(pred) -> loss (lower is better) used inside the greedy search."""
    m = metrics.REGISTRY[metric_name]
    if task == metrics.REGRESSION or m.needs == "proba":
        sign = -1.0 if m.greater_is_better else 1.0
        return lambda p: sign * metrics.evaluate(metric_name, y, p)
    yi = y.astype(int)
    if metric_name in _BALANCED:
        cnt = np.bincount(yi, minlength=k).astype(float)
        w = (len(yi) / (k * np.maximum(cnt, 1)))[yi]
    else:
        w = np.ones(len(yi))
    w = w / w.sum()

    def loss(p):
        pp = np.clip(p[np.arange(len(yi)), yi], 1e-12, 1)
        return float(-(w * np.log(pp)).sum())
    return loss


def greedy_blend(preds: list[np.ndarray], loss, n_iter: int = 40, max_models: int | None = None):
    """Caruana-style selection with replacement; returns (weights, loss)."""
    n = len(preds)
    if n == 0:
        return np.array([]), float("inf")
    single = [loss(p) for p in preds]
    best_i = int(np.argmin(single))
    counts = np.zeros(n)
    counts[best_i] = 1
    current = preds[best_i].astype(np.float64).copy()
    best_loss = single[best_i]
    for step in range(1, n_iter):
        cand_losses = []
        for i in range(n):
            if max_models and counts[i] == 0 and (counts > 0).sum() >= max_models:
                cand_losses.append(np.inf)
                continue
            mix = (current * step + preds[i]) / (step + 1)
            cand_losses.append(loss(mix))
        j = int(np.argmin(cand_losses))
        if cand_losses[j] >= best_loss - 1e-12:
            break
        counts[j] += 1
        current = (current * step + preds[j]) / (step + 1)
        best_loss = cand_losses[j]
    return counts / counts.sum(), best_loss


def blend(preds: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    out = np.zeros_like(preds[0], dtype=np.float64)
    for p, w in zip(preds, weights):
        if w > 0:
            out += w * p
    return out


def _plain_logloss(y: np.ndarray):
    yi = y.astype(int)

    def loss(p):
        return float(-np.log(np.clip(p[np.arange(len(yi)), yi], 1e-12, 1)).mean())
    return loss


def build(metric_name: str, task: str, y: np.ndarray, oofs: list[np.ndarray], k: int,
          y_range: tuple | None = None, groups: np.ndarray | None = None) -> dict:
    """Fit blend weights + decision on OOF; returns weights, decision and the (cross-fitted,
    when ``groups`` are given) score.

    For label metrics two proxies are tried and the blend with the better real score is
    kept: the class-balanced log loss rewards a class-weighted model merely for already
    leaning towards rare classes (an advantage the decision rule removes), so on its own it
    can lock onto that one model; plain log loss mixes calibrated models instead.
    """
    proxies = [proxy_loss(metric_name, task, y, k)]
    m = metrics.REGISTRY[metric_name]
    if task != metrics.REGRESSION and m.needs == "label" and metric_name in _BALANCED:
        proxies.append(_plain_logloss(y))
    best = None
    for loss in proxies:
        weights, _ = greedy_blend(oofs, loss)
        if best is not None and np.allclose(weights, best["weights"]):
            continue
        oof = blend(oofs, weights)
        if groups is not None:
            score, _, decision = postprocess.crossfit(metric_name, task, y, oof, groups, y_range=y_range)
        else:
            decision = postprocess.fit(metric_name, task, y, oof, y_range=y_range)
            score = postprocess.score(metric_name, y, oof, decision)
        if best is None or m.better(score, best["score"]):
            best = {"weights": weights, "decision": decision, "score": score, "oof": oof}
    best["weights"] = np.asarray(best["weights"]).tolist()
    return best
