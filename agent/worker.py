"""Experiment worker: one cross-validated model run in an isolated subprocess.

    python -m agent.worker <job.json>

Reads the shared base data, builds the requested feature set (cached), trains one model
per fold with early stopping, and writes ``oof.npy``, ``test.npy`` and ``result.json``
into the job directory. The parent kills it on timeout or memory overrun.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_columns(df_list):
    cols = list(df_list[0].columns)
    mapping, used = {}, set()
    for i, c in enumerate(cols):
        s = re.sub(r"[^0-9A-Za-z_]+", "_", str(c)).strip("_")[:60] or "f"
        if s in used or s[0].isdigit():
            s = f"f{i}_{s}"
        used.add(s)
        mapping[c] = s
    return [d.rename(columns=mapping) for d in df_list], mapping


def _target_encode(Xtr, ytr, others, cols, task, k, seed, n_inner=5, smooth=20.0):
    """Out-of-fold mean-target encoding fitted on the training fold only."""
    from sklearn.model_selection import KFold

    Xtr = Xtr.copy()
    others = [o.copy() for o in others]
    if task == "multiclass":
        Y = np.eye(k)[ytr.astype(int)]
    else:
        Y = ytr.reshape(-1, 1).astype(float)
    prior = Y.mean(0)
    for c in cols:
        if c not in Xtr.columns:
            continue
        key = Xtr[c].astype(str).to_numpy()
        names = [f"{c}__te{j}" if Y.shape[1] > 1 else f"{c}__te" for j in range(Y.shape[1])]
        enc_tr = np.zeros((len(Xtr), Y.shape[1]))
        for tr, va in KFold(n_inner, shuffle=True, random_state=seed).split(key):
            df = pd.DataFrame(Y[tr]).assign(key=key[tr]).groupby("key")
            s, n = df.sum(), df.size()
            m = (s.values + smooth * prior) / (n.values[:, None] + smooth)
            lut = pd.DataFrame(m, index=s.index)
            enc_tr[va] = lut.reindex(key[va]).fillna(pd.Series(prior)).to_numpy()
        df = pd.DataFrame(Y).assign(key=key).groupby("key")
        s, n = df.sum(), df.size()
        lut = pd.DataFrame((s.values + smooth * prior) / (n.values[:, None] + smooth), index=s.index)
        for j, nm in enumerate(names):
            Xtr[nm] = enc_tr[:, j].astype("float32")
        for o in others:
            ok = o[c].astype(str).to_numpy()
            enc = lut.reindex(ok).to_numpy()
            enc = np.where(np.isnan(enc), prior[None, :], enc)
            for j, nm in enumerate(names):
                o[nm] = enc[:, j].astype("float32")
    return Xtr, others


def _transform_target(y, mode, shift):
    if mode == "log1p":
        return np.log1p(y - shift)
    if mode == "sqrt":
        return np.sqrt(y - shift)
    return y


def _inverse_target(p, mode, shift):
    if mode == "log1p":
        return np.expm1(p) + shift
    if mode == "sqrt":
        return np.square(np.clip(p, 0, None)) + shift
    return p


def run(job: dict) -> dict:
    from agent import features, metrics, models, postprocess

    t_start = time.time()
    out_dir = Path(job["out_dir"])
    base = pd.read_pickle(job["base_path"])
    Xtr0, Xte0, y, folds, meta = base["Xtr"], base["Xte"], base["y"], base["folds"], base["meta"]
    task, k, metric = meta["task_type"], int(meta.get("n_classes", 0)), job["metric"]
    ctx = {"importance": job.get("importance", []), "raw_text": meta.get("raw_text", {})}
    Xtr, Xte, fs_info = features.build_features(Xtr0, Xte0, job.get("feature_ops", []), ctx,
                                                cache_dir=Path(job["fs_cache_dir"]))
    if job.get("drop_cols"):
        dc = [c for c in job["drop_cols"] if c in Xtr.columns]
        Xtr, Xte = Xtr.drop(columns=dc), Xte.drop(columns=dc)
    (Xtr, Xte), mapping = _safe_columns([Xtr, Xte])
    inv_map = {v: k_ for k_, v in mapping.items()}
    model = job["model"]
    family, params = model["family"], model.get("params", {})
    tt = model.get("target_transform", "none") if task == "regression" else "none"
    # Shift so the transformed target is defined: y - shift >= 0.
    shift = min(0.0, float(np.min(y))) if tt in ("log1p", "sqrt") else 0.0
    te_cols = [mapping.get(c, c) for c in model.get("te_cols", []) if mapping.get(c, c) in Xtr.columns]
    if model.get("te_cols") == "auto":
        te_cols = [mapping[c] for c in meta.get("high_card_cats", []) if c in mapping]

    n_jobs = int(job.get("n_jobs", 8))
    seed = int(job.get("seed", 42))
    time_limit = float(job.get("time_limit", 1e9))
    fold_ids = sorted(int(f) for f in np.unique(folds) if f >= 0)
    n_out = k if task != "regression" else 0
    oof = np.full((len(Xtr), n_out) if n_out else len(Xtr), np.nan, dtype=np.float64)
    test_pred = np.zeros((len(Xte), n_out) if n_out else len(Xte), dtype=np.float64)
    fold_info, imp_total = [], {}
    y_fit = _transform_target(y.astype(float), tt, shift) if task == "regression" else y.astype(int)
    for i, f in enumerate(fold_ids):
        elapsed = time.time() - t_start
        left = time_limit - elapsed
        per_fold = left / max(1, len(fold_ids) - i)
        if i > 0 and fold_info and fold_info[-1]["seconds"] > left * 1.05:
            raise TimeoutError(f"not enough time for fold {i}: {left:.0f}s left")
        tr = np.where(folds != f)[0]
        va = np.where(folds == f)[0]
        A, B, C = Xtr.iloc[tr], Xtr.iloc[va], Xte
        if te_cols:
            A, (B, C) = _target_encode(A, y_fit[tr], [B, C], te_cols, task, k, seed)
        t0 = time.time()
        pva, pte, info = models.fit_predict(family, params, task, A, y_fit[tr], B, y_fit[va], C,
                                            n_classes=k, n_jobs=n_jobs, seed=seed + f,
                                            time_limit=max(5.0, per_fold * 0.95),
                                            objective=model.get("objective", "l2"),
                                            class_weight=model.get("class_weight"))
        if task == "regression":
            pva, pte = _inverse_target(pva, tt, shift), _inverse_target(pte, tt, shift)
        oof[va] = pva
        test_pred += pte / len(fold_ids)
        for name, v in (info.get("importance") or {}).items():
            imp_total[name] = imp_total.get(name, 0.0) + float(v)
        fold_info.append({"fold": f, "seconds": round(time.time() - t0, 2),
                          "best_iteration": info.get("best_iteration")})
    if task == "regression" and not np.all(np.isfinite(oof)):
        raise ValueError("non-finite OOF predictions")
    # Score with a decision rule fitted on OOF (same rule for every fold), on the rows
    # whose validation conditions mimic the test (eval mask, see profiler.eval_mask).
    mask = base.get("eval_mask")
    mask = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, bool)
    y_eval = y.astype(float) if task == "regression" else y.astype(int)
    y_range = tuple(meta.get("y_range")) if meta.get("y_range") else None
    decision = postprocess.fit(metric, task, y_eval[mask], oof[mask], y_range=y_range)
    score = postprocess.score(metric, y_eval[mask], oof[mask], decision)
    if task == "regression":
        raw_decision = None
    elif metrics.REGISTRY[metric].needs == "label":
        raw_decision = {"kind": "label"}
    else:
        raw_decision = {"kind": "proba"}
    raw_score = postprocess.score(metric, y_eval[mask], oof[mask], raw_decision)
    fold_scores = []
    for f in fold_ids:
        idx = np.where((folds == f) & mask)[0]
        try:
            fold_scores.append(postprocess.score(metric, y_eval[idx], oof[idx], decision))
        except Exception:
            fold_scores.append(float("nan"))
    np.save(out_dir / "oof.npy", oof.astype(np.float32))
    np.save(out_dir / "test.npy", test_pred.astype(np.float32))
    imp_sorted = sorted(imp_total.items(), key=lambda kv: -kv[1])
    tot = sum(v for _, v in imp_sorted) or 1.0
    return {
        "status": "ok", "score": score, "raw_score": raw_score, "fold_scores": fold_scores,
        "decision": decision, "folds": fold_info, "fs_id": fs_info.get("fs_id"),
        "fs_errors": fs_info.get("errors", []), "n_features": Xtr.shape[1],
        "importance": [[inv_map.get(n, n), round(v / tot, 5)] for n, v in imp_sorted[:60]],
        "zero_importance": [inv_map.get(n, n) for n, v in imp_total.items() if v <= 0][:200],
        "seconds": round(time.time() - t_start, 2),
    }


def main() -> None:
    job_path = Path(sys.argv[1])
    job = json.loads(job_path.read_text(encoding="utf-8"))
    out_dir = Path(job["out_dir"])
    try:
        os.environ.setdefault("OMP_NUM_THREADS", str(job.get("n_jobs", 8)))
        res = run(job)
    except BaseException as e:  # noqa: BLE001
        res = {"status": "error", "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-4000:]}
    from agent.io_utils import write_json

    write_json(out_dir / "result.json", res)


if __name__ == "__main__":
    main()
