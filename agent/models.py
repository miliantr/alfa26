"""Model families behind one fold-level interface.

``fit_predict(family, params, task, ...)`` trains on one fold with early stopping on the
validation part and returns predictions for validation and test rows:
regression -> ``(n,)`` values, classification -> ``(n, K)`` probabilities in class-index
order (classes absent from a training fold get probability 0).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import log
from .features import cat_columns, num_columns

L = log.get("models")

REG, BIN, MUL = "regression", "binary", "multiclass"

DEFAULTS = {
    "lgbm": dict(learning_rate=0.05, num_leaves=63, min_child_samples=20, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, cat_smooth=10,
                 min_data_per_group=50, max_cat_to_onehot=8, n_estimators=5000, early_stopping=200),
    "lgbm_fast": dict(learning_rate=0.1, num_leaves=31, min_child_samples=20, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, n_estimators=600, early_stopping=50),
    "catboost": dict(learning_rate=0.1, depth=6, l2_leaf_reg=3.0, n_estimators=3000, early_stopping=150,
                     border_count=128, max_ctr_complexity=2),
    "xgb": dict(learning_rate=0.05, max_depth=7, min_child_weight=1.0, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, max_bin=256, n_estimators=5000, early_stopping=200, max_cat_to_onehot=8),
    "linear": dict(C=1.0, alpha=1.0),
    "extratrees": dict(n_estimators=300, min_samples_leaf=2, max_features=0.5),
    "randomforest": dict(n_estimators=300, min_samples_leaf=1, max_features=0.33),
}

FAMILIES = tuple(k for k in DEFAULTS if k != "lgbm_fast")


def class_weights_vector(y: np.ndarray, k: int, mode: str | None) -> np.ndarray | None:
    if not mode or mode == "none":
        return None
    counts = np.bincount(y.astype(int), minlength=k).astype(float)
    counts[counts == 0] = 1.0
    w = len(y) / (k * counts)
    if mode == "sqrt":
        w = np.sqrt(w)
    return w


def _sample_weight(task, y, k, mode):
    if task == REG:
        return None
    w = class_weights_vector(y, k, mode)
    return None if w is None else w[y.astype(int)]


def _full_proba(p: np.ndarray, present: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((p.shape[0], k), dtype=np.float64)
    out[:, present.astype(int)] = p
    return out


# ----------------------------------------------------------------------- LightGBM
def _fit_lgbm(params, task, Xtr, ytr, Xva, yva, Xte, k, n_jobs, seed, time_limit, objective, cw):
    import lightgbm as lgb

    p = {**DEFAULTS["lgbm"], **params}
    n_est = int(p.pop("n_estimators"))
    es = int(p.pop("early_stopping"))
    lp = {k_: v for k_, v in p.items()}
    lp.update(n_jobs=n_jobs, verbose=-1, seed=seed, deterministic=False, force_col_wise=True)
    if task == REG:
        lp["objective"] = {"l1": "l1", "mape": "mape", "huber": "huber", "poisson": "poisson"}.get(objective, "regression")
    elif task == BIN:
        lp["objective"] = "binary"
    else:
        lp.update(objective="multiclass", num_class=k)
    sw = _sample_weight(task, ytr, k, cw)
    dtr = lgb.Dataset(Xtr, ytr, weight=sw, free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, weight=_sample_weight(task, yva, k, cw), reference=dtr)
    t0 = time.time()

    def _time_cb(env):
        if time_limit and time.time() - t0 > time_limit:
            raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)
    _time_cb.order = 40

    booster = lgb.train(lp, dtr, num_boost_round=n_est, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(es, verbose=False), _time_cb])
    it = booster.best_iteration or booster.current_iteration()
    pva = booster.predict(Xva, num_iteration=it)
    pte = booster.predict(Xte, num_iteration=it)
    if task == BIN:
        pva, pte = np.column_stack([1 - pva, pva]), np.column_stack([1 - pte, pte])
    imp = dict(zip(booster.feature_name(), booster.feature_importance("gain")))
    return pva, pte, {"best_iteration": int(it), "importance": imp}


# ----------------------------------------------------------------------- CatBoost
def _cb_frame(X: pd.DataFrame, cats: list[str]) -> pd.DataFrame:
    X = X.copy()
    for c in cats:
        X[c] = X[c].astype(str)
    return X


def _fit_catboost(params, task, Xtr, ytr, Xva, yva, Xte, k, n_jobs, seed, time_limit, objective, cw):
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    p = {**DEFAULTS["catboost"], **params}
    n_est = int(p.pop("n_estimators"))
    es = int(p.pop("early_stopping"))
    cats = cat_columns(Xtr)
    A, B, C = _cb_frame(Xtr, cats), _cb_frame(Xva, cats), _cb_frame(Xte, cats)
    common = dict(iterations=n_est, od_type="Iter", od_wait=es, thread_count=n_jobs, random_seed=seed,
                  allow_writing_files=False, verbose=False, use_best_model=True, **p)

    class _TimeCb:
        def __init__(self):
            self.t0 = time.time()

        def after_iteration(self, info):
            return not (time_limit and time.time() - self.t0 > time_limit)

    if task == REG:
        loss = {"l1": "MAE", "mape": "MAPE", "huber": "Huber:delta=1.0", "poisson": "Poisson"}.get(objective, "RMSE")
        model = CatBoostRegressor(loss_function=loss, **common)
    else:
        loss = "Logloss" if task == BIN else "MultiClass"
        acw = {"balanced": "Balanced", "sqrt": "SqrtBalanced"}.get(cw or "", None)
        extra = {"auto_class_weights": acw} if acw else {}
        model = CatBoostClassifier(loss_function=loss, class_names=list(range(k)), **extra, **common)
    model.fit(Pool(A, ytr, cat_features=cats), eval_set=Pool(B, yva, cat_features=cats), callbacks=[_TimeCb()])
    if task == REG:
        pva, pte = model.predict(B), model.predict(C)
    else:
        pva, pte = model.predict_proba(B), model.predict_proba(C)
    imp = dict(zip(Xtr.columns, model.get_feature_importance()))
    return pva, pte, {"best_iteration": int(model.get_best_iteration() or 0), "importance": imp}


# ------------------------------------------------------------------------ XGBoost
def _fit_xgb(params, task, Xtr, ytr, Xva, yva, Xte, k, n_jobs, seed, time_limit, objective, cw):
    import xgboost as xgb

    p = {**DEFAULTS["xgb"], **params}
    n_est = int(p.pop("n_estimators"))
    es = int(p.pop("early_stopping"))
    xp = {"eta": p.pop("learning_rate"), "tree_method": "hist", "nthread": n_jobs, "seed": seed,
          "verbosity": 0, **p}
    if task == REG:
        xp["objective"] = {"l1": "reg:absoluteerror", "huber": "reg:pseudohubererror"}.get(objective, "reg:squarederror")
        if objective == "mape":
            xp["objective"] = "reg:absoluteerror"
    elif task == BIN:
        xp.update(objective="binary:logistic", eval_metric="logloss")
    else:
        xp.update(objective="multi:softprob", num_class=k, eval_metric="mlogloss")
    sw = _sample_weight(task, ytr, k, cw)
    if task == REG and objective == "mape":
        sw = 1.0 / np.maximum(np.abs(ytr), 1e-6)
    dtr = xgb.DMatrix(Xtr, ytr, weight=sw, enable_categorical=True)
    dva = xgb.DMatrix(Xva, yva, enable_categorical=True)
    dte = xgb.DMatrix(Xte, enable_categorical=True)

    class _TimeCb(xgb.callback.TrainingCallback):
        def __init__(self):
            self.t0 = time.time()

        def after_iteration(self, model, epoch, evals_log):
            return bool(time_limit and time.time() - self.t0 > time_limit)

    bst = xgb.train(xp, dtr, num_boost_round=n_est, evals=[(dva, "val")], early_stopping_rounds=es,
                    verbose_eval=False, callbacks=[_TimeCb()])
    best = getattr(bst, "best_iteration", None)
    rng = (0, int(best) + 1) if best is not None else (0, 0)
    pva = bst.predict(dva, iteration_range=rng)
    pte = bst.predict(dte, iteration_range=rng)
    if task == BIN:
        pva, pte = np.column_stack([1 - pva, pva]), np.column_stack([1 - pte, pte])
    imp = bst.get_score(importance_type="total_gain")
    return pva, pte, {"best_iteration": int(best or 0), "importance": imp}


# ------------------------------------------------------------------ sklearn models
def _encode_dense(Xtr, Xva, Xte):
    """Ordinal codes for categoricals, NaN -> -999 (tree ensembles)."""
    out = []
    for X in (Xtr, Xva, Xte):
        X = X.copy()
        for c in cat_columns(X):
            X[c] = X[c].cat.codes.astype("float32")
        out.append(X.astype("float32").fillna(-999.0).to_numpy())
    return out


def _linear_design(Xtr, Xva, Xte):
    from scipy import sparse
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

    nums, cats = num_columns(Xtr), cat_columns(Xtr)
    mats = [[], [], []]
    if nums:
        imp = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
        qt = QuantileTransformer(n_quantiles=min(200, max(10, len(Xtr) // 10)), output_distribution="normal",
                                 subsample=100000, random_state=0)
        A = imp.fit_transform(Xtr[nums].astype("float64"))
        ncols = len(nums)
        A_num = qt.fit_transform(A[:, :ncols])
        mats[0].append(sparse.csr_matrix(np.hstack([A_num, A[:, ncols:]])))
        for i, X in ((1, Xva), (2, Xte)):
            M = imp.transform(X[nums].astype("float64"))
            mats[i].append(sparse.csr_matrix(np.hstack([qt.transform(M[:, :ncols]), M[:, ncols:]])))
    if cats:
        ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, max_categories=60)
        mats[0].append(ohe.fit_transform(Xtr[cats].astype(str)))
        mats[1].append(ohe.transform(Xva[cats].astype(str)))
        mats[2].append(ohe.transform(Xte[cats].astype(str)))
    return [sparse.hstack(m).tocsr() if m else sparse.csr_matrix((len(X), 0)) for m, X in zip(mats, (Xtr, Xva, Xte))]


def _fit_linear(params, task, Xtr, ytr, Xva, yva, Xte, k, n_jobs, seed, time_limit, objective, cw):
    from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge

    p = {**DEFAULTS["linear"], **params}
    A, B, C = _linear_design(Xtr, Xva, Xte)
    if task == REG:
        model = Ridge(alpha=p["alpha"], random_state=seed)
        model.fit(A, ytr)
        return model.predict(B), model.predict(C), {}
    present = np.unique(ytr)
    model = LogisticRegression(C=p["C"], max_iter=400, class_weight="balanced" if cw else None,
                               random_state=seed)
    model.fit(A, ytr)
    return (_full_proba(model.predict_proba(B), model.classes_, k),
            _full_proba(model.predict_proba(C), model.classes_, k), {})


def _fit_trees(kind, params, task, Xtr, ytr, Xva, yva, Xte, k, n_jobs, seed, time_limit, objective, cw):
    from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier,
                                  RandomForestRegressor)

    p = {**DEFAULTS[kind], **params}
    A, B, C = _encode_dense(Xtr, Xva, Xte)
    common = dict(n_estimators=int(p["n_estimators"]), min_samples_leaf=int(p["min_samples_leaf"]),
                  max_features=p["max_features"], n_jobs=n_jobs, random_state=seed)
    if task == REG:
        cls = ExtraTreesRegressor if kind == "extratrees" else RandomForestRegressor
        model = cls(**common)
        model.fit(A, ytr)
        return model.predict(B), model.predict(C), {}
    cls = ExtraTreesClassifier if kind == "extratrees" else RandomForestClassifier
    model = cls(class_weight="balanced_subsample" if cw else None, **common)
    model.fit(A, ytr)
    imp = dict(zip(Xtr.columns, model.feature_importances_))
    return (_full_proba(model.predict_proba(B), model.classes_, k),
            _full_proba(model.predict_proba(C), model.classes_, k), {"importance": imp})


def fit_predict(family: str, params: dict, task: str, Xtr, ytr, Xva, yva, Xte, n_classes: int,
                n_jobs: int, seed: int, time_limit: float | None, objective: str = "l2",
                class_weight: str | None = None):
    fam = "lgbm" if family == "lgbm_fast" else family
    if family == "lgbm_fast":
        params = {**DEFAULTS["lgbm_fast"], **params}
    if fam == "lgbm":
        return _fit_lgbm(params, task, Xtr, ytr, Xva, yva, Xte, n_classes, n_jobs, seed, time_limit, objective, class_weight)
    if fam == "catboost":
        return _fit_catboost(params, task, Xtr, ytr, Xva, yva, Xte, n_classes, n_jobs, seed, time_limit, objective, class_weight)
    if fam == "xgb":
        return _fit_xgb(params, task, Xtr, ytr, Xva, yva, Xte, n_classes, n_jobs, seed, time_limit, objective, class_weight)
    if fam == "linear":
        return _fit_linear(params, task, Xtr, ytr, Xva, yva, Xte, n_classes, n_jobs, seed, time_limit, objective, class_weight)
    if fam in ("extratrees", "randomforest"):
        return _fit_trees(fam, params, task, Xtr, ytr, Xva, yva, Xte, n_classes, n_jobs, seed, time_limit, objective, class_weight)
    raise ValueError(f"unknown model family {family}")
