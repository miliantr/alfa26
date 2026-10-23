"""Data profiling: column typing, id/leak detection, train/test shift, validation choice.

Produces a :class:`DataProfile` that the pipeline uses to build features and folds and
that the LLM sees (compact form) when proposing hypotheses.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

import agent_log as log
import agent_metrics as metrics

L = log.get("profile")

ID_NAMES = re.compile(r"^(id|row_?id|index|uid|uuid|guid|key|record_?id|sample_?id|object_?id)$|(^|_)id$", re.I)


@dataclass
class ColumnInfo:
    name: str
    kind: str  # numeric | categorical | datetime | text | bool | id | constant | duplicate
    dtype: str
    n_unique: int
    missing: float
    high_card: bool = False
    note: str = ""


@dataclass
class DataProfile:
    n_train: int
    n_test: int
    columns: dict = field(default_factory=dict)  # name -> ColumnInfo (as dict)
    num_features: list = field(default_factory=list)
    cat_features: list = field(default_factory=list)
    datetime_features: list = field(default_factory=list)
    text_features: list = field(default_factory=list)
    dropped: dict = field(default_factory=dict)  # name -> reason
    time_column: str | None = None
    time_kind: str | None = None  # "ordinal" | "datetime"
    validation: dict = field(default_factory=dict)
    target: dict = field(default_factory=dict)
    adversarial: dict = field(default_factory=dict)
    leakage: list = field(default_factory=list)
    duplicate_rows: float = 0.0
    unseen_categories: dict = field(default_factory=dict)
    entity: dict = field(default_factory=dict)  # recurring-entity key (find_entity_key)
    seconds: float = 0.0

    @property
    def features(self) -> list[str]:
        return self.num_features + self.cat_features + self.datetime_features + self.text_features

    def to_dict(self) -> dict:
        d = asdict(self)
        d["features"] = self.features
        return d

    def compact(self, max_cols: int = 40) -> dict:
        """Small summary for LLM prompts."""
        cols = []
        for name in self.features[:max_cols]:
            c = self.columns[name]
            cols.append(f"{name}:{c['kind']}{'(hi-card)' if c['high_card'] else ''}"
                        f" nun={c['n_unique']} na={c['missing']:.2f}")
        return {
            "rows": {"train": self.n_train, "test": self.n_test},
            "n_features": len(self.features),
            "columns": cols + ([f"... {len(self.features) - max_cols} more"] if len(self.features) > max_cols else []),
            "dropped": self.dropped,
            "time_column": self.time_column,
            "validation": self.validation.get("scheme"),
            "target": self.target,
            "adversarial_auc": self.adversarial.get("auc"),
            "shift_top_features": self.adversarial.get("top_features", [])[:5],
            "duplicate_rows_frac": round(self.duplicate_rows, 3),
            "leakage_suspects": self.leakage,
            "entity_key": self.entity or None,
        }


# ------------------------------------------------------------------------- helpers
def _looks_datetime(s: pd.Series) -> bool:
    sample = s.dropna().astype(str).head(500)
    if sample.empty:
        return False
    if not sample.str.contains(r"\d{1,4}[-/.:]\d{1,2}", regex=True).mean() > 0.9:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed") if hasattr(pd, "to_datetime") else None
    return parsed is not None and parsed.notna().mean() > 0.9


def _looks_text(s: pd.Series) -> bool:
    sample = s.dropna().astype(str).head(1000)
    if sample.empty:
        return False
    return sample.str.len().mean() > 30 and sample.str.count(" ").mean() >= 3


def _dict_dtype(dictionary: list[dict], col: str) -> str:
    for r in dictionary:
        if str(r.get("feature")) == col:
            return str(r.get("dtype") or "").lower()
    return ""


def _dict_role(dictionary: list[dict], col: str) -> str:
    for r in dictionary:
        if str(r.get("feature")) == col:
            return str(r.get("role") or "").lower()
    return ""


def _is_monotonic_id(s: pd.Series) -> bool:
    v = pd.to_numeric(s, errors="coerce")
    if v.isna().any() or not v.is_unique:
        return False
    if not np.allclose(v % 1, 0):
        return False
    if v.is_monotonic_increasing or v.is_monotonic_decreasing:
        return True
    r = np.corrcoef(np.arange(len(v)), v.to_numpy(float))[0, 1] if len(v) > 2 else 0
    return abs(r) > 0.99


# ------------------------------------------------------------------------- main API
def profile(train: pd.DataFrame, test: pd.DataFrame, target_cols: list[str], id_col: str | None,
            task_type: str, dictionary: list[dict] | None = None, y: np.ndarray | None = None,
            seed: int = 42, n_jobs: int = 8, time_hint: str | None = None,
            adversarial: bool = True) -> DataProfile:
    t0 = time.time()
    dictionary = dictionary or []
    prof = DataProfile(n_train=len(train), n_test=len(test))
    candidates = [c for c in train.columns if c not in target_cols and c in test.columns]
    seen_hashes: dict = {}
    n = len(train)
    for c in candidates:
        s_tr, s_te = train[c], test[c]
        both = pd.concat([s_tr, s_te], ignore_index=True)
        nun = int(both.nunique(dropna=True))
        info = ColumnInfo(name=c, kind="numeric", dtype=str(s_tr.dtype), n_unique=nun,
                          missing=float(s_tr.isna().mean()))
        role = _dict_role(dictionary, c)
        ddt = _dict_dtype(dictionary, c)
        if c == id_col or role == "id":
            info.kind, info.note = "id", "id column"
        elif nun <= 1:
            info.kind, info.note = "constant", "single value"
        elif pd.api.types.is_bool_dtype(s_tr):
            info.kind = "bool"
        elif pd.api.types.is_numeric_dtype(s_tr) and pd.api.types.is_numeric_dtype(s_te):
            if ddt.startswith(("cat", "категор")) and nun <= max(1000, 0.2 * n):
                info.kind = "categorical"
            elif ID_NAMES.search(str(c)) and s_tr.is_unique and _is_monotonic_id(s_tr):
                info.kind, info.note = "id", "id-like monotonic unique integer"
            else:
                info.kind = "numeric"
        else:
            if ddt.startswith(("date", "time", "дат")) or _looks_datetime(s_tr):
                info.kind = "datetime"
            elif ddt.startswith(("text", "текст")) or _looks_text(s_tr):
                info.kind = "text"
            else:
                info.kind = "categorical"
                if s_tr.nunique() > 0.9 * n and s_tr.is_unique and n > 50:
                    info.kind, info.note = "id", "unique string per row"
        if info.kind == "categorical":
            info.high_card = nun > max(50, int(0.02 * n))
            unseen = set(s_te.dropna().astype(str)) - set(s_tr.dropna().astype(str))
            if unseen:
                prof.unseen_categories[c] = len(unseen)
        # Exact duplicate columns (keep the first).
        if info.kind not in ("id", "constant"):
            try:
                h = pd.util.hash_pandas_object(both, index=False).sum()
                key = (h, str(both.dtype))
                if key in seen_hashes and both.equals(pd.concat([train[seen_hashes[key]], test[seen_hashes[key]]], ignore_index=True)):
                    info.kind, info.note = "duplicate", f"same as {seen_hashes[key]}"
                else:
                    seen_hashes[key] = c
            except Exception:
                pass
        prof.columns[c] = asdict(info)
        bucket = {"numeric": prof.num_features, "bool": prof.num_features, "categorical": prof.cat_features,
                  "datetime": prof.datetime_features, "text": prof.text_features}.get(info.kind)
        if bucket is not None:
            bucket.append(c)
        else:
            prof.dropped[c] = info.note or info.kind

    try:
        prof.duplicate_rows = float(train[prof.features].duplicated().mean()) if prof.features else 0.0
    except Exception:
        prof.duplicate_rows = 0.0
    prof.target = _target_summary(train, target_cols, task_type, y)
    _detect_time(prof, train, test, time_hint)
    prof.validation = choose_validation(prof, task_type, n)
    if y is not None:
        prof.leakage = _leak_suspects(train, prof, y, task_type)
        try:
            prof.entity = find_entity_key(train, test, prof, y, task_type)
        except Exception as e:  # noqa: BLE001
            L.warning("entity key search failed: %s", e)
    if adversarial and prof.features:
        try:
            prof.adversarial = adversarial_validation(train[prof.features], test[prof.features],
                                                      prof.cat_features, seed=seed, n_jobs=n_jobs)
        except Exception as e:
            L.warning("adversarial validation failed: %s", e)
    prof.seconds = round(time.time() - t0, 2)
    return prof


def _key_codes(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return frame[cols].astype(str).groupby(cols, sort=False, dropna=False).ngroup().to_numpy()


def _entity_stats(k_tr: np.ndarray, k_te: np.ndarray, Y: np.ndarray, period, total: float) -> dict:
    """coverage: test rows whose key occurs in train; repeat: train rows whose key recurs
    (in two or more periods on a time split); purity: 1 - within-key spread / overall."""
    keys = pd.Series(k_tr)
    if period is not None:
        n_per = pd.Series(period).groupby(k_tr).nunique()
    else:
        n_per = keys.value_counts()
    rep = np.isin(k_tr, n_per.index[n_per >= 2].to_numpy())
    out = {"coverage": float(np.isin(k_te, k_tr).mean()), "repeat": float(rep.mean()), "purity": 0.0}
    if rep.sum() < 20 or total <= 0:
        return out
    sub = pd.DataFrame(Y[rep]).assign(_k=k_tr[rep])
    cols = [c for c in sub.columns if c != "_k"]
    dev = sub[cols] - sub.groupby("_k")[cols].transform("mean")
    within = float((dev ** 2).to_numpy().sum() / max(1, len(sub) * len(cols)))
    out["purity"] = float(max(0.0, 1.0 - np.sqrt(within / total)))
    return out


def find_entity_key(train: pd.DataFrame, test: pd.DataFrame, prof: DataProfile, y, task_type: str,
                    max_cols: int = 5, max_rows: int = 200_000) -> dict:
    """Columns that together identify a recurring entity (a building, a customer, a
    device): the key recurs in train (across periods on a time split), most test rows
    carry a key seen in train, and the target barely varies within a key. Its
    out-of-fold target mean is then the entity's own history. Greedy forward search on
    coverage x repeat x purity; {} when no key qualifies."""
    if y is None or len(train) < 200 or len(test) == 0:
        return {}
    cands = [c for c in prof.num_features + prof.cat_features if c != prof.time_column]
    if not cands:
        return {}
    idx = np.arange(len(train))
    if len(idx) > max_rows:
        idx = np.sort(np.random.default_rng(0).choice(idx, max_rows, replace=False))
    yy = np.asarray(y)[idx]
    if task_type == "regression":
        Y = yy.astype(float).reshape(-1, 1)
        Y = np.where(np.isfinite(Y), Y, np.nanmean(Y))
    else:
        Y = (yy.reshape(-1, 1) == np.unique(yy).reshape(1, -1)).astype(float)
    total = float(np.mean(Y.var(axis=0)))
    period = train[prof.time_column].to_numpy()[idx] if prof.time_column else None
    both = pd.concat([train[cands].iloc[idx], test[cands]], ignore_index=True)
    n_tr = len(idx)
    key: list[str] = []
    best = {"score": 0.0}
    for _ in range(max_cols):
        step = None
        for c in cands:
            if c in key:
                continue
            codes = _key_codes(both, key + [c])
            st = _entity_stats(codes[:n_tr], codes[n_tr:], Y, period, total)
            st["score"] = st["coverage"] * st["repeat"] * st["purity"]
            if step is None or st["score"] > step[1]["score"]:
                step = (c, st)
        if step is None or step[1]["score"] <= best["score"] + 0.01:
            break
        key.append(step[0])
        best = step[1]
    if not key or best["purity"] < 0.5 or best["coverage"] < 0.3 or best["repeat"] < 0.3:
        return {}
    return {"cols": key, **{k: round(v, 3) for k, v in best.items()}}


def _target_summary(train, target_cols, task_type, y) -> dict:
    t = target_cols[0]
    col = train[t] if t in train.columns else pd.Series(y)
    if task_type == metrics.REGRESSION:
        v = pd.to_numeric(pd.Series(y) if y is not None else col, errors="coerce").dropna()
        q = v.quantile([0.0, 0.01, 0.5, 0.99, 1.0]).round(4).tolist()
        return {"mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4),
                "skew": round(float(v.skew()), 3), "min": q[0], "p01": q[1], "median": q[2],
                "p99": q[3], "max": q[4], "nonnegative": bool((v >= 0).all()),
                "integer": bool(np.allclose(v % 1, 0)), "n_unique": int(v.nunique())}
    vc = col.astype(str).value_counts()
    return {"n_classes": int(len(vc)), "counts": {k: int(v) for k, v in vc.items()},
            "imbalance_ratio": round(float(vc.max() / max(vc.min(), 1)), 2)}


def _detect_time(prof: DataProfile, train, test, hint: str | None) -> None:
    cands = []
    for c in prof.num_features + prof.cat_features + prof.datetime_features:
        if c not in test.columns:
            continue
        if c in prof.datetime_features:
            a = pd.to_datetime(train[c], errors="coerce", format="mixed")
            b = pd.to_datetime(test[c], errors="coerce", format="mixed")
            kind = "datetime"
        else:
            a = pd.to_numeric(train[c], errors="coerce")
            b = pd.to_numeric(test[c], errors="coerce")
            kind = "ordinal"
            if a.nunique() > 1000 or a.nunique() < 2:
                continue
        if a.notna().mean() < 0.98 or b.notna().mean() < 0.98:
            continue
        if b.min() >= a.max():
            cands.append((0 if c == hint else 1, a.nunique(), c, kind))
    if hint and hint in train.columns and not cands:
        L.info("time column hint %s not confirmed by data", hint)
    if cands:
        cands.sort()
        _, _, c, kind = cands[0]
        prof.time_column, prof.time_kind = c, kind
        L.info("time split detected on %s (%s): test lies after train", c, kind)


def choose_validation(prof: DataProfile, task_type: str, n: int) -> dict:
    k = 5 if n >= 1000 else (3 if n < 300 else 5)
    if n > 1_000_000:
        k = 3
    if prof.time_column:
        return {"scheme": "group_time", "column": prof.time_column, "kind": prof.time_kind, "n_splits": k,
                "why": "test lies after train on this column; validate on unseen periods"}
    # Small data: repeat the CV with other shuffles to cut selection noise.
    repeats = 3 if n < 5000 else (2 if n < 20000 else 1)
    if task_type in metrics.CLASSIFICATION:
        return {"scheme": "stratified", "n_splits": k, "repeats": repeats, "why": "classification"}
    return {"scheme": "kfold_binned", "n_splits": k, "repeats": repeats,
            "why": "regression; stratified on target quantiles"}


def make_folds(validation: dict, train: pd.DataFrame, y_enc: np.ndarray, task_type: str,
               seed: int = 42) -> np.ndarray:
    """Fold id per training row (0..k-1). Same folds for every experiment."""
    from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

    n = len(train)
    k = int(validation.get("n_splits", 5))
    folds = np.full(n, -1, dtype=np.int16)
    idx = np.arange(n)
    scheme = validation.get("scheme")
    if scheme == "group_time":
        col = validation["column"]
        if validation.get("kind") == "datetime":
            t = pd.to_datetime(train[col], errors="coerce", format="mixed")
            groups = pd.qcut(t.rank(method="first"), q=min(k, max(2, t.nunique())), labels=False).to_numpy()
        else:
            groups = pd.to_numeric(train[col], errors="coerce").fillna(-1).to_numpy()
        n_groups = len(np.unique(groups))
        if n_groups >= 3:
            # Few periods: leave-one-period-out; many: k folds of whole periods.
            gkf = GroupKFold(n_splits=n_groups if n_groups <= 6 else min(k, n_groups))
            for f, (_, va) in enumerate(gkf.split(idx, groups=groups)):
                folds[va] = f
            return folds
        L.warning("too few periods (%d) for group folds; falling back", n_groups)
    if task_type in metrics.CLASSIFICATION:
        import warnings

        try:
            # Classes smaller than k only trigger a warning; they land in fewer folds.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
                for f, (_, va) in enumerate(skf.split(idx, y_enc)):
                    folds[va] = f
            return folds
        except ValueError:
            pass
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        for f, (_, va) in enumerate(kf.split(idx)):
            folds[va] = f
        return folds
    # Regression: stratify on target quantile bins for balanced folds.
    try:
        bins = pd.qcut(pd.Series(y_enc).rank(method="first"), q=min(10, max(2, n // (k * 5))), labels=False).to_numpy()
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for f, (_, va) in enumerate(skf.split(idx, bins)):
            folds[va] = f
    except Exception:
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        for f, (_, va) in enumerate(kf.split(idx)):
            folds[va] = f
    return folds


def eval_mask(train: pd.DataFrame, test: pd.DataFrame, folds: np.ndarray, cat_cols: list[str],
              max_card: int = 200) -> tuple[np.ndarray, dict]:
    """Rows to score in CV so that validation mimics the test.

    If the real test has (almost) no category values unseen in train, but a validation
    fold contains values unseen in its own training part (e.g. a state that only exists
    in the left-out year), those rows are excluded from scoring: they measure an
    extrapolation the test never asks for. Their OOF predictions are still produced.
    """
    mask = np.ones(len(train), dtype=bool)
    info: dict = {}
    for c in cat_cols:
        a = train[c].astype(str)
        if a.nunique() > max_card:
            continue
        b = test[c].astype(str)
        u_test = float((~b.isin(set(a))).mean())
        if u_test > 0.005:
            continue  # the test does extrapolate on this column: keep such rows
        unseen = np.zeros(len(train), dtype=bool)
        for f in np.unique(folds):
            va = folds == f
            seen = set(a[~va])
            unseen |= va & ~a.isin(seen).to_numpy()
        frac = float(unseen.mean())
        if frac > 0.002:
            mask &= ~unseen
            info[c] = {"masked_rows": int(unseen.sum()), "test_unseen": round(u_test, 4)}
    if mask.mean() < 0.5:  # never throw away most of the validation signal
        return np.ones(len(train), dtype=bool), {"skipped": "mask would drop >50% rows", **info}
    return mask, info


def _leak_suspects(train: pd.DataFrame, prof: DataProfile, y: np.ndarray, task_type: str) -> list:
    """Features that alone predict the target almost perfectly (flagged, not dropped)."""
    out = []
    yy = np.asarray(y, float)
    for c in prof.num_features[:500]:
        v = pd.to_numeric(train[c], errors="coerce")
        ok = v.notna().to_numpy()
        if ok.sum() < 30:
            continue
        if task_type == metrics.REGRESSION:
            r = pd.Series(v[ok].to_numpy()).corr(pd.Series(yy[ok]), method="spearman")
            if abs(r) > 0.995:
                out.append({"feature": c, "spearman": round(float(r), 4)})
        else:
            # A feature whose values partition the classes perfectly and is near-unique.
            if v.nunique() > 0.5 * len(v):
                g = pd.DataFrame({"v": v[ok].to_numpy(), "y": yy[ok]}).sort_values("v")
                purity = (g["y"].to_numpy()[1:] == g["y"].to_numpy()[:-1]).mean()
                base = (np.bincount(yy[ok].astype(int)) / ok.sum()) ** 2
                if purity > 0.98 and purity > base.sum() + 0.3:
                    out.append({"feature": c, "sorted_purity": round(float(purity), 4)})
    return out


def adversarial_validation(Xtr: pd.DataFrame, Xte: pd.DataFrame, cat_cols: list[str], seed: int = 42,
                           n_jobs: int = 8, max_rows: int = 40000) -> dict:
    """Train-vs-test classifier: AUC near 0.5 means no shift; top features show the drift."""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(seed)
    a = Xtr.iloc[rng.choice(len(Xtr), min(len(Xtr), max_rows), replace=False)]
    b = Xte.iloc[rng.choice(len(Xte), min(len(Xte), max_rows), replace=False)]
    X = pd.concat([a, b], ignore_index=True)
    for c in X.columns:
        if c in cat_cols or not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(str).astype("category")
    yv = np.r_[np.zeros(len(a)), np.ones(len(b))]
    oof = np.zeros(len(X))
    gains = np.zeros(X.shape[1])
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=seed).split(X, yv):
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, n_jobs=n_jobs, verbose=-1,
                               random_state=seed)
        m.fit(X.iloc[tr], yv[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        gains += m.booster_.feature_importance("gain")
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(yv, oof))
    order = np.argsort(-gains)
    tot = gains.sum() or 1.0
    top = [{"feature": X.columns[i], "share": round(float(gains[i] / tot), 3)} for i in order[:8]]
    return {"auc": round(auc, 4), "top_features": top}
