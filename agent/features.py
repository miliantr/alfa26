"""Feature preparation and feature-engineering recipes.

``prepare_base`` turns raw train/test frames into a model-ready base: float32 numerics,
pandas categoricals with one shared vocabulary, expanded datetimes and text statistics.
Recipes then add columns on top of the base; a *feature set* is an ordered list of recipe
ops, identified by a hash so its output can be cached and reused across experiments.

Rules every recipe follows: no target use, fit statistics on train only, same columns on
train and test, row count unchanged.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import log

L = log.get("features")

NAN_TOKEN = "__nan__"


# --------------------------------------------------------------------------- base
def _expand_datetime(s: pd.Series, prefix: str) -> pd.DataFrame:
    t = pd.to_datetime(s, errors="coerce", format="mixed")
    out = pd.DataFrame(index=s.index)
    out[f"{prefix}__year"] = t.dt.year
    out[f"{prefix}__month"] = t.dt.month
    out[f"{prefix}__day"] = t.dt.day
    out[f"{prefix}__dow"] = t.dt.dayofweek
    out[f"{prefix}__doy"] = t.dt.dayofyear
    if (t.dt.hour.fillna(0) != 0).any() or (t.dt.minute.fillna(0) != 0).any():
        out[f"{prefix}__hour"] = t.dt.hour
        out[f"{prefix}__minute"] = t.dt.minute
    out[f"{prefix}__epoch_days"] = (t - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0
    return out.astype("float32")


def _text_stats(s: pd.Series, prefix: str) -> pd.DataFrame:
    v = s.fillna("").astype(str)
    return pd.DataFrame({
        f"{prefix}__len": v.str.len(),
        f"{prefix}__words": v.str.split().str.len(),
        f"{prefix}__digits": v.str.count(r"\d"),
        f"{prefix}__upper": v.str.count(r"[A-ZА-Я]"),
    }, index=s.index).astype("float32")


def prepare_base(train: pd.DataFrame, test: pd.DataFrame, profile) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    num, cat = list(profile.num_features), list(profile.cat_features)
    parts_tr, parts_te = [], []
    for c in num:
        parts_tr.append(pd.to_numeric(train[c], errors="coerce").astype("float32").rename(c))
        parts_te.append(pd.to_numeric(test[c], errors="coerce").astype("float32").rename(c))
    for c in cat:
        a = train[c].astype("string").fillna(NAN_TOKEN).astype(str)
        b = test[c].astype("string").fillna(NAN_TOKEN).astype(str)
        # Shared vocabulary from train+test values: an unsupervised use of test features
        # (no labels) that keeps category codes consistent; documented in NOTES.md.
        vocab = pd.Index(sorted(set(a.unique()) | set(b.unique())))
        parts_tr.append(pd.Series(pd.Categorical(a, categories=vocab), index=train.index, name=c))
        parts_te.append(pd.Series(pd.Categorical(b, categories=vocab), index=test.index, name=c))
    extra_num = []
    for c in profile.datetime_features:
        dtr, dte = _expand_datetime(train[c], c), _expand_datetime(test[c], c)
        parts_tr += [dtr[x] for x in dtr.columns]
        parts_te += [dte[x] for x in dte.columns]
        extra_num += list(dtr.columns)
    for c in profile.text_features:
        ttr, tte = _text_stats(train[c], c), _text_stats(test[c], c)
        parts_tr += [ttr[x] for x in ttr.columns]
        parts_te += [tte[x] for x in tte.columns]
        extra_num += list(ttr.columns)
    Xtr = pd.concat(parts_tr, axis=1) if parts_tr else pd.DataFrame(index=train.index)
    Xte = pd.concat(parts_te, axis=1) if parts_te else pd.DataFrame(index=test.index)
    Xtr.index = pd.RangeIndex(len(Xtr))
    Xte.index = pd.RangeIndex(len(Xte))
    meta = {"num_cols": num + extra_num, "cat_cols": cat, "text_cols": list(profile.text_features),
            "raw_text": {c: (train[c].fillna("").astype(str).tolist(), test[c].fillna("").astype(str).tolist())
                         for c in profile.text_features[:3]}}
    return Xtr, Xte, meta


def cat_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if isinstance(df[c].dtype, pd.CategoricalDtype)]


def num_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not isinstance(df[c].dtype, pd.CategoricalDtype)]


# ------------------------------------------------------------------------ recipes
def _safe_name(s: str) -> str:
    return re.sub(r"[^\w]+", "_", s)[:80]


def r_freq_encoding(Xtr, Xte, ctx, cols=None, **_):
    cols = [c for c in (cols or cat_columns(Xtr)) if c in Xtr.columns]
    a, b = {}, {}
    for c in cols:
        key = Xtr[c].astype(str)
        freq = key.value_counts(normalize=True)
        a[f"{c}__freq"] = key.map(freq).astype("float32").to_numpy()
        b[f"{c}__freq"] = Xte[c].astype(str).map(freq).fillna(0).astype("float32").to_numpy()
    return pd.DataFrame(a), pd.DataFrame(b)


def r_cat_combo(Xtr, Xte, ctx, pairs=None, top=4, **_):
    cats = [c for c in cat_columns(Xtr) if Xtr[c].nunique() < 5000]
    if pairs is None:
        ranked = [c for c in ctx.get("importance", []) if c in cats] + [c for c in cats if c not in ctx.get("importance", [])]
        ranked = ranked[:top]
        pairs = [(ranked[i], ranked[j]) for i in range(len(ranked)) for j in range(i + 1, len(ranked))]
    a, b = {}, {}
    for p, q in pairs:
        if p not in Xtr.columns or q not in Xtr.columns:
            continue
        name = f"{p}__x__{q}"
        sa = Xtr[p].astype(str) + "|" + Xtr[q].astype(str)
        sb = Xte[p].astype(str) + "|" + Xte[q].astype(str)
        vocab = pd.Index(sorted(set(sa.unique()) | set(sb.unique())))
        a[name] = pd.Categorical(sa, categories=vocab)
        b[name] = pd.Categorical(sb, categories=vocab)
    return pd.DataFrame(a), pd.DataFrame(b)


def r_num_pairs(Xtr, Xte, ctx, cols=None, top=6, ops=("ratio", "diff"), **_):
    nums = num_columns(Xtr)
    if cols is None:
        cols = [c for c in ctx.get("importance", []) if c in nums][:top] or nums[:top]
    cols = [c for c in cols if c in nums]
    a, b = {}, {}
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            p, q = cols[i], cols[j]
            for op in ops:
                name = f"{p}__{op}__{q}"
                for src, dst in ((Xtr, a), (Xte, b)):
                    x, y = src[p].astype("float64"), src[q].astype("float64")
                    if op == "ratio":
                        v = x / y.where(y.abs() > 1e-12)
                    elif op == "diff":
                        v = x - y
                    elif op == "prod":
                        v = x * y
                    else:
                        v = x + y
                    dst[name] = v.replace([np.inf, -np.inf], np.nan).astype("float32").to_numpy()
    return pd.DataFrame(a), pd.DataFrame(b)


def _column_groups(cols: list[str], min_size: int = 3) -> dict[str, list[str]]:
    """Group columns sharing a name token (e.g. *_temp_c, january_*)."""
    groups: dict[str, list[str]] = {}
    for c in cols:
        toks = [t for t in re.split(r"[_\W]+", c.lower()) if t]
        for t in set(toks):
            if len(t) >= 3 and not t.isdigit():
                groups.setdefault(t, []).append(c)
    out = {}
    for t, g in groups.items():
        if min_size <= len(g) <= max(min_size, int(0.8 * len(cols))) and len(g) < len(cols):
            key = tuple(sorted(g))
            if key not in {tuple(sorted(v)) for v in out.values()}:
                out[t] = g
    return dict(sorted(out.items(), key=lambda kv: -len(kv[1]))[:12])


def r_row_stats(Xtr, Xte, ctx, groups=None, **_):
    nums = num_columns(Xtr)
    groups = groups or _column_groups(nums)
    a, b = {}, {}
    for name, cols in groups.items():
        cols = [c for c in cols if c in nums]
        if len(cols) < 2:
            continue
        for src, dst in ((Xtr, a), (Xte, b)):
            m = src[cols].astype("float32")
            dst[f"grp_{_safe_name(name)}__mean"] = m.mean(axis=1).to_numpy()
            dst[f"grp_{_safe_name(name)}__std"] = m.std(axis=1).to_numpy()
            dst[f"grp_{_safe_name(name)}__min"] = m.min(axis=1).to_numpy()
            dst[f"grp_{_safe_name(name)}__max"] = m.max(axis=1).to_numpy()
    return pd.DataFrame(a).astype("float32"), pd.DataFrame(b).astype("float32")


def r_group_stats(Xtr, Xte, ctx, cats=None, nums=None, top_cat=3, top_num=5, **_):
    imp = ctx.get("importance", [])
    allc, alln = cat_columns(Xtr), num_columns(Xtr)
    cats = cats or ([c for c in imp if c in allc][:top_cat] or allc[:top_cat])
    nums = nums or ([c for c in imp if c in alln][:top_num] or alln[:top_num])
    a, b = {}, {}
    for g in cats:
        if g not in Xtr.columns:
            continue
        for x in nums:
            if x not in Xtr.columns:
                continue
            stats = Xtr.groupby(Xtr[g].astype(str), observed=True)[x].agg(["mean", "std"])
            for src, dst in ((Xtr, a), (Xte, b)):
                key = src[g].astype(str)
                mu = key.map(stats["mean"]).astype("float64")
                sd = key.map(stats["std"]).astype("float64")
                dst[f"{x}__dev_by__{g}"] = (src[x] - mu).astype("float32").to_numpy()
                dst[f"{x}__z_by__{g}"] = ((src[x] - mu) / sd.where(sd > 1e-12)).astype("float32").to_numpy()
    return pd.DataFrame(a), pd.DataFrame(b)


def r_missing_count(Xtr, Xte, ctx, **_):
    cols = [c for c in Xtr.columns if Xtr[c].isna().any()]
    a = {"row__n_missing": Xtr[cols].isna().sum(axis=1).astype("float32").to_numpy()} if cols else {}
    b = {"row__n_missing": Xte[cols].isna().sum(axis=1).astype("float32").to_numpy()} if cols else {}
    for c in cols[:20]:
        a[f"{c}__isna"] = Xtr[c].isna().astype("float32").to_numpy()
        b[f"{c}__isna"] = Xte[c].isna().astype("float32").to_numpy()
    return pd.DataFrame(a), pd.DataFrame(b)


def r_log1p(Xtr, Xte, ctx, cols=None, **_):
    nums = num_columns(Xtr)
    if cols is None:
        cols = []
        for c in nums:
            v = Xtr[c].dropna()
            if len(v) and v.min() >= 0 and v.skew() > 2:
                cols.append(c)
    a = {f"{c}__log1p": np.log1p(Xtr[c].clip(lower=0)).astype("float32").to_numpy() for c in cols if c in nums}
    b = {f"{c}__log1p": np.log1p(Xte[c].clip(lower=0)).astype("float32").to_numpy() for c in cols if c in nums}
    return pd.DataFrame(a), pd.DataFrame(b)


def r_text_tfidf_svd(Xtr, Xte, ctx, n_components=16, **_):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    a, b = {}, {}
    for c, (ta, tb) in (ctx.get("raw_text") or {}).items():
        vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        Ma = vec.fit_transform(ta)
        Mb = vec.transform(tb)
        k = max(1, min(n_components, Ma.shape[1] - 1))
        svd = TruncatedSVD(n_components=k, random_state=0)
        Za, Zb = svd.fit_transform(Ma), svd.transform(Mb)
        for i in range(k):
            a[f"{c}__svd{i}"] = Za[:, i].astype("float32")
            b[f"{c}__svd{i}"] = Zb[:, i].astype("float32")
    return pd.DataFrame(a), pd.DataFrame(b)


def r_cached_columns(Xtr, Xte, ctx, path=None, **_):
    """Columns produced earlier (e.g. by sandboxed LLM code) and stored on disk."""
    d = pd.read_pickle(path)
    return d["train"].reset_index(drop=True), d["test"].reset_index(drop=True)


RECIPES = {
    "freq_encoding": r_freq_encoding,
    "cat_combo": r_cat_combo,
    "num_pairs": r_num_pairs,
    "row_stats": r_row_stats,
    "group_stats": r_group_stats,
    "missing_count": r_missing_count,
    "log1p": r_log1p,
    "text_tfidf_svd": r_text_tfidf_svd,
    "cached_columns": r_cached_columns,
}

RECIPE_DOCS = {
    "freq_encoding": "frequency of each category value (train stats) for categorical columns",
    "cat_combo": "pairwise concatenations of the most important categoricals (args: pairs=[[a,b]], top)",
    "num_pairs": "ratio/diff/prod of pairs among top numeric features (args: cols, top, ops)",
    "row_stats": "row-wise mean/std/min/max over column groups sharing a name token (args: groups={name:[cols]})",
    "group_stats": "deviation and z-score of numeric features from their mean within a categorical (args: cats, nums)",
    "missing_count": "row missing-value count and is-NA indicators",
    "log1p": "log1p of skewed non-negative numeric features",
    "text_tfidf_svd": "TF-IDF + SVD components of text columns",
    "drop": "remove features (args: cols=[...])",
}


def feature_set_id(ops: list[dict]) -> str:
    blob = json.dumps(ops, sort_keys=True, default=str)
    return "fs_" + hashlib.sha1(blob.encode()).hexdigest()[:10]


def build_features(Xtr: pd.DataFrame, Xte: pd.DataFrame, ops: list[dict], ctx: dict,
                   cache_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply ``ops`` in order; each op's output is validated before it is kept."""
    fsid = feature_set_id(ops)
    if cache_dir is not None:
        p = Path(cache_dir) / f"{fsid}.pkl"
        if p.exists():
            try:
                d = pd.read_pickle(p)
                return d["train"], d["test"], d["info"]
            except Exception:
                pass
    info = {"fs_id": fsid, "ops": [], "errors": []}
    A, B = Xtr, Xte
    for op in ops:
        name = op.get("op")
        args = op.get("args", {}) or {}
        try:
            if name == "drop":
                cols = [c for c in args.get("cols", []) if c in A.columns]
                A, B = A.drop(columns=cols), B.drop(columns=cols)
                info["ops"].append({"op": name, "removed": cols})
                continue
            fn = RECIPES[name]
            na, nb = fn(A, B, {**ctx, **op.get("ctx", {})}, **args)
            na, nb = _validate_new(na, nb, A, B)
            if na.shape[1]:
                A = pd.concat([A, na], axis=1)
                B = pd.concat([B, nb], axis=1)
            info["ops"].append({"op": name, "added": list(na.columns)})
        except Exception as e:  # a broken recipe is skipped, never fatal
            info["errors"].append(f"{name}: {e!r}")
            L.warning("recipe %s failed: %r", name, e)
    info["n_features"] = A.shape[1]
    if cache_dir is not None:
        try:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            pd.to_pickle({"train": A, "test": B, "info": info}, Path(cache_dir) / f"{fsid}.pkl")
        except Exception:
            pass
    return A, B, info


def _validate_new(na: pd.DataFrame, nb: pd.DataFrame, A: pd.DataFrame, B: pd.DataFrame):
    if na.shape[1] == 0 or nb.shape[1] == 0:  # nothing to add (e.g. no eligible columns)
        return pd.DataFrame(index=A.index), pd.DataFrame(index=B.index)
    if len(na) != len(A) or len(nb) != len(B):
        raise ValueError(f"row count changed: {len(na)}/{len(A)}, {len(nb)}/{len(B)}")
    na = na.reset_index(drop=True)
    nb = nb.reset_index(drop=True)
    common = [c for c in na.columns if c in nb.columns and c not in A.columns]
    na, nb = na[common], nb[common]
    keep = []
    for c in common:
        if isinstance(na[c].dtype, pd.CategoricalDtype):
            keep.append(c)
            continue
        va = pd.to_numeric(na[c], errors="coerce").astype("float64")
        vb = pd.to_numeric(nb[c], errors="coerce").astype("float64")
        va = va.where(np.isfinite(va))
        vb = vb.where(np.isfinite(vb))
        if va.isna().mean() > 0.98 or va.nunique() <= 1:
            continue  # empty or constant on train
        if vb.isna().mean() > 0.98 and va.isna().mean() < 0.5:
            continue  # NaN explosion on test only
        na[c], nb[c] = va.astype("float32"), vb.astype("float32")
        keep.append(c)
    return na[keep], nb[keep]
