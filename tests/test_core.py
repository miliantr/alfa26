import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as skm

from agent import ensemble, features, postprocess, profiler

rng = np.random.default_rng(0)


@pytest.mark.parametrize("name,ref", [
    ("f1_macro", lambda y, l: skm.f1_score(y, l, average="macro")),
    ("f1_weighted", lambda y, l: skm.f1_score(y, l, average="weighted")),
    ("accuracy", skm.accuracy_score),
    ("balanced_accuracy", skm.balanced_accuracy_score),
    ("precision_macro", lambda y, l: skm.precision_score(y, l, average="macro", zero_division=0)),
])
def test_fast_label_scores_match_sklearn(name, ref):
    y = rng.integers(0, 5, 2000)
    lab = np.where(rng.random(2000) < 0.6, y, rng.integers(0, 5, 2000))
    lab[lab == 4] = 3  # one class never predicted
    assert postprocess.fast_label_score(name, y, lab, 5) == pytest.approx(ref(y, lab))


def _imbalanced(n=6000, k=4):
    y = rng.choice(k, n, p=[0.7, 0.2, 0.07, 0.03])
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += 1.2
    logits[:, 0] += 1.0  # model biased towards the majority class
    P = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return y, P


def test_class_weights_improve_macro_f1():
    y, P = _imbalanced()
    d = postprocess.fit("f1_macro", "multiclass", y, P)
    assert d["kind"] == "class_weights"
    assert postprocess.score("f1_macro", y, P, d) > postprocess.score("f1_macro", y, P, {"kind": "label"}) + 0.02


def test_binary_threshold_improves_f1():
    y = (rng.random(4000) < 0.1).astype(int)
    p = np.clip(0.05 + 0.3 * y + rng.normal(0, 0.15, 4000), 0, 1)
    P = np.column_stack([1 - p, p])
    d = postprocess.fit("f1", "binary", y, P)
    assert d["kind"] == "threshold" and d["threshold"] < 0.5
    assert postprocess.score("f1", y, P, d) >= postprocess.score("f1", y, P, {"kind": "label"})


def test_regression_clip_and_temperature():
    y = rng.uniform(0, 10, 1000)
    raw = y + rng.normal(0, 3, 1000)
    d = postprocess.fit("rmse", "regression", y, raw)
    assert d["kind"] == "clip"
    out = postprocess.apply(d, raw)
    assert out.min() >= 0 and out.max() <= 10
    yc, P = _imbalanced()
    Pk = P ** 3 / (P ** 3).sum(1, keepdims=True)  # over-confident
    dt = postprocess.fit("logloss", "multiclass", yc, Pk)
    assert dt["kind"] == "temperature" and dt["temperature"] > 1


def test_greedy_blend_beats_members():
    y = rng.normal(size=3000)
    a = y + rng.normal(0, 1, 3000)
    b = y + rng.normal(0, 1, 3000)
    loss = ensemble.proxy_loss("rmse", "regression", y, 0)
    w, best = ensemble.greedy_blend([a, b], loss)
    assert best < min(loss(a), loss(b)) and abs(w[0] - 0.5) < 0.25


def _frames():
    df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.uniform(1, 2, 200),
                       "jan_temp": rng.normal(size=200), "feb_temp": rng.normal(size=200),
                       "mar_temp": rng.normal(size=200), "c": rng.choice(list("xyz"), 200)})
    df.loc[:10, "a"] = np.nan
    tr, te = df.iloc[:150].reset_index(drop=True), df.iloc[150:].reset_index(drop=True)
    for x in (tr, te):
        x["c"] = pd.Categorical(x["c"], categories=list("xyz"))
    return tr, te


@pytest.mark.parametrize("op", sorted(set(features.RECIPES) - {"cached_columns", "text_tfidf_svd"}))
def test_recipes_keep_rows_and_columns_aligned(op):
    tr, te = _frames()
    A, B, info = features.build_features(tr, te, [{"op": op}], {"importance": ["b", "a", "c"]})
    assert len(A) == len(tr) and len(B) == len(te)
    assert list(A.columns) == list(B.columns)
    assert not info["errors"], info["errors"]


def test_broken_recipe_is_skipped():
    tr, te = _frames()
    A, B, info = features.build_features(tr, te, [{"op": "num_pairs", "args": {"cols": ["nope"]}},
                                                  {"op": "does_not_exist"}], {})
    assert list(A.columns) == list(tr.columns) and info["errors"]


def _time_data():
    tr = pd.DataFrame({"year": np.repeat([1, 2, 3, 4], 50), "x": rng.normal(size=200),
                       "state": np.r_[["s1"] * 150, ["s2"] * 50]})
    te = pd.DataFrame({"year": [5] * 40, "x": rng.normal(size=40), "state": ["s1"] * 20 + ["s2"] * 20})
    tr["y"] = rng.normal(size=200)
    return tr, te


def test_time_split_detected_and_group_folds():
    tr, te = _time_data()
    prof = profiler.profile(tr, te, ["y"], None, "regression", adversarial=False)
    assert prof.time_column == "year" and prof.validation["scheme"] == "group_time"
    folds = profiler.make_folds(prof.validation, tr, tr["y"].to_numpy(), "regression")
    for f in np.unique(folds):  # every fold is one whole period
        assert tr.loc[folds == f, "year"].nunique() == 1


def test_eval_mask_drops_rows_unseen_in_fold_but_not_in_test():
    tr, te = _time_data()
    tr.loc[tr.year == 4, "state"] = "s3"  # s3 exists only in year 4 ...
    te["state"] = ["s1", "s3"] * 20     # ... and the test uses it (seen in full train)
    folds = (tr["year"] - 1).to_numpy()
    mask, info = profiler.eval_mask(tr, te, folds, ["state"])
    assert "state" in info and not mask[tr.year.to_numpy() == 4].any() and mask[tr.year.to_numpy() < 4].all()
    te["state"] = ["s9"] * 40  # the test itself is unseen: keep everything
    mask2, info2 = profiler.eval_mask(tr, te, folds, ["state"])
    assert mask2.all() and not info2
