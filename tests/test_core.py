import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as skm

import agent_ensemble as ensemble
import agent_features as features
import agent_postprocess as postprocess
import agent_profiler as profiler

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


def test_stateful_recipes_are_refitted_inside_folds():
    """A validation fold's own rows must not shape its features: the test's rows never
    shape theirs (live task 1: such a feature won 4/4 folds and hurt the next period)."""
    tr, te = _frames()
    ops = [{"op": "group_stats", "args": {"cats": ["c"], "nums": ["b"]}}, {"op": "freq_encoding"},
           {"op": "make_key", "args": {"cols": ["c"]}}]
    A, B, info = features.build_features(tr, te, ops, {})
    assert features.has_refit(info) and not info["errors"]
    fit, val = np.arange(100), np.arange(100, 150)
    fa, fb, fc = features.refit_in_fold(A, B, info, fit, val, {})
    assert (len(fa), len(fb), len(fc)) == (100, 50, len(te)) and list(fa.columns) == list(A.columns)
    key_fit = tr["c"].iloc[:100].astype(str)
    mu = tr["b"].iloc[:100].groupby(key_fit).mean()
    np.testing.assert_allclose(fb["b__dev_by__c"], tr["b"].iloc[100:].to_numpy() - tr["c"].iloc[100:].astype(str).map(mu),
                               rtol=1e-5)
    np.testing.assert_allclose(fc["b__dev_by__c"], te["b"].to_numpy() - te["c"].astype(str).map(mu), rtol=1e-5)
    assert not np.allclose(fb["b__dev_by__c"], A["b__dev_by__c"].iloc[100:])  # full-train fit differs
    np.testing.assert_allclose(fb["c__freq"], tr["c"].iloc[100:].astype(str).map(key_fit.value_counts(normalize=True)),
                               rtol=1e-5)
    np.testing.assert_allclose(fa["key__c__count"], key_fit.map(key_fit.value_counts()), rtol=1e-6)
    assert isinstance(fa["key__c"].dtype, pd.CategoricalDtype) and (fa["key__c"] == A["key__c"].iloc[:100].to_numpy()).all()


def _panel(n_ent=300, noise=2.0, seen=0.8):
    """Buildings observed in years 1-4 (train) and year 5 (test, mostly the same ones)."""
    ent = pd.DataFrame({"area": rng.uniform(1e3, 1e5, n_ent).round(1), "built": rng.integers(1900, 2015, n_ent),
                        "kind": rng.choice(list("abcdef"), n_ent), "effect": rng.normal(0, 30, n_ent)})
    tr = pd.concat([ent.assign(year=yr, temp=rng.normal(size=n_ent)) for yr in (1, 2, 3, 4)], ignore_index=True)
    te = ent.sample(frac=seen, random_state=0).assign(year=5)
    te["temp"] = rng.normal(size=len(te))
    tr["y"] = 80 + tr["effect"] + 3 * tr["temp"] + rng.normal(0, noise, len(tr))
    return tr.drop(columns="effect"), te.drop(columns="effect")


def test_entity_key_found_for_recurring_entities_only():
    tr, te = _panel()
    prof = profiler.profile(tr, te, ["y"], None, "regression", y=tr["y"].to_numpy(), time_hint="year",
                            adversarial=False)
    ent = prof.entity
    assert ent and "area" in ent["cols"] and "year" not in ent["cols"] and "temp" not in ent["cols"]
    assert ent["coverage"] >= 0.75 and ent["repeat"] > 0.9 and ent["purity"] > 0.7
    # i.i.d. rows: no key recurs, so none is proposed
    iid = pd.DataFrame({"a": rng.normal(size=600), "b": rng.normal(size=600), "c": rng.choice(list("xyz"), 600)})
    iid["y"] = iid["a"] * 2 + rng.normal(size=600)
    prof2 = profiler.profile(iid.iloc[:450], iid.iloc[450:].drop(columns="y"), ["y"], None, "regression",
                             y=iid["y"].iloc[:450].to_numpy(), adversarial=False)
    assert prof2.entity == {}


def test_auto_smoothing_trusts_stable_entities_not_noisy_categories():
    import agent_worker as worker

    key = np.repeat(np.arange(200), 4).astype(str)
    stable = np.repeat(rng.normal(0, 30, 200), 4) + rng.normal(0, 1, 800)
    assert worker._auto_smooth(key, stable.reshape(-1, 1)) == 0.5
    noisy = rng.normal(0, 30, 800)  # the key says nothing about the target
    assert worker._auto_smooth(key, noisy.reshape(-1, 1)) >= 20.0


def test_worker_entity_history_with_auto_smoothing(tmp_path):
    import agent_worker as worker

    tr, te = _panel()
    y = tr.pop("y").to_numpy()
    for x in (tr, te):
        x["kind"] = pd.Categorical(x["kind"], categories=list("abcdef"))
    folds = (tr["year"].to_numpy() - 1).astype(int)
    pd.to_pickle({"Xtr": tr.reset_index(drop=True), "Xte": te.reset_index(drop=True), "y": y, "folds": folds,
                  "eval_mask": None, "folds_rep": None,
                  "meta": {"task_type": "regression", "n_classes": 0, "metric": "rmse",
                           "y_range": [float(y.min()), float(y.max())]}}, tmp_path / "base.pkl")
    base = {"base_path": str(tmp_path / "base.pkl"), "fs_cache_dir": str(tmp_path / "fs"), "metric": "rmse",
            "seed": 1, "n_jobs": 2, "time_limit": 120}
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    plain = worker.run({**base, "out_dir": str(tmp_path / "a"),
                        "model": {"family": "lgbm", "objective": "l2", "target_transform": "none"}})
    ent = worker.run({**base, "out_dir": str(tmp_path / "b"),
                      "feature_ops": [{"op": "make_key", "args": {"cols": ["area", "built", "kind"], "name": "entity_key"}}],
                      "model": {"family": "lgbm", "objective": "l2", "target_transform": "none",
                                "te_cols": ["entity_key"], "te_smooth": "auto"}})
    assert plain["status"] == ent["status"] == "ok"
    assert ent["score"] < 0.5 * plain["score"], (ent["score"], plain["score"])


def test_worker_refits_stateful_recipes_in_every_fold(tmp_path, monkeypatch):
    import agent_worker as worker

    tr, te = _frames()
    tr["a"] = tr["a"].fillna(0)
    y = (2 * tr["b"] + (tr["c"] == "x") + rng.normal(scale=0.1, size=len(tr))).to_numpy()
    folds = np.arange(len(tr)) % 3
    pd.to_pickle({"Xtr": tr, "Xte": te, "y": y, "folds": folds, "eval_mask": None, "folds_rep": None,
                  "meta": {"task_type": "regression", "n_classes": 0, "metric": "rmse",
                           "y_range": [float(y.min()), float(y.max())]}}, tmp_path / "base.pkl")
    calls = []
    real = features.refit_in_fold
    monkeypatch.setattr(features, "refit_in_fold", lambda *a, **k: calls.append(1) or real(*a, **k))
    job = {"base_path": str(tmp_path / "base.pkl"), "fs_cache_dir": str(tmp_path / "fs"), "out_dir": str(tmp_path),
           "feature_ops": [{"op": "group_stats", "args": {"cats": ["c"], "nums": ["b", "a"]}}],
           "model": {"family": "lgbm", "objective": "l2", "target_transform": "none"}, "metric": "rmse",
           "seed": 1, "n_jobs": 2, "time_limit": 60}
    res = worker.run(job)
    assert res["status"] == "ok" and len(calls) == 3 and np.isfinite(res["score"])
    assert np.isfinite(np.load(tmp_path / "oof.npy")).all() and np.load(tmp_path / "test.npy").shape == (len(te),)


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


def test_blend_not_locked_on_class_weighted_model():
    """Regression: the class-balanced proxy alone picked a class-weighted model by itself
    (it already leans to rare classes); build() must keep the better of both proxies."""
    r = np.random.default_rng(5)
    n, k = 6000, 4
    y = r.choice(k, n, p=[0.6, 0.25, 0.1, 0.05])
    folds = r.integers(0, 5, n)

    def model(noise_seed):
        z = np.random.default_rng(noise_seed).normal(0, 1.2, (n, k))
        z[np.arange(n), y] += 1.5
        z += np.log([0.6, 0.25, 0.1, 0.05])
        return np.exp(z) / np.exp(z).sum(1, keepdims=True)

    a, b = model(1), model(2)
    c = a * np.array([0.5, 1.0, 2.0, 4.0])
    c /= c.sum(1, keepdims=True)  # "class-weighted" copy of a
    oofs = [a, b, c]
    balanced = ensemble.proxy_loss("f1_macro", "multiclass", y, k)
    wb, _ = ensemble.greedy_blend(oofs, balanced)
    only_balanced = postprocess.crossfit("f1_macro", "multiclass", y, ensemble.blend(oofs, wb), folds)[0]
    res = ensemble.build("f1_macro", "multiclass", y, oofs, k, groups=folds)
    assert res["score"] >= only_balanced - 1e-12
