import numpy as np
import pandas as pd

import agent_profiler as profiler
from agent_research_loop import ResearchLoop, Tuner, clamp_params


def test_entity_history_is_tried_first():
    from types import SimpleNamespace

    ent = {"cols": ["floor_area", "year_built", "facility_type"], "coverage": 0.79, "repeat": 0.86, "purity": 0.73}
    h = ResearchLoop._priority_hypotheses(SimpleNamespace(prof=SimpleNamespace(entity=ent)))
    assert len(h) == 1 and h[0]["recipe"] == "make_key" and h[0]["args"]["te"] is True
    assert h[0]["args"]["cols"] == ent["cols"]
    assert ResearchLoop._priority_hypotheses(SimpleNamespace(prof=SimpleNamespace(entity={}))) == []


def test_clamp_params_whitelist_and_ranges():
    p, dropped = clamp_params("lgbm", {"num_leaves": 100000, "learning_rate": -1, "rm_rf": True,
                                       "extra_trees": 1, "min_child_samples": "50"})
    assert p == {"num_leaves": 1024, "learning_rate": 0.005, "extra_trees": True, "min_child_samples": 50}
    assert dropped == ["rm_rf"]
    p, dropped = clamp_params("xgb", {"max_depth": "deep"})
    assert p == {} and dropped == ["max_depth"]


def test_tuner_ask_tell_cycle():
    t = Tuner(seed=0, direction="minimize")
    seen = []
    for i in range(6):
        params = t.ask("lgbm", 1000)
        # ranges shrink with data size: no degenerate (constant-prediction) trials
        assert params and 8 <= params["num_leaves"] <= 50 and params["min_child_samples"] <= 25
        t.tell(params, float(i))
        seen.append(params["num_leaves"])
    assert len(set(seen)) > 1
    t.tell({"unknown": 1}, 1.0)  # telling an unknown trial is a no-op


class _FakeSpec:
    def __init__(self, task_type):
        self.task_type, self.metric, self.text, self.title, self.dictionary = task_type, "rmse", "", "t", []
        self.classes = []


class _FakeSettings:
    seed = 0
    llm_timeout = 10


def _prof(task):
    rng = np.random.default_rng(0)
    tr = pd.DataFrame({"jan_temp": rng.normal(size=300), "feb_temp": rng.normal(size=300),
                       "mar_temp": rng.normal(size=300), "x": rng.normal(size=300),
                       "c": rng.choice(list("abc"), 300), "y": rng.lognormal(size=300)})
    tr.loc[:20, "x"] = np.nan
    te = tr.drop(columns="y").iloc[:50]
    return profiler.profile(tr, te, ["y"], None, task, adversarial=False, y=tr["y"].to_numpy())


def test_deterministic_queue_is_task_aware():
    prof = _prof("regression")
    loop = ResearchLoop.__new__(ResearchLoop)
    loop.prof, loop.spec = prof, _FakeSpec("regression")
    q = loop._deterministic_queue()
    actions = [(h["action"], h.get("recipe") or h.get("family")) for h in q]
    assert ("feature_engineering", "freq_encoding") in actions
    assert ("feature_engineering", "row_stats") in actions      # *_temp group exists
    assert ("feature_engineering", "missing_count") in actions  # x has missing values
    assert any(h.get("target_transform") == "sqrt" for h in q)  # skewed non-negative target
    assert sum(1 for a, _ in actions if a == "tune") >= 3


def test_expensive_family_without_blend_weight_is_skipped():
    from types import SimpleNamespace as NS

    lgb = NS(id="e1", model={"family": "lgbm"})
    cat = NS(id="e2", model={"family": "catboost"})
    est = {"lgbm": 60.0, "catboost": 1000.0, "xgb": 120.0}
    pipe = NS(sub_desc={"members": ["e1", "e2"], "weights": [1.0, 0.0]}, ok=[lgb, cat],
              estimate_seconds=lambda fam, n=None: est[fam])
    loop = ResearchLoop.__new__(ResearchLoop)
    loop.pipe, loop.champ = pipe, lgb
    loop.deadline = NS(research_remaining=lambda: 3000.0)
    assert loop._worth_it("lgbm")            # champion family
    assert not loop._worth_it("catboost")    # no weight, 1000 s > 10% of 3000 s
    assert loop._worth_it("xgb")             # cheap enough to try
    pipe.sub_desc = {"members": ["e1", "e2"], "weights": [0.5, 0.5]}
    assert loop._worth_it("catboost")        # carries blend weight now
