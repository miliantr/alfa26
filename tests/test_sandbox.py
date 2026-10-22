import numpy as np
import pandas as pd
import pytest

import agent_sandbox as sandbox


@pytest.fixture
def frames():
    rng = np.random.default_rng(0)
    tr = pd.DataFrame({"a": rng.normal(size=120), "b": rng.uniform(1, 3, 120), "c": rng.choice(list("xy"), 120)})
    te = pd.DataFrame({"a": rng.normal(size=40), "b": rng.uniform(1, 3, 40), "c": rng.choice(list("xy"), 40)})
    return tr, te


GOOD = """
import numpy as np
def transform(df):
    out = pd.DataFrame(index=df.index)
    out["ab_ratio"] = df["a"] / df["b"]
    out["c_is_x"] = (df["c"] == "x").astype(int)
    return out
"""

GOOD_TWO_ARG = """
def transform(train, test):
    m = train["b"].mean()
    return (pd.DataFrame({"b_dev": train["b"] - m}), pd.DataFrame({"b_dev": test["b"] - m}))
"""


def test_good_code_runs(frames, tmp_path):
    tr, te = frames
    res = sandbox.run_feature_code(GOOD, tr, te, tmp_path, "g1", timeout=60)
    assert res["status"] == "ok", res
    d = pd.read_pickle(res["path"])
    assert len(d["train"]) == 120 and len(d["test"]) == 40
    assert list(d["train"].columns) == list(d["test"].columns) == ["llm_g1_ab_ratio", "llm_g1_c_is_x"]
    res2 = sandbox.run_feature_code(GOOD_TWO_ARG, tr, te, tmp_path, "g2", timeout=60)
    assert res2["status"] == "ok", res2


@pytest.mark.parametrize("code,why", [
    ("import os\ndef transform(df):\n    os.system('echo x')\n    return df", "import not allowed"),
    ("def transform(df):\n    open('x.txt','w').write('1')\n    return df", "name not allowed: open"),
    ("def transform(df):\n    return df.__class__.__bases__", "attribute not allowed"),
    ("def transform(df):\n    df.to_csv('leak.csv')\n    return df", "attribute not allowed: to_csv"),
    ("def transform(df):\n    return eval('1')", "name not allowed: eval"),
    ("from subprocess import run\ndef transform(df):\n    return df", "import not allowed"),
    ("def f(df):\n    return df", "no function named transform"),
    ("def transform(df:\n", "syntax error"),
])
def test_static_rejections(code, why):
    probs = sandbox.check_code(code)
    assert any(why in p for p in probs), probs


def test_runtime_import_guard_and_timeout(frames, tmp_path):
    tr, te = frames
    # __import__ via builtins is not reachable; a dynamic import attempt fails at runtime
    sneaky = "def transform(df):\n    m = [x for x in ().__class__.__mro__]\n    return df"
    assert sandbox.run_feature_code(sneaky, tr, te, tmp_path, "s1")["status"] == "rejected"
    slow = "def transform(df):\n    x = 0\n    while True:\n        x += 1\n    return df"
    res = sandbox.run_feature_code(slow, tr, te, tmp_path, "s2", timeout=3)
    assert res["status"] == "timeout"


def test_output_validation(frames, tmp_path):
    tr, te = frames
    rows = "def transform(df):\n    return df.iloc[:10][['a']].rename(columns={'a': 'z'})"
    assert sandbox.run_feature_code(rows, tr, te, tmp_path, "v1")["status"] == "invalid"
    const = "def transform(df):\n    return pd.DataFrame({'k': [1.0] * len(df)})"
    assert sandbox.run_feature_code(const, tr, te, tmp_path, "v2")["status"] == "invalid"
    infs = "def transform(df):\n    return pd.DataFrame({'k': df['a'] / 0.0})"
    assert sandbox.run_feature_code(infs, tr, te, tmp_path, "v3")["status"] == "invalid"
    err = "def transform(df):\n    return df['missing_col']"
    assert sandbox.run_feature_code(err, tr, te, tmp_path, "v4")["status"] == "error"
