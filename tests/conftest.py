import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness.synth import make_frame, write_task  # noqa: E402


def _split(df, n_test=60):
    return df.iloc[:-n_test].reset_index(drop=True), df.iloc[-n_test:].reset_index(drop=True)


@pytest.fixture
def reg_task(tmp_path):
    df = make_frame(300, seed=1)
    df.insert(0, "id", np.arange(1000, 1300))
    df["y"] = df["x0"].fillna(0) * 3 + np.random.default_rng(0).normal(size=300)
    tr, te = _split(df)
    return write_task(tmp_path / "reg", tr, te.drop(columns="y"), "y",
                      "Метрика оценки — **MAE**, меньшее значение лучше.", output_kind="value")


@pytest.fixture
def multi_task(tmp_path):
    df = make_frame(400, seed=2)
    df.insert(0, "id", np.arange(400))
    df["label"] = np.random.default_rng(1).choice(["cat", "dog", "fox"], size=400)
    tr, te = _split(df)
    return write_task(tmp_path / "multi", tr, te.drop(columns="label"), "label",
                      "Основная метрика — **Macro F1**. Accuracy справочно.", output_kind="label")


@pytest.fixture
def binary_proba_task(tmp_path):
    df = make_frame(400, seed=3)
    df.insert(0, "id", np.arange(400))
    df["target"] = (df["x1"].fillna(0) > 0).astype(int)
    tr, te = _split(df)
    return write_task(tmp_path / "bin", tr, te.drop(columns="target"), "target",
                      "Метрика — ROC-AUC. Нужно предсказать вероятность класса 1.", output_kind="proba")


@pytest.fixture
def proba_wide_task(tmp_path):
    df = make_frame(400, seed=4)
    df.insert(0, "id", np.arange(400))
    df["kind"] = np.random.default_rng(2).choice(["A", "B", "C"], size=400)
    tr, te = _split(df)
    return write_task(tmp_path / "wide", tr, te.drop(columns="kind"), "kind",
                      "Метрика — LogLoss по вероятностям классов.", output_kind="proba_wide")
