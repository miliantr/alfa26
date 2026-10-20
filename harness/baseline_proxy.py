"""Stand-ins for the organizers' baseline on a local holdout.

``baseline.py`` asks the LLM for "the simplest ML model that solves the task", so its
score is somewhere between a linear model and an untuned random forest. We compute both;
the normalized score against the stronger one is the conservative estimate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from agent import metrics
from agent.io_utils import read_csv
from agent.task_parser import TaskSpec


def _split_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    num, cat = [], []
    for c in df.columns:
        (num if pd.api.types.is_numeric_dtype(df[c]) else cat).append(c)
    return num, cat


def _pipeline(kind: str, is_reg: bool, num: list[str], cat: list[str], n_jobs: int):
    if kind == "linear":
        pre = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                  OneHotEncoder(handle_unknown="ignore", max_categories=30)), cat),
        ])
        est = LinearRegression() if is_reg else LogisticRegression(max_iter=2000)
    else:
        pre = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                  OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)), cat),
        ])
        est = (RandomForestRegressor(n_estimators=100, n_jobs=n_jobs, random_state=0) if is_reg
               else RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=0))
    return make_pipeline(pre, est)


def run_proxy(spec: TaskSpec, kind: str, out_path: Path, n_jobs: int = 8) -> Path:
    """Fit the proxy on the task's train.csv and write a submission-style file."""
    from agent.submission import SubmissionTemplate

    train = read_csv(Path(spec.train_path))
    test = read_csv(Path(spec.test_path))
    target = spec.target_cols[0]
    train = train[train[target].notna()].reset_index(drop=True)
    drop = [c for c in [spec.id_col] + spec.target_cols if c and c in train.columns]
    X = train.drop(columns=drop).copy()
    Xt = test[X.columns].copy()
    num, cat = _split_cols(pd.concat([X, Xt]))
    for c in cat:
        X[c], Xt[c] = X[c].astype(str), Xt[c].astype(str)
    is_reg = spec.task_type == metrics.REGRESSION
    pipe = _pipeline(kind, is_reg, num, cat, n_jobs)
    tpl = SubmissionTemplate.from_spec(spec)
    if is_reg:
        pipe.fit(X, pd.to_numeric(train[target]))
        df = tpl.from_predictions(pipe.predict(Xt))
    else:
        enc = {c: i for i, c in enumerate(spec.classes)}
        pipe.fit(X, train[target].astype(str).map(enc).to_numpy())
        proba = np.zeros((len(Xt), len(spec.classes)))
        proba[:, pipe.classes_] = pipe.predict_proba(Xt)
        if spec.output_kind == "label":
            df = tpl.from_predictions([spec.classes[i] for i in proba.argmax(1)])
        elif spec.output_kind == "proba":
            df = tpl.from_predictions(proba[:, spec.classes.index(spec.positive_label)])
        else:
            df = tpl.from_predictions(proba)
    df.to_csv(out_path, index=False)
    return out_path
