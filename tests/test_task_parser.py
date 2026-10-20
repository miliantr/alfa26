from pathlib import Path

import numpy as np
import pandas as pd

from agent import task_parser
from harness.synth import make_frame, write_task

REPO = Path(__file__).resolve().parents[1]


def test_regression(reg_task):
    s = task_parser.parse_task(reg_task)
    assert s.target_cols == ["y"] and s.id_col == "id"
    assert s.task_type == "regression" and s.metric == "mae" and s.metric_source == "text"
    assert s.output_kind == "value"
    assert s.submission_cols == ["id", "y"]
    assert [Path(p).name for p in s.output_paths] == ["submission.csv"]
    assert Path(s.output_paths[0]).parent == reg_task.parent


def test_multiclass(multi_task):
    s = task_parser.parse_task(multi_task)
    assert s.task_type == "multiclass" and s.metric == "f1_macro"
    assert s.classes == ["cat", "dog", "fox"] and s.output_kind == "label"


def test_binary_proba(binary_proba_task):
    s = task_parser.parse_task(binary_proba_task)
    assert s.task_type == "binary" and s.metric == "roc_auc"
    assert s.output_kind == "proba" and s.positive_label == "1"
    assert s.label_dtype == "int"


def test_proba_wide(proba_wide_task):
    s = task_parser.parse_task(proba_wide_task)
    assert s.metric == "logloss" and s.output_kind == "proba_wide"
    assert s.proba_columns == ["A", "B", "C"]
    assert s.submission_cols == ["id", "A", "B", "C"]


def test_mentions_of_inputs_are_never_outputs(tmp_path):
    """Regression test: "В `train.csv` 64 столбца" once made train.csv an output path."""
    df = make_frame(100)
    df.insert(0, "id", range(100))
    df["y"] = np.arange(100.0)
    md = write_task(tmp_path, df.iloc[:80], df.iloc[80:].drop(columns="y"), "y", "Метрика: RMSE")
    text = md.read_text(encoding="utf-8") + "\nВ `train.csv` есть таргет, в `test.csv` его нет.\n"
    md.write_text(text, encoding="utf-8")
    s = task_parser.parse_task(md)
    assert [Path(p).name for p in s.output_paths] == ["submission.csv"]


def test_paths_found_by_search_without_mentions(tmp_path):
    df = make_frame(100)
    df.insert(0, "id", range(100))
    df["y"] = np.arange(100.0)
    md = write_task(tmp_path, df.iloc[:80], df.iloc[80:].drop(columns="y"), "y", "Метрика: RMSE",
                    sub="some/deep/folder")
    md.write_text("# Задача\nПредскажите y. Метрика — RMSE.\n", encoding="utf-8")
    s = task_parser.parse_task(md)
    assert s.train_path and s.train_path.endswith("train.csv")
    assert s.target_cols == ["y"] and s.metric == "rmse"


def test_no_dictionary_unit_and_no_metric(tmp_path):
    df = make_frame(200, seed=5)
    df.insert(0, "id", range(200))
    df["cls"] = np.random.default_rng(0).choice(["x", "y", "z", "w"], 200)
    md = write_task(tmp_path, df.iloc[:150], df.iloc[150:].drop(columns="cls"), "cls",
                    "Постройте модель.", unit=False, output_kind="label")
    s = task_parser.parse_task(md)
    assert s.task_type == "multiclass" and s.metric == "f1_macro" and s.metric_source == "default"
    assert any("default" in w for w in s.warnings)


def test_numeric_multiclass_and_regression_heuristics(tmp_path):
    rng = np.random.default_rng(0)
    df = make_frame(600, seed=6)
    df.insert(0, "id", range(600))
    df["grade"] = rng.integers(1, 6, 600)
    md = write_task(tmp_path / "a", df.iloc[:500], df.iloc[500:].drop(columns="grade"), "grade",
                    "Метрика — accuracy.", output_kind="label")
    s = task_parser.parse_task(md)
    assert s.task_type == "multiclass" and s.label_dtype == "int"
    md2 = write_task(tmp_path / "b", df.iloc[:500], df.iloc[500:].drop(columns="grade"), "grade",
                     "Метрика — RMSE.", output_kind="value")
    assert task_parser.parse_task(md2).task_type == "regression"


def test_llm_merge_heuristics_win(reg_task):
    s = task_parser.parse_task(reg_task)
    s = task_parser.merge_llm_view(s, {"task_type": "binary", "target_columns": ["y"], "metric": "RMSE"})
    assert s.metric == "mae" and s.task_type == "regression"
    assert any("keeping heuristic" in w for w in s.warnings)


def test_llm_fills_missing_metric(tmp_path):
    df = make_frame(200, seed=7)
    df.insert(0, "id", range(200))
    df["y"] = np.arange(200.0)
    md = write_task(tmp_path, df.iloc[:150], df.iloc[150:].drop(columns="y"), "y", "Постройте модель.")
    s = task_parser.parse_task(md)
    assert s.metric_source == "default"
    s = task_parser.merge_llm_view(s, {"task_type": "regression", "target_columns": ["y"], "metric": "MAE"})
    assert s.metric == "mae" and s.metric_source == "llm"


def test_public_tasks_if_present():
    for name, target, metric, kind in (("task1_energy", "site_eui", "rmse", "value"),
                                       ("task2_attacks", "attack_cat", "f1_macro", "label")):
        md = REPO / "data" / name / "task.md"
        if not md.exists():
            continue
        s = task_parser.parse_task(md)
        assert s.target_cols == [target] and s.metric == metric and s.output_kind == kind
        assert s.id_col == "id"
        assert [Path(p).name for p in s.output_paths] == ["submission.csv"]
