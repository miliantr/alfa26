from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import agent_submission as submission
import agent_task_parser as task_parser
from agent_io_utils import read_csv


def _tpl(md):
    spec = task_parser.parse_task(md)
    return spec, submission.SubmissionTemplate.from_spec(spec)


def _target(spec):
    return read_csv(spec.train_path, usecols=spec.target_cols, dtype=str)[spec.target_cols[0]]


@pytest.mark.parametrize("fx", ["reg_task", "multi_task", "binary_proba_task", "proba_wide_task"])
def test_constant_is_valid_and_written(fx, request):
    md = request.getfixturevalue(fx)
    spec, tpl = _tpl(md)
    df = tpl.constant(_target(spec))
    assert tpl.validate(df) == []
    assert tpl.write(df, "t")
    out = Path(spec.output_paths[0])
    assert out.exists() and submission.check_file(out, tpl) == []
    assert not list(out.parent.glob(".submission*.tmp"))  # atomic write leaves no temp


def test_ids_keep_exact_strings_and_order(tmp_path):
    from harness.synth import make_frame, write_task

    df = make_frame(50)
    df.insert(0, "id", [f"{i:05d}" for i in range(50)])  # leading zeros must survive
    df["y"] = np.arange(50.0)
    te = df.iloc[30:].drop(columns="y").iloc[::-1]  # reversed order in test
    md = write_task(tmp_path, df.iloc[:30], te, "y", "Метрика: RMSE")
    spec, tpl = _tpl(md)
    assert tpl.ids == list(te["id"])
    sub = tpl.from_predictions(np.ones(20))
    assert tpl.validate(sub) == []


def test_validation_catches_errors(reg_task, multi_task, binary_proba_task):
    spec, tpl = _tpl(reg_task)
    good = tpl.constant(_target(spec))
    bad = good.copy(); bad.loc[0, "y"] = np.nan
    assert any("non-finite" in e for e in tpl.validate(bad))
    bad = good.copy(); bad.loc[1, "y"] = np.inf
    assert any("non-finite" in e for e in tpl.validate(bad))
    assert any("columns" in e for e in tpl.validate(good[["y", "id"]]))
    assert any("rows" in e for e in tpl.validate(good.iloc[:-1]))
    assert any("order" in e for e in tpl.validate(good.iloc[::-1].reset_index(drop=True)))
    dup = good.copy(); dup.loc[1, "id"] = dup.loc[0, "id"]
    assert tpl.validate(dup)

    spec, tpl = _tpl(multi_task)
    good = tpl.constant(_target(spec))
    bad = good.copy(); bad.loc[0, "label"] = "zebra"
    assert any("outside allowed" in e for e in tpl.validate(bad))

    spec, tpl = _tpl(binary_proba_task)
    good = tpl.constant(_target(spec))
    bad = good.copy(); bad.loc[0, "target"] = 1.5
    assert any("[0, 1]" in e for e in tpl.validate(bad))


def test_invalid_frame_is_not_written(reg_task):
    spec, tpl = _tpl(reg_task)
    df = tpl.constant(_target(spec))
    df.loc[0, "y"] = np.nan
    assert not tpl.write(df, "bad")
    assert not Path(spec.output_paths[0]).exists()


def test_refuses_to_overwrite_inputs(reg_task):
    spec, tpl = _tpl(reg_task)
    before = Path(spec.train_path).read_bytes()
    spec.output_paths = [spec.train_path]
    assert not tpl.write(tpl.constant(_target(spec)), "danger")
    assert Path(spec.train_path).read_bytes() == before


def test_labels_written_as_ints(tmp_path):
    from harness.synth import make_frame, write_task

    df = make_frame(200)
    df.insert(0, "id", range(200))
    df["k"] = np.random.default_rng(0).integers(0, 3, 200)
    md = write_task(tmp_path, df.iloc[:150], df.iloc[150:].drop(columns="k"), "k",
                    "Метрика — Macro F1", output_kind="label")
    spec, tpl = _tpl(md)
    sub = tpl.from_predictions(["2"] * 50)
    assert tpl.validate(sub) == []
    tpl.write(sub)
    assert Path(spec.output_paths[0]).read_text().splitlines()[1].endswith(",2")


def test_emergency_writes_when_nothing_exists(reg_task):
    from agent_emergency import ensure_submission

    out = reg_task.parent / "submission.csv"
    assert not out.exists()
    ensure_submission(reg_task, reason="test")
    assert out.exists()
    spec, tpl = _tpl(reg_task)
    assert submission.check_file(out, tpl) == []
