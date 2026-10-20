"""Orchestration of one task run.

Phases (each one leaves a valid submission on disk):
  0. constant prediction           — seconds after start
  1. quick model (fast LightGBM)   — first minutes
  2. model zoo on base features    — strong, diverse models
  3. AutoResearch loop             — LLM / deterministic hypotheses (research_loop.py)
  4. final assembly                — seed bagging of the best, blend, post-process

Multi-target tasks run phases 1-4 once per target column (each with its share of the
remaining time) and assemble the full submission after every target.
"""
from __future__ import annotations

import gc
import shutil
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import log, metrics, profiler
from .config import load_settings
from .deadline import Deadline
from .features import prepare_base
from .io_utils import read_csv, write_json
from .llm_client import make_client
from .pipeline import Pipeline
from .resources import detect_cpus, detect_memory_bytes, snapshot
from .submission import SubmissionTemplate
from .task_parser import LLM_SCHEMA, PARSE_SYSTEM, llm_parse_prompt, merge_llm_view, parse_task, sub_spec


def make_run_dir(task_md: Path, configured: str | None) -> Path:
    cands = [Path(configured)] if configured else []
    cands.append(task_md.parent / "agent_run" / time.strftime("%Y%m%d_%H%M%S"))
    for c in cands:
        try:
            c.mkdir(parents=True, exist_ok=True)
            probe = c / ".probe"
            probe.write_text("ok")
            probe.unlink()
            return c
        except Exception:
            continue
    return Path(tempfile.mkdtemp(prefix="agent_run_"))


# ----------------------------------------------------------------------- data load
def load_data(spec, L):
    train = read_csv(Path(spec.train_path))
    test = read_csv(Path(spec.test_path))
    t = spec.target_cols[0]
    y_str = read_csv(Path(spec.train_path), usecols=[t], dtype=str)[t]
    if spec.task_type == metrics.REGRESSION:
        y = pd.to_numeric(y_str, errors="coerce").to_numpy(float)
        keep = np.isfinite(y)
    else:
        idx = {c: i for i, c in enumerate(spec.classes)}
        y = y_str.map(idx).to_numpy()
        keep = ~pd.isna(y)
        y = np.where(keep, y, -1).astype(int)
    if (~keep).any():
        L.warning("dropping %d training rows with missing target", int((~keep).sum()))
        train = train.loc[keep].reset_index(drop=True)
        y = y[keep]
    return train, test, y


def objective_for(metric_name: str) -> tuple[str, str]:
    """(objective, target_transform) for regression metrics."""
    return {
        "rmse": ("l2", "none"), "mse": ("l2", "none"), "r2": ("l2", "none"),
        "rmsle": ("l2", "log1p"), "msle": ("l2", "log1p"),
        "mae": ("l1", "none"), "medae": ("l1", "none"), "smape": ("l1", "none"),
        "wape": ("l1", "none"), "mape": ("mape", "none"),
    }.get(metric_name, ("l2", "none"))


def zoo_plan(task: str, metric_name: str, prof, n_rows: int) -> list[tuple[str, dict]]:
    m = metrics.REGISTRY[metric_name]
    plan: list[tuple[str, dict]] = []
    if task == metrics.REGRESSION:
        obj, tt = objective_for(metric_name)
        plan.append(("lgbm", {"family": "lgbm", "objective": obj, "target_transform": tt}))
        skew = abs(prof.target.get("skew", 0) or 0)
        if tt == "none" and prof.target.get("nonnegative") and skew > 1.0 and obj == "l2":
            plan.append(("lgbm_log1p", {"family": "lgbm", "objective": "l2", "target_transform": "log1p"}))
        plan.append(("catboost", {"family": "catboost", "objective": obj, "target_transform": tt}))
        plan.append(("xgb", {"family": "xgb", "objective": obj, "target_transform": tt}))
        plan.append(("linear", {"family": "linear", "objective": obj, "target_transform": tt}))
        if n_rows <= 300_000:
            plan.append(("extratrees", {"family": "extratrees", "target_transform": tt}))
        return plan
    imbalance = prof.target.get("imbalance_ratio", 1.0) or 1.0
    balanced_metric = metric_name in ("f1_macro", "balanced_accuracy", "recall_macro", "precision_macro",
                                      "mcc", "kappa", "qwk", "f1") or (m.needs == "label" and imbalance > 5)
    plan.append(("lgbm", {"family": "lgbm"}))
    if balanced_metric and imbalance > 3:
        plan.append(("lgbm_cw_sqrt", {"family": "lgbm", "class_weight": "sqrt"}))
    plan.append(("catboost", {"family": "catboost"}))
    plan.append(("xgb", {"family": "xgb"}))
    plan.append(("linear", {"family": "linear"}))
    if n_rows <= 300_000:
        plan.append(("extratrees", {"family": "extratrees"}))
    return plan


# ---------------------------------------------------------------------------- main
def run(task_md: Path) -> None:
    settings = load_settings()
    deadline = Deadline(settings.time_budget)
    task_md = Path(task_md).resolve()
    run_dir = make_run_dir(task_md, settings.run_dir)
    L = log.setup(run_dir)
    L.info("task=%s run_dir=%s budget=%.0fs llm=%s", task_md, run_dir, settings.time_budget, settings.llm_mode)
    summary: dict = {"phases": [], "resources": snapshot(), "settings": settings.to_dict(), "targets": {}}

    def phase(name, **kw):
        summary["phases"].append({"phase": name, "elapsed": round(deadline.elapsed(), 1), **kw})
        write_json(run_dir / "summary.json", summary)

    # ---- phase 0: constant -------------------------------------------------------
    spec = parse_task(task_md)
    L.info("spec: %s", spec.summary())
    for w in spec.warnings:
        L.warning("spec: %s", w)
    multi = len(spec.target_cols) > 1
    tpl = SubmissionTemplate.from_spec(spec)
    subs: dict = {}
    y0 = None
    if multi:
        subs = {t: sub_spec(spec, t, run_dir / f"target_{i}" / "submission_part.csv")
                for i, t in enumerate(spec.target_cols)}
        _write_constant_parts(subs)
        _assemble(spec, tpl, subs, L)
    else:
        y0 = read_csv(Path(spec.train_path), usecols=spec.target_cols[:1], dtype=str)[spec.target_cols[0]]
        tpl.write(tpl.constant(y0), "constant")
    phase("constant")

    # ---- task understanding by the LLM (cross-checked; heuristics win) ------------
    llm = make_client(settings, run_dir)
    view: dict = {}
    try:
        if llm.available():
            view = llm.chat_json(PARSE_SYSTEM, llm_parse_prompt(spec), schema=LLM_SCHEMA, max_tokens=700,
                                 purpose="task_parse", timeout=min(90.0, settings.llm_timeout)) or {}
            if view and not multi:
                kind_before = spec.output_kind
                spec = merge_llm_view(spec, view)
                L.info("llm view merged: metric=%s (%s) notes=%s", spec.metric, spec.metric_source,
                       str(view.get("domain_notes", ""))[:120])
                if spec.output_kind != kind_before:
                    tpl = SubmissionTemplate.from_spec(spec)
                    tpl.write(tpl.constant(y0), "constant")
    except Exception as e:  # noqa: BLE001
        L.warning("llm task parsing skipped: %r", e)
    summary["llm_view"] = view
    phase("understanding", metric=spec.metric, llm=llm.usage())

    try:
        if not multi:
            solve_target(spec, tpl, settings, deadline, run_dir, llm, view, L, summary, phase)
        else:
            names = list(subs)
            for i, t in enumerate(names):
                share = max(60.0, deadline.remaining() / (len(names) - i))
                sub_deadline = Deadline(share, start_time=deadline.now())
                sub_dir = Path(subs[t].output_paths[0]).parent
                sub_tpl = SubmissionTemplate.from_spec(subs[t])
                L.info("target %d/%d: %s (%s, %s) with %.0fs", i + 1, len(names), t, subs[t].task_type,
                       subs[t].metric, share)
                try:
                    solve_target(subs[t], sub_tpl, settings, sub_deadline, sub_dir, llm, view, L,
                                 summary["targets"].setdefault(t, {}), None)
                except Exception as e:  # noqa: BLE001
                    L.error("target %s failed: %r", t, e)
                _assemble(spec, tpl, subs, L)
    finally:
        summary["metric"] = spec.metric
        summary["llm_usage"] = llm.usage()
        summary["elapsed"] = round(deadline.elapsed(), 1)
        write_json(run_dir / "summary.json", summary)
        L.info("done in %.0fs", deadline.elapsed())


def solve_target(spec, tpl, settings, deadline, run_dir, llm, view, L, summary, phase) -> Pipeline | None:
    """Phases 1-4 for one target; every improvement is written through ``tpl``."""
    if phase is None:
        def phase(name, **kw):
            summary.setdefault("phases", []).append({"phase": name, "elapsed": round(deadline.elapsed(), 1), **kw})
    pipe = None
    run_dir = Path(run_dir)
    try:
        # ---- data, profile, base features, folds --------------------------------
        train, test, y = load_data(spec, L)
        n_jobs = detect_cpus(cap=settings.max_cpus)
        mem_limit = int(detect_memory_bytes() * settings.mem_fraction)
        hint = view.get("time_column") if isinstance(view.get("time_column"), str) else None
        prof = profiler.profile(train, test, spec.target_cols, spec.id_col, spec.task_type,
                                spec.dictionary, y=y, seed=settings.seed, n_jobs=n_jobs, time_hint=hint)
        raw_cols = [c for c in prof.features if c in train.columns]
        raw_frames = (train[raw_cols].reset_index(drop=True), test[raw_cols].reset_index(drop=True))
        L.info("profile: %d num, %d cat, %d dt, %d text; dropped=%s; validation=%s; adv_auc=%s (%.1fs)",
               len(prof.num_features), len(prof.cat_features), len(prof.datetime_features),
               len(prof.text_features), prof.dropped, prof.validation.get("scheme"),
               prof.adversarial.get("auc"), prof.seconds)
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "profile.json", prof.to_dict())
        write_json(run_dir / "task_spec.json", spec.to_dict())
        Xtr, Xte, fmeta = prepare_base(train, test, prof)
        folds = profiler.make_folds(prof.validation, train, y, spec.task_type, seed=settings.seed)
        emask, minfo = profiler.eval_mask(train, test, folds, prof.cat_features)
        prof.validation["eval_mask"] = minfo
        if minfo:
            L.info("eval mask: scoring %d/%d rows (%s)", int(emask.sum()), len(emask), minfo)
        meta = {"task_type": spec.task_type, "n_classes": len(spec.classes), "classes": spec.classes,
                "metric": spec.metric, "n_features": Xtr.shape[1],
                "y_range": [float(np.min(y)), float(np.max(y))] if spec.task_type == metrics.REGRESSION else None,
                "high_card_cats": [c for c in prof.cat_features if prof.columns[c]["high_card"]],
                **{k: v for k, v in fmeta.items()}}
        base_path = run_dir / "base.pkl"
        pd.to_pickle({"Xtr": Xtr, "Xte": Xte, "y": y, "folds": folds, "meta": meta, "eval_mask": emask}, base_path)
        del train, test
        gc.collect()
        phase("prepared", n_features=Xtr.shape[1], folds=int(folds.max() + 1))

        pipe = Pipeline(run_dir=run_dir, base_path=base_path, y=y, folds=folds, meta=meta,
                        metric=spec.metric, task=spec.task_type, settings=settings, deadline=deadline,
                        template=tpl, spec=spec, n_jobs=n_jobs, mem_limit=mem_limit, eval_mask=emask)

        # ---- phase 1: quick model -----------------------------------------------
        obj, tt = objective_for(spec.metric) if spec.task_type == metrics.REGRESSION else ("l2", "none")
        for name, model in (("quick_lgbm", {"family": "lgbm_fast", "objective": obj, "target_transform": tt}),
                            ("quick_linear", {"family": "linear", "objective": obj, "target_transform": tt}),
                            ("quick_trees", {"family": "extratrees", "params": {"n_estimators": 100}})):
            e = pipe.run(name, model, source="quick",
                         timeout=max(60.0, min(900.0, deadline.research_remaining() * 0.2)))
            if e.status == "ok":
                pipe.refresh_submission("quick")
                break
        phase("quick", score=pipe.sub_score)

        # ---- phase 2: model zoo -------------------------------------------------
        for name, model in zoo_plan(spec.task_type, spec.metric, prof, len(Xtr)):
            est = pipe.estimate_seconds(model["family"])
            if not deadline.can_start(est * 1.2):
                L.info("skip %s: est %.0fs > research time left %.0fs", name, est, deadline.research_remaining())
                continue
            e = pipe.run(name, model, source="zoo")
            if e.status == "ok":
                pipe.refresh_submission(name)
        phase("zoo", score=pipe.sub_score)

        # ---- phase 3: AutoResearch ----------------------------------------------
        try:
            from .research_loop import ResearchLoop

            ResearchLoop(pipe, spec, prof, settings, deadline, run_dir, llm=llm, raw_frames=raw_frames,
                         base_X=Xtr, domain_notes=str(view.get("domain_notes", ""))).run()
        except Exception as e:  # noqa: BLE001
            L.error("research loop crashed: %r\n%s", e, traceback.format_exc())
        phase("research", score=pipe.sub_score)

        # ---- phase 4: final assembly --------------------------------------------
        final_assembly(pipe, deadline, L)
        phase("final", score=pipe.sub_score)
    except Exception as e:  # noqa: BLE001
        L.error("pipeline error: %r\n%s", e, traceback.format_exc())
        summary["error"] = repr(e)
    finally:
        if pipe is not None:
            summary["experiments"] = [x.record() for x in pipe.experiments]
            summary["submission"] = {"score": pipe.sub_score, **{k: v for k, v in pipe.sub_desc.items()}}
        _cleanup(run_dir)
        L.info("target %s done at %.0fs; submission OOF score = %s", spec.target_cols, deadline.elapsed(),
               pipe.sub_score if pipe else None)
    return pipe


def _write_constant_parts(subs: dict) -> None:
    for t, s in subs.items():
        Path(s.output_paths[0]).parent.mkdir(parents=True, exist_ok=True)
        t_tpl = SubmissionTemplate.from_spec(s)
        y = read_csv(Path(s.train_path), usecols=[t], dtype=str)[t]
        t_tpl.write(t_tpl.constant(y), f"constant:{t}")


def _assemble(spec, tpl: SubmissionTemplate, subs: dict, L) -> bool:
    """Merge per-target part files into the full submission (validated)."""
    try:
        values = {}
        for t, s in subs.items():
            df = pd.read_csv(s.output_paths[0], dtype=str, keep_default_na=False)
            col = df[t]
            values[t] = col.tolist() if s.output_kind == "label" else pd.to_numeric(col, errors="coerce").to_numpy()
        frame = tpl.frame({c: values[c] for c in tpl.value_cols})
        # Validate each column with its own target's rules, then write.
        errs = []
        for c in tpl.value_cols:
            t_tpl = SubmissionTemplate.from_spec(subs[c])
            errs += t_tpl.validate(frame[[x for x in t_tpl.columns]])
        if errs:
            L.error("assembled submission invalid: %s", errs[:3])
            return False
        from .io_utils import atomic_write_csv

        inputs = {Path(p).resolve() for p in (spec.train_path, spec.test_path, spec.sample_path, spec.dict_path) if p}
        ok = False
        for p in spec.output_paths:
            if Path(p).resolve() not in inputs:
                atomic_write_csv(frame, Path(p))
                ok = True
        if ok:
            L.info("multi-target submission assembled: %s", ", ".join(spec.output_paths))
        return ok
    except Exception as e:  # noqa: BLE001
        L.error("cannot assemble multi-target submission: %r", e)
        return False


def final_assembly(pipe: Pipeline, deadline: Deadline, L) -> None:
    """Seed-bag the best model while final-phase time allows, then re-blend."""
    best = pipe.best()
    if best is None:
        return
    for s in (1, 2):
        est = pipe.estimate_seconds(best.model["family"]) * 1.2
        if est > deadline.final_remaining() - 30:
            break
        e = pipe.run(f"{best.name}_seed{s}", best.model, feature_ops=best.feature_ops, drop_cols=best.drop_cols,
                     source="seed", seed=pipe.settings.seed + 1000 * s, parent=best.id, phase="final",
                     timeout=deadline.job_timeout(est, factor=2.0, phase="final"))
        if e.status == "ok":
            pipe.refresh_submission("seed")


def _cleanup(run_dir: Path) -> None:
    """Drop bulky caches; keep the journal, logs, specs and predictions."""
    shutil.rmtree(Path(run_dir) / "fs", ignore_errors=True)
    try:
        (Path(run_dir) / "base.pkl").unlink()
    except Exception:
        pass
