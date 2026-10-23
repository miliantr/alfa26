# Alfa-Future Hackathon 2026 — autonomous ML-research agent

One unchanged codebase that reads a task (`task.md` + CSVs), decides what kind of ML
problem it is and what to optimise, runs its own research loop under a hard 3-hour
budget, and always leaves a valid `submission.csv` behind. Built for the case
"AI agent for ML model development" (Changellenge × Alfa-Bank): the score that matters
comes from **two hidden tasks with unknown content and metrics**, so the design
priority is *floor first* (never crash, always beat a simple strong model), *ceiling
second*.

The platform contract (launch interface, env vars, paths, output format) is in
[CONTRACT.md](CONTRACT.md); every assumption and finding is logged in [NOTES.md](NOTES.md).

## Architecture

```
solution.py ── top-level guard ──────────────────────────────────────────────┐
   │                                                                         │ any exception
   ▼                                                                         ▼
agent_runner.py                                                     agent_emergency.py
   │ 0. parse task (task_parser) ─► constant submission (seconds)    (template constant, or
   │ 0b. LLM reads the Russian description -> JSON spec,              stdlib copy of sample)
   │     cross-checked with heuristics (heuristics win)
   │ 1. load + profile (profiler): types, ids, leaks, time split,
   │    adversarial validation, folds, eval mask
   │ 2. quick model  ─► submission
   │ 3. model zoo (LightGBM, CatBoost, XGBoost, linear, ExtraTrees) ─► blend ─► submission
   │ 4. AutoResearch loop (research_loop) ─► blend ─► submission after every gain
   │ 5. final assembly: seed bagging of the best model, re-blend, post-process
   ▼
agent_pipeline.py ── experiment pool, cost model, paired comparisons, journal
   │                        │
   │ job.json               │ OOF + test predictions (shared folds)
   ▼                        ▼
agent_executor.py ─► python -m agent_worker  (subprocess: hard timeout + RSS guard)
                agent_features.py (cached feature sets) · agent_models.py · agent_postprocess.py
```

The code is flat on purpose: the platform copies only the ZIP's root-level `.py` files
and `requirements.txt` into the task directory, so every module is a root-level
`agent_<name>.py` (`harness/ship.py` defines what ships; `make_submission.sh` refuses a
ZIP with subfolders or with an `agent_*` import that is not in it).

| Module (`agent_<name>.py`) | Role |
|---|---|
| `task_parser` | locate files from `task.md` (recursive fallback), id/target, task type, primary metric (Russian/English patterns, code snippets), output kind (`value` / `label` / `proba` / `proba_wide`), multi-target sub-specs, LLM cross-check |
| `metrics` | 25 metrics with direction, prediction kind and objective hint; unknown-metric fallback per task type; the case's normalized-score formula |
| `submission` | template from `test.csv` ids (exact strings, exact order), strict validation, atomic writes, never overwrites inputs |
| `deadline` | clock from process start; research / final / safety phases; job timeouts never cross a phase end |
| `profiler` | column typing, id-like/constant/duplicate columns, leak suspects, time-split detection, adversarial validation, fold choice, eval mask |
| `features` | base frame (shared category vocabulary, datetime/text expansion) + recipe library (frequency, combos, ratios, row stats over name groups, group deviations, missingness, composite keys, TF-IDF) |
| `models` | LightGBM / CatBoost / XGBoost / linear / ExtraTrees / RF with early stopping, soft time limits, class weights, metric-driven objectives |
| `worker` / `executor` | one CV run per subprocess; OOF target encoding inside folds; target transforms scored in original space |
| `postprocess` | decision rules fitted on OOF: per-class weights (macro-F1 etc.), binary thresholds, QWK cut-points, temperature scaling, clipping |
| `ensemble` | greedy forward selection with replacement on a metric-aware proxy (class-balanced log loss for macro metrics) |
| `llm_client` | budgeted OpenAI-compatible client (calls/tokens caps, cache, retries, `<think>` stripping, JSON repair + schema validation, auto-disable) and deterministic mock |
| `sandbox` | AST-checked, isolated-process execution of LLM-written feature code |
| `research_loop` | hypothesis → experiment → paired evaluation → keep/discard; Optuna TPE via ask/tell |

## AutoResearch loop

1. **State for the Researcher** (compact JSON, ~2–3k tokens): task summary, columns with
   dictionary descriptions, validation scheme, train/test shift, target profile, champion
   model and params, top features, error analysis (per-class F1 and top confusions, or
   residual bias by target quintile and worst groups), the last 14 journal entries, the
   recipe catalogue, time and LLM budget left.
2. **Hypotheses** (strict JSON, 1–3 per call): `feature_engineering` (library recipe or
   custom `transform(df)` code), `drop_features`, `target_encoding`, `model` (family,
   params, class weights, objective, target transform), `tune` (one Optuna TPE trial).
3. **Engineer**: parameters are whitelisted and clamped; feature code is AST-checked and
   run in the sandbox; the target is never passed to generated code.
4. **Evaluator**: same folds for every experiment; keep a change only if it beats the
   reference overall *and* on most folds or by more than one standard error of the fold
   deltas. Accepted feature changes become the new base feature set.
5. **Ensembler**: every successful experiment joins the pool; the blend is refitted and the
   submission rewritten whenever the OOF score improves. Accepted features are propagated
   to the other families that carry blend weight, when that is cheap.
6. **Fallbacks**: if the LLM is unreachable, slow, out of budget or returns garbage three
   times in a row, the loop continues with the deterministic planner (recipes chosen from
   the data profile + Bayesian tuning).

Every hypothesis and experiment is journaled (`journal.jsonl`, `research_journal.jsonl`,
`llm_log.jsonl`, `summary.json` in the run directory).

## Safeguards

- Valid submission within seconds (constant), then only validated, atomic overwrites.
- `solution.py` guard + emergency writer; process always exits 0 with a file on disk.
- Experiments in subprocesses with hard timeouts and memory guards; a failing experiment
  is logged and skipped.
- Threads from the cgroup CPU quota (not the host core count); memory limit from cgroups.
- Leakage: id-like columns dropped, near-perfect single-feature predictors flagged, target
  encoding strictly out-of-fold, statistics-based recipes (group stats, frequencies, key
  counts) refitted inside every fold, decision rules fitted on OOF only, no test labels,
  no test set for selection. Test features are used only for category vocabularies
  (documented).
- Validation mimics the test: time-split detection → leave-period-out folds; eval mask for
  categories that a fold sees as new but the real test does not.
- LLM: only `OPENAI_BASE_URL`, per-run caps (30 calls, 160k tokens by default), heuristics
  win over the LLM on task facts. Thinking is switched off per request
  (`reasoning_effort: "none"` on the organizers' proxy); `OPENAI_MODEL` is optional.

## Running locally

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements_drive_folder.txt pytest
python -m pytest -q tests                                     # unit tests
python run_local.py --task data/task1_energy/task.md --holdout --time-budget 900 --llm mock
python run_local.py --task data/task2_attacks/task.md --holdout --docker --time-budget 10800
python -m harness.suite --budget 240                          # generalization suite
./make_submission.sh --verify                                 # build + verify the ZIP
```

`--holdout` carves a scorable holdout out of `train.csv` (last period for time-split
tasks), runs the agent exactly like the platform (task path argument, cwd outside the
task dir, hard kill at the budget) and scores it against linear and random-forest
baseline proxies. `--docker` runs it inside the offline replica in the platform's
layout (`docker/Dockerfile`: Python 3.12 + the organizers' requirements, `--network
none`, `--cpus 8`, non-root; only the ZIP's root files are copied into the task
directory and started there as `python3 solution.py task.md`).

Environment overrides (all optional, platform runs set none): `AGENT_TIME_BUDGET`,
`AGENT_LLM_MODE=real|mock|off`, `AGENT_LLM_MOCK=good|garbage|error|slow|mixed`,
`AGENT_LLM_MAX_CALLS`, `AGENT_LLM_MAX_TOKENS`, `AGENT_RUN_DIR`, `AGENT_MAX_CPUS`.

## Results

Test labels are hidden, so every score here is measured on a holdout carved from
`train.csv` (`run_local.py --holdout`): the last period for task 1, a stratified 25%
for task 2. "Normalized" applies the case's formula against the stronger of two simple
baseline proxies (linear model, random forest) on the same holdout. For task 1 the
`test_like` subset excludes holdout rows whose categories the training part never saw,
because the real test has no such rows (see NOTES.md, "eval mask").

### Public tasks: dress rehearsals

**v0.3 with the live Qwen proxy** (agent code `9a4378b`): local, 6 CPU threads per task
(both tasks in parallel on a 12-thread box), **60-minute budget**, the organizers' LLM.
Both runs exited 0 inside the budget (3 424 s / 3 379 s), peak RSS 0.85 / 0.89 GB,
0 failed experiments, 0 LLM failures.

| Stage | Task 1 RMSE, OOF | Task 1 RMSE, holdout (test-like) | Norm. | Task 2 Macro F1, OOF | Task 2 Macro F1, holdout | Norm. |
| --- | --- | --- | --- | --- | --- | --- |
| a) fallback (quick model) | 45.93 | 36.43 | 9.4 | 0.6335 | 0.6317 | 13.9 |
| b) core pipeline (model zoo) | 43.62 | **34.56** | **14.0** | 0.6426 | 0.6328 | 14.1 |
| c) full agent (research + final) | 40.49 | 36.10 | 10.2 | 0.6462 | **0.6364** | **15.0** |
| Baselines on the same holdout | — | linear 44.69, RF 40.20 | | — | linear 0.418, RF 0.572 | |

LLM use: task 1 7 calls (21.3k + 3.2k tokens), task 2 4 calls (10.2k + 1.5k tokens);
35 / 18 experiments; 3 / 0 hypotheses kept (task 1: Qwen's group statistics by facility
type and building class, the planner's frequency encoding, Qwen's drop of weather-direction
columns).

**Known risk on time splits (task 1).** Research improved leave-one-period-out CV on every
fold, yet the next period got worse: 34.56 after the zoo, 36.10 at the end. Scoring each of
the 35 experiments on that period, the plain XGBoost from the zoo is the best (34.52), and
nearly every change that won on CV made that period worse; the kept group statistics alone
went 34.52 → 39.97 although they won 4 of 4 folds with honest per-fold fitting. A strictly
forward validation (train on earlier periods only) would have rejected the worst variant
but not all of them (NOTES.md). The three live task-1 runs ended at 36.92, 34.78 and 36.10;
the earlier mock-LLM rehearsal below, with different accepted changes, improved the same
period to 33.25. On time-split tasks the research phase is therefore a gamble on the next
period, while the zoo result it starts from is stable across runs. Task 2 (random split)
gains from research.

The earlier rehearsal of the pipeline before the fold refits (commit `cd77a4e`): local,
6 CPU threads per task, mock LLM, 60-minute budget. Every phase ran and the process exited
cleanly inside the budget.

| Stage | Task 1 RMSE, OOF | Task 1 RMSE, holdout (test-like) | Norm. | Task 2 Macro F1, OOF | Task 2 Macro F1, holdout | Norm. |
| --- | --- | --- | --- | --- | --- | --- |
| a) fallback (quick model) | 45.93 | 36.43 | 9.4 | 0.6335 | 0.6317 | 13.9 |
| b) core pipeline (model zoo) | 43.62 | 34.56 | 14.0 | 0.6426 | 0.6328 | 14.1 |
| c) full agent (research + final) | 40.57 | **33.25** | **17.3** | 0.6512 | **0.6408** | **16.0** |
| Baselines on the same holdout | — | linear 44.69, RF 40.20 | | — | linear 0.418, RF 0.572 | |

| Run facts | Task 1 | Task 2 |
| --- | --- | --- |
| Wall time / budget | 3 415 s / 3 600 s | 3 375 s / 3 600 s |
| Peak RSS (agent + workers) | 1.13 GB | 0.92 GB |
| Experiments (ok / failed) | 69 / 0 | 26 / 5 (all handled) |
| Accepted hypotheses | 6 | 2 (incl. ratio features: 0.6437 → 0.6512 OOF) |
| LLM calls (mock) | 30 of 30 | 24 of 30 |

**3-hour runs.** Three full-length attempts were cut short by the dev machine, not by the
agent: Docker Desktop stopped twice (after 23 min and 1 h 44 min) and the host shut down
once (after about 1 h 56 min). Each interrupted run had 0 timeouts and 0 errors, and the
submission left on disk was valid; the 1 h 56 min run scored the same as the table above
(task 1 test-like RMSE 33.25, task 2 macro-F1 0.6408). Peak RSS in the 1 h 44 min Docker
run was 1.23 GB.

On task 1's full holdout (including the 3 191 rows of a state the holdout's training
part never saw) the agent scores RMSE 51.0 vs linear 49.1: trees extrapolate badly to an
unseen state. The real test has no unseen states, which is why the test-like subset is the
estimate for the platform.

### Generalization suite: 14 synthetic unseen tasks (240 s each, mock LLM)

All 14 finished within budget with a valid submission (`python -m harness.suite`; run on
the v0.3 agent code, `9a4378b`, results in `runs/suite_v03`).

| Task | Metric | Agent | Linear | Random forest | Normalized |
| --- | --- | --- | --- | --- | --- |
| 400 features, 3 000 rows | ROC-AUC | 0.9559 | 0.8296 | 0.9079 | 52.1 |
| 20% injected NaNs, renamed columns, time split | RMSLE | 0.5508 | 10.6055 | 0.7653 | 28.0 |
| Datetime + text, time split | MAE | 1.9503 | 2.6199 | 2.6539 | 25.6 |
| Ordinal grades | QWK | 0.4143 | 0.1633 | 0.2421 | 22.7 |
| Probabilities for 5 classes | LogLoss | 0.3506 | 0.4465 | 0.5950 | 21.5 |
| Attack vs normal | LogLoss | 0.0997 | 0.1449 | 0.1108 | 9.9 |
| Anonymised features, integer labels | Weighted F1 | 0.8153 | 0.7508 | 0.8001 | 7.6 |
| Missing values in the target | MAE | 21.1441 | diverged | 21.6414 | 2.3 |
| Two targets in one submission | RMSE | 43.08 / 25.85 | — | — | valid |
| Building class (trivially separable) | ROC-AUC | 1.0000 | 0.9999 | 1.0000 | — |
| Handwritten digits, 1 800 rows | Macro F1 | 0.9677 | 0.9605 | 0.9707 | 0 |
| 442 rows (diabetes) | R² | 0.4243 | 0.4543 | 0.3684 | 0 |
| 569 rows (breast cancer) | F1 | 0.9531 | 0.9612 | 0.9206 | 0 |
| 178 rows (wine) | Accuracy | 1.0000 | 1.0000 | 0.9787 | 0 (tie) |

Small datasets are the known weak spot: on a few hundred rows the agent ties or trails a
plain linear model. Against the v0.1 run every task is within ±0.005 except digits
(0.9735 then): repeated CV for small data (3 × 5 folds under 5 000 rows), added after that
run, makes each model three times as slow, so in the suite's 240 s only the quick model
and LightGBM finish (v0.1 also had ExtraTrees, the best family there). With the
platform's 3 hours every family runs.

### Submission checklist (commit `9a4378b`, `python -m harness.checklist`)

| Check | Result |
| --- | --- |
| Imports ⊆ intersection of both requirements files | pass (`tests/test_imports.py`) |
| No network libraries outside `agent_llm_client.py` | pass |
| Flat ZIP: `solution.py`, `agent_*.py`, `requirements.txt` at the root, no subfolders | pass (23 files, 79 KB) |
| ZIP unzipped and run in the platform layout (files copied into the task dir, `python3 solution.py task.md`) in the offline Docker image | pass (exit 0, 238 s, peak 0.71 GB, valid file) |
| Valid submission with the LLM unreachable (connection refused) | pass (146 s) |
| Valid submission with an LLM endpoint that hangs | pass (client gives up after 2 timeouts; quick model at 74 s) |
| Live runs against the organizers' Qwen proxy | pass (3 × 15 min on task 1, 1 × 15 min on task 2; exit 0, valid files; NOTES.md) |
| Valid submission after hard kills at 459 s and 636 s | pass |
| Offline Docker run on the real task-2 data (`--network none`, 8 CPUs) | pass (211 s, peak RSS 0.85 GB) |
| Peak RAM < 28 GB | pass (1.13 GB max in the rehearsals) |
| Finishes or saves within the budget | pass (57 of 60 min end to end; 3-hour runs interrupted by the host left valid files) |
| Unit tests | 120 passed |

The checklist's ZIP item used to accept any ZIP already in `dist/`; it now requires the
build to succeed (on Windows it also has to call Git's bash, not the WSL launcher).
