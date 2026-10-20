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
agent/runner.py                                                     agent/emergency.py
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
agent/pipeline.py ── experiment pool, cost model, paired comparisons, journal
   │                        │
   │ job.json               │ OOF + test predictions (shared folds)
   ▼                        ▼
agent/executor.py ─► python -m agent.worker  (subprocess: hard timeout + RSS guard)
                         features.py (cached feature sets) · models.py · postprocess.py
```

| Module | Role |
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
  encoding strictly out-of-fold, decision rules fitted on OOF only, no test labels, no test
  set for selection. Test features are used only for category vocabularies (documented).
- Validation mimics the test: time-split detection → leave-period-out folds; eval mask for
  categories that a fold sees as new but the real test does not.
- LLM: only `OPENAI_BASE_URL`, per-run caps (30 calls, 160k tokens by default), heuristics
  win over the LLM on task facts.

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
baseline proxies. `--docker` runs the same inside the offline replica
(`docker/Dockerfile`: Python 3.12 + the organizers' requirements, `--network none`,
`--cpus 8`, non-root, read-only code dir).

Environment overrides (all optional, platform runs set none): `AGENT_TIME_BUDGET`,
`AGENT_LLM_MODE=real|mock|off`, `AGENT_LLM_MOCK=good|garbage|error|slow|mixed`,
`AGENT_LLM_MAX_CALLS`, `AGENT_LLM_MAX_TOKENS`, `AGENT_RUN_DIR`, `AGENT_MAX_CPUS`.

## Results

See the results section at the end of this file (filled from the dress rehearsals).
