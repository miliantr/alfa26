# Notes and assumptions

Running log of ambiguities and the safe choice taken for each. Newest at the bottom of
each section.

## Workspace / inputs

- **Inputs fetched from the case links.** The working folder only had the PDF. With the
  owner's OK, `baseline.py`, both `requirements.txt` variants and the task data were
  downloaded from the Google Drive links in the PDF (pp. 6, 9) on 2026-10-20.
- **Local data layout mirrors the platform**, not a flat `data/task1_energy/train.csv`:
  `task.md` references `regression/participants/*.csv` and
  `multiclass/participants/*.csv` relative to itself, so the local copy is
  `data/task1_energy/task.md` + `data/task1_energy/regression/participants/*.csv` (same
  for task 2). That keeps the shipped `task.md` valid without edits.
- **Two `requirements.txt` variants** (PDF link vs Drive folder) differ in `requests`,
  `aiohttp` and `langchain-openai`. Safe choice: import only the intersection. The PDF
  variant is kept as `requirements.txt`, the folder one as
  `requirements_drive_folder.txt`.
- **`mcp` is not in either requirements file** although `baseline.py` imports it. We do
  not use it. The local baseline proxy reproduces "the simplest ML model" without the
  MCP/LLM loop.
- **No git remote yet.** The owner chose to commit locally for now; pushes start once an
  `origin` URL is provided.

## Dev machine vs platform

- Dev host: i5-11400H, 6 cores / 12 threads, 15.7 GB RAM, Windows 10, Docker Desktop
  installed (daemon off by default). The platform has 8 CPU / 32 GB, so Docker runs here
  use `--cpus=8` and the largest memory limit the host allows. Peak RAM on the platform is
  extrapolated from local measurements rather than tested at 32 GB.

## Data observations that drive design (found by profiling, not hardcoded)

- Task 1 is a **temporal split**: train `Year_Factor` ∈ {1..5}, test = {6}. Random K-fold
  would overstate quality. Generic rule: if an ordinal numeric column has all test values
  beyond the train range, validate by grouping on that column (leave-period-out).
- Task 1 target is right-skewed (skew 4.8, max 993) → try a log1p target transform, but
  always score in original space.
- Task 2: 38% duplicated feature rows in train, some with conflicting labels; unseen
  `state` values in test; test class priors differ from train (official UNSW-NB15 split).

## Pipeline decisions (M4/M5)

- **Category vocabulary uses train ∪ test values** (no labels involved) so that category
  codes are identical across train/test and unseen test values get their own code instead
  of colliding. All fitted statistics (frequency/group/target encodings) use train only.
- **Experiments run in subprocess workers** with a hard timeout and an RSS guard; the
  main process never trains models, so a hung or exploding fit cannot kill the run or
  block the submission writes.
- **Final test predictions = average of the K fold models** (validated); no unvalidated
  full refit.
- **Decision rules (class weights, thresholds, clipping) are fitted on OOF** and scored
  with the same rule on every fold.
- **Eval mask (validation mimics test).** Found on task 1: `State_11` exists only in
  `Year_Factor`=5. With leave-one-year-out CV its fold scores measure extrapolation to a
  state never seen, while the real test (year 6) has no unseen states. Rule: for
  low-cardinality categoricals, if the real test has ≤0.5% unseen values but a CV fold
  has unseen values, those fold rows still get OOF predictions but are not scored.
- **Local holdout for task 1 is harsher than the platform**: holding out year 5 makes
  `State_11` (3 191 rows) unseen, where tree models extrapolate badly (RMSE 135 there).
  The harness therefore also reports a `test_like` score on holdout rows whose
  categories were seen in the holdout's train part (only for columns on which the real
  test has no unseen values). That is the estimate used for platform-like conditions.
- **Group-time folds**: ≤6 periods → leave-one-period-out; more → k folds of whole
  periods.

## LLM / research decisions (M6/M7)

- **No real-LLM credentials yet.** Everything LLM-related was developed and tested with the
  deterministic mock (`AGENT_LLM_MODE=mock`, behaviours good/garbage/error/slow/mixed).
  The real endpoint path (OpenAI SDK, `extra_body` for Qwen non-thinking mode with
  automatic fallback on HTTP 400, `/no_think` suffix, `<think>` stripping) still needs one
  live smoke test with the team key.
- **LLM budget per task run**: 30 calls / 160k tokens by default; ~1 call for task
  understanding + 1 call per 3 hypotheses. After 3 consecutive failures the client turns
  itself off and the deterministic planner continues.
- **Heuristics beat the LLM on task facts** (type, target, metric). The LLM only fills gaps
  (e.g. no metric found in the text) and contributes domain notes / time-column hints.
- **Champion = proxy model for feature hypotheses**: LightGBM unless another family is
  significantly (>2%) better at <4x the cost (task 1 picks XGBoost).
- **Acceptance rule**: better overall AND (better on ≥60% of folds OR mean fold gain > 1 SE).
- **Propagation** of accepted features to other families only for families that carry
  blend weight and only if the run costs <25% of the research time left (a 4-minute
  CatBoost re-run late in a short run was not worth it).
- **Train/test covariate shift on task 2** (adversarial AUC 0.89, ct_* counters drift).
  Tested importance weighting `p(test|x)/p(train|x)` for shift-aware CV: it does NOT
  recover the official test class mix (Generic 0.23→0.13 vs 0.23 true, Fuzzers 0.10→0.16
  vs 0.07; ESS 29%), so CV is not re-weighted. Plain stratified CV is kept.

## Generalization suite findings (M9)

- 14 synthetic unseen tasks (regression / binary / multiclass / multi-target; MAE, RMSLE,
  R², ROC-AUC, LogLoss, F1, weighted/macro F1, accuracy, QWK; renamed columns, no `unit`,
  no dictionary, injected missingness, missing targets, integer labels, wide 400-column,
  tiny 178-row, datetime + text with time split): **14/14 valid, all within budget**.
- Tiny datasets exposed optimistic model selection: OOF-fitted decision rules (class
  weights, thresholds) and many tuning trials on 5 noisy folds. Fixes: decision rules are
  now **cross-fitted** (each fold's rule fitted on the other folds) for scoring; **repeated
  CV** (3x for n<5k, 2x for n<20k) with averaged OOF/test predictions; tuning ranges bounded
  by training size; stricter acceptance (≥60% fold wins AND mean gain > 0.5 SE).
- Residual risk: on very small data (a few hundred rows) the agent can still pick a tuned
  GBDT over a linear model when CV prefers it; on the 133-row diabetes holdout linear
  was better by 0.04 R² (within holdout noise, CV disagrees).
