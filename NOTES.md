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
- **Label-shift adaptation rejected (task 2).** EM prior re-estimation (Saerens 2002) on
  test predictions moves priors only halfway to the official hold-out mix (TV 0.131 ->
  0.084) and pushes the rare classes the wrong way (Analysis/Backdoor/Shellcode/Worms/
  Fuzzers up, while they are rarer in the test). Adapting macro-F1 decisions to those
  priors would hurt rare-class precision, so decisions stay fitted on train OOF.

## Checklist run (2026-10-20, commit 818677e)

imports ⊆ requirements ✅ · no network libs outside llm_client ✅ · ZIP root has
solution.py (23 files) ✅ · valid submission with the LLM unreachable ✅ (146 s) · valid
submission after hard kills at 459 s and 636 s ✅ · offline Docker run on the real task-2
data (network none, 8 CPUs) ✅ (peak RSS 0.81 GB).

## Dress rehearsals (M10)

- **Docker Desktop did not survive two 3-hour attempts.** Both times the daemon went away
  (`error waiting for container: unexpected EOF`) when the Claude desktop app was quit:
  attempt 1 after 23 min, attempt 2 after 1 h 44 min (197 experiments, 0 timeouts,
  0 errors, peak 1.23 GB, OOF RMSE 45.93 -> 40.91 before it died).
- **Root cause of the fragility:** the Claude app is an MSIX-packaged app. Its writes to
  `%APPDATA%` are redirected into `AppData\Local\Packages\Claude_*\LocalCache\Roaming`, so
  the uv-managed Python (and the venv that points to it) was invisible outside the app,
  and processes started from the app die when it quits.
- **Fix used:** `.venv\pyvenv.cfg` `home` now points to the interpreter's real path under
  `LocalCache\Roaming\uv\python\...` (backup in `pyvenv.cfg.bak`), and the rehearsals run
  as WMI-created processes (`harness\rehearse.cmd`), detached from the app.
- **Rehearsal setting:** both tasks in parallel, local (no Docker), 6 CPU threads each
  (12 logical threads on the dev box), 3-hour budget, mock LLM. 6 threads instead of the
  platform's 8 makes it a conservative throughput estimate. The no-network property is
  covered by the Docker runs (checklist item 2b and the ZIP verification).
- **Likely reason Docker died:** Docker Desktop was itself started from inside the app
  (so it shared the app's process container) and went down with it.
- **Completed end-to-end rehearsal (commit cd77a4e, 60-minute budget, both tasks):**
  exit 0 in 3 415 s / 3 375 s, peak RSS 1.13 / 0.92 GB, task 1 test-like RMSE 33.25 (vs RF
  40.20, norm. 17.3), task 2 macro-F1 0.6408 (vs RF 0.5724, norm. 16.0). The interrupted
  3-hour run (host shutdown at ~1 h 56 min) left a valid file with the same scores.
- **Fixes found by the rehearsals:** median-based time estimates and a 5x timeout for
  tuning trials (6 of ~44 trials had timed out); first run of a family gets 5x; blend
  selection tries plain and class-balanced log loss (the balanced proxy alone locked onto
  the class-weighted model on task 2); research jobs capped at 25% of the window left;
  tune/model hypotheses skipped for families without blend weight that cost >10% of it.
- **Known efficiency item (not fixed):** CatBoost's cost estimate (3x LightGBM) is far
  below reality on these tasks (9x on task 1, 16x on 10-class task 2: 18 min per run on 6
  threads for macro-F1 0.596). It runs last in the zoo, so it only costs research time
  (~8% of the 3-hour window on task 2), never the floor.

## LLM path hardening (after v0.1)

- **Hanging endpoint** (accepts connections, never answers) is a more likely failure in a
  locked-down container than a refused connection. Before: every request retried 3x with
  backoff, three failed calls needed to disable the client, the start-up parse call had a
  90 s timeout -> ~17 min of research lost and the first real model delayed ~4.6 min.
  Now: timeouts are not retried, two timeouts disable the client, the parse call has 45 s
  -> ~165 s lost; quick model on disk at 74 s in an end-to-end run (task 2).
- **Real OpenAI-SDK path tested without credentials** against a local OpenAI-compatible
  stub: think-block stripping, fenced JSON, HTTP 400 on `chat_template_kwargs` -> retry
  without it, `/no_think` suffix, usage accounting (unit test), plus a 15-minute agent run in
  real mode on task 1 (task spec merged, sandboxed feature code executed, a recipe and a
  parameter change accepted, the bogus parameter dropped, garbage replies tolerated).
- **Prompt size**: research prompts were ~11k characters (~4-5k tokens: every column with
  its Russian description). Now full detail for the 20 most important columns only:
  ~6.8k characters, ~2.8k tokens per call, ~85k tokens per task at the 30-call cap.

## Organizers' key instructions and the live Qwen proxy (2026-10-22/23)

The API-key email links "AutoML_Инструкция по ключам" (see CONTRACT.md). Three findings
would each have cost points on the platform; v0.1 and v0.2 must not be submitted.

- **ZIP layout (would have scored 0):** "only the root-level `.py` files and
  `requirements.txt` are copied" into the task directory. Our code lived in an `agent/`
  package, which the platform would have dropped. The agent is now flat (`agent_*.py`,
  absolute imports, worker started as `python -m agent_worker`); `harness/ship.py`
  defines the ZIP, `make_submission.sh` refuses subfolders, relative imports and
  `agent_*` imports that are not shipped, and adds the Drive-folder `requirements.txt`
  (the variant in `problem/`, the one the instructions' `langchain-openai` mention
  matches). The Docker rehearsal now uses the platform layout: root files copied into
  the task directory, `python3 solution.py task.md` started there.
- **`OPENAI_MODEL` is optional** ("the proxy substitutes the task's model"). Both the
  settings and the client used to switch the LLM off without it. Now the key and the base
  URL decide; the model defaults to the documented `qwen/qwen3.8-27b` (verified live with
  the variable unset). A missing base URL still disables the LLM, so the SDK never falls
  back to api.openai.com.
- **Thinking mode:** the proxy returns Qwen's reasoning in `reasoning_content` and
  ignores `/no_think` and `chat_template_kwargs`. The first live task-parse call spent its
  800 tokens reasoning; the client then took that text as the answer, pulled `["mae"]` out
  of it and failed validation (18.5 s). `reasoning_effort: "none"` gives 0 reasoning
  tokens: the same call now takes 2.9 s / 183 tokens and returns the correct spec.
  Fallback order on HTTP 400: `reasoning_effort` + `chat_template_kwargs` ->
  `chat_template_kwargs` (bare vLLM) -> nothing. A reply cut off while reasoning is never
  parsed (one retry with a 3x budget); "Budget has been exceeded" disables the client at once.
- **Live end-to-end runs** (15 min, holdout, real proxy):
  - task 2: exit 0, 832 s, peak 0.89 GB, macro-F1 0.6328. One LLM call (task parse,
    8.4 s, correct metric): at this budget the model zoo fills the window (CatBoost
    alone 424 s), research started with 34 s left.
  - task 1 (run 1): exit 0, 817 s, test-like RMSE 34.24; 6 calls, 14.3k + 4.5k tokens;
    12 Qwen hypotheses evaluated, 1 accepted (drop 3 weak columns, 3/4 folds), 9 rejected
    by the paired rule, 1 skipped for time. Hypothesis calls take 19-27 s (~55 tokens/s).
    Two issues fixed after it: the first reply hit `max_tokens=1500` mid-code, JSON
    repair kept the broken third hypothesis and the sandbox rejected it (now 2500 tokens,
    one-sentence rationales, and the last hypothesis of a cut-off reply is dropped); one
    call hit a 45 s timeout late in the window (now at least 60 s while time allows; in a
    3-hour run it is 120 s anyway).
  - task 1 (run 2, after those fixes): hypothesis calls 2-6 s and 250-440 tokens, no
    cut-offs or timeouts, 3 changes kept on CV (4/4, 4/4, 3/4 folds) - yet test-like
    RMSE 36.92: research made the holdout worse (zoo 34.56 -> research 36.61).
- **Why research hurt in run 2: stateful recipes were fitted on the whole train.**
  Scoring every experiment's own test predictions on the holdout: the kept `group_stats`
  feature (deviations of floor area and heating/cooling degree days from facility-type
  and building-class means) moved CV 44.16 -> 42.86 but the future period 34.52 -> 40.21;
  everything built on it inherited that. Recipe statistics (`group_stats`,
  `freq_encoding`, `make_key` counts) were computed once on all training rows, so in
  leave-one-period-out CV each validation period shaped its own features; the test's rows
  never shape theirs. Rebuilt offline with the same xgb: no feature CV 45.45 / holdout
  34.61; full-train fit 44.11 / 38.90; train+test (transductive) fit 43.88 / 41.79, so
  that is no fix; refitted inside each fold 45.07 / 37.00, with fold deltas
  [+0.54, +1.44, -2.05, +1.62], which the paired rule rejects (mean 0.39 < 0.5 SE = 0.42).
  Fix: `build_features` records each stateful recipe's resolved arguments, and the worker
  refits those columns inside every fold on the fold's training rows (validation rows and
  the fold model's test rows get the fold-train statistics), as target encoding already
  was. Run 1's kept change (a drop) was unaffected, which is why it did not hurt.
- task 1 (run 3, with fold refits): exit 0, 848 s, peak 0.74 GB; 5 calls (4-8 s each,
  14.5k + 2.0k tokens, no failures). Qwen proposed the same `group_stats` feature again:
  now 2/4 folds, rejected. Kept: frequency encoding (refitted per fold) and a drop of 4
  weak columns (4/4). Test-like RMSE: zoo 34.56 -> final 34.78, i.e. research is about
  neutral here (run 2: 36.92). The frequency feature still looks better on CV
  (44.16 -> 43.57) than on the single holdout period (34.52 -> 35.50); with one future
  period that gap is within noise, and the 60-minute mock rehearsal saw research improve
  the same holdout (34.56 -> 33.25).
- A model hypothesis is now compared with the champion when the best same-family run on
  the same features is weaker: in run 2 a log1p LightGBM was "kept" for beating a failed
  tuning trial (49.14) although it was worse than the plain LightGBM on 3 of 4 folds. That
  only affected the journal the Researcher sees: the blend picks members by OOF anyway.

## v0.3 re-validation (2026-10-23)

- **Generalization suite on 9a4378b** (`runs/suite_v03`, mock LLM, 240 s): 14/14 valid
  within budget; every task within ±0.005 of the v0.1 run except digits (macro-F1
  0.9735 -> 0.9677, now just below RF 0.9707). Cause: repeated CV for small data
  (3 x 5 folds under 5 000 rows) was added after the v0.1 suite run, so LightGBM took
  112 s on 1 348 rows and only the quick model and LightGBM fit into 240 s (v0.1 also
  ran ExtraTrees, the best family there). Not an issue at the platform's 3 hours.
- **60-minute rehearsals with the live Qwen proxy** (`runs/v03_live_*`, 6 threads each,
  in parallel): both exit 0 (3 424 s / 3 379 s), peak 0.85 / 0.89 GB, 0 failed
  experiments, 0 LLM failures. Task 1: 7 calls, 24.5k tokens; task 2: 4 calls, 11.7k.
  Holdout: task 1 test-like RMSE 36.10 (zoo 34.56, norm. 10.2); task 2 macro-F1 0.6364
  (zoo 0.6328, norm. 15.0). Quick and zoo phases are identical to every earlier run.
- **Research and the next period on task 1.** Qwen proposed group statistics again, this
  time over floor_area, energy_star_rating, year_built and heating/cooling degree days;
  with honest per-fold fitting they won 4/4 leave-one-period-out folds (CV 44.16 ->
  42.50) and made the holdout period 34.52 -> 39.97. Scoring all 35 experiments of the
  run on that period: the zoo's plain XGBoost is the best (34.52); nearly every change
  that won on CV lost on it (drops, tuning, other recipes alike). Offline, same xgb,
  per-fold fitting, cats = facility_type + building_class:

  | nums | LOPO CV | folds better | holdout |
  | --- | --- | --- | --- |
  | none | 45.45 | - | 34.61 |
  | floor_area | 44.38 | 4/4 | 35.87 |
  | energy_star_rating, year_built | 43.99 | 4/4 | 35.92 |
  | heating/cooling degree days | 46.06 | 0/4 (rejected) | 37.17 |
  | all five | 43.29 | 4/4 | 39.52 |

  Forward chaining (train on earlier periods only; folds = periods 2, 3, 4) is more
  conservative but not a fix: base [45.99, 56.66, 42.50]; all five [45.62, 57.81, 40.56]
  (2/3, mean gain 0.39 < 0.5 SE 0.45: rejected); energy_star + year_built
  [45.50, 56.84, 40.67] (2/3, mean gain 0.71: accepted, yet holdout 35.92). The most
  recent training period (4) improved in every variant, so no check built from the
  training years alone flags these features; the holdout period behaves differently.
- **Decision: no further guard for now.** Over the four task-1 runs the final holdout was
  33.25 (mock LLM), 36.92, 34.78 and 36.10 (live) against a stable zoo at 34.56; task 2
  and the suite gain or hold. Adding forward-chaining acceptance or hedging the final
  blend toward the zoo on time splits are the candidate mitigations; both also cost gains
  where research helps, and the evidence is one future period of one task. Recorded as a
  known risk (README, slide 15).
