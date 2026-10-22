# Platform contract

Source of truth: `baseline.py` (organizers), the two `task.md` files shipped with the
public tasks, the case PDF ("Альфа Хак_AutoML") and the organizers' key and upload
instructions ("AutoML_Инструкция по ключам", linked from the API-key email; cited as
"key instructions"). Everything in the agent must stay
consistent with this file. Facts are marked with their source; anything we inferred is
marked **[assumption]** and is also listed in `NOTES.md`.

## 1. Launch interface

| Item | Value | Source |
|---|---|---|
| Entry point | `solution.py` at the **root** of the ZIP; other modules may sit next to it | PDF p.6 |
| ZIP layout | **Only the root-level `.py` files and `requirements.txt` are copied** into the task directory. Subfolders, data, `.whl` and config files are dropped; a module in a subfolder is missing at run time ("may score 0"). Hence our flat `agent_*.py` modules | key instructions |
| Command | `python3 solution.py task.md`, run inside the task directory (the solution files sit next to `task.md`). The baseline docstring shows `uv run baseline.py task.md` with the three env vars | key instructions, `baseline.py` docstring |
| Positional arg | `sys.argv[1]` = path to the task Markdown file; default `task.md` (relative to cwd) | `baseline.py` `__main__` |
| Path resolution | Relative paths are relative to the **directory of the task Markdown file** (`task_path.resolve().parent`); the baseline's tool subprocess runs with `cwd = task_path.parent` | `baseline.py` (`run_python` doc, `StdioServerParameters(cwd=...)`), `task.md` ("Пути указаны относительно каталога этого Markdown-файла") |
| Working dir | The task directory. We still resolve everything from `task.md`'s location, never from cwd, so other layouts work too | key instructions |
| Scratch files | Allowed in the task directory, its subdirectories and `/tmp` | key instructions |
| `--mcp-server` | Internal self-invocation of the baseline only; the platform does not pass it | `baseline.py` |
| Python | 3.12, Docker container | PDF p.6–7 |
| Dependencies | Installed by the platform from the ZIP's root `requirements.txt` (the provided list; no extra packages or `.whl`); no installs at runtime | PDF p.7, key instructions |

## 2. Environment variables

| Var | Meaning | How we use it |
|---|---|---|
| `OPENAI_API_KEY` | key for the organizers' LLM proxy, set by the platform | read implicitly by `openai.OpenAI()`; never logged; a missing key disables the LLM |
| `OPENAI_BASE_URL` | the organizers' OpenAI-compatible proxy, set by the platform; writing our own address or key into the code is forbidden | read implicitly by `openai.OpenAI()`; the **only** network destination we ever contact; a missing value disables the LLM (the SDK would otherwise default to api.openai.com) |
| `OPENAI_MODEL` | model name; optional: "the proxy substitutes the task's model" | passed as `model=`; when unset we send the documented `qwen/qwen3.8-27b` (the baseline would raise a KeyError) |

No other platform variables exist. Our own dev-only overrides use the `AGENT_` prefix
(e.g. `AGENT_TIME_BUDGET`, `AGENT_LLM_MODE=mock|off|real`, `AGENT_RUN_DIR`) and all
default to platform-safe behaviour when unset.

## 3. Inputs

Observed public layout (paths are written inside `task.md`):

```
<task_dir>/task.md
<task_dir>/regression/participants/{train,test,data_dictionary,sample_submission}.csv   # task 1
<task_dir>/multiclass/participants/{train,test,data_dictionary,sample_submission}.csv   # task 2
```

- `task.md`: Russian Markdown — title, problem statement, **data paths relative to the
  md file**, field descriptions, target, **metric**, and **output format** section.
- Each task always ships `train.csv`, `test.csv`, `data_dictionary.csv`,
  `sample_submission.csv` (PDF p.9). The sub-folder (`regression/`, `multiclass/`) varies,
  so the agent parses paths from `task.md` and falls back to a recursive search under
  `<task_dir>`.
- CSV: UTF-8, `,` separator, `.` decimal, empty cell = missing (PDF p.9, task.md).
- `data_dictionary.csv` columns: `feature`, `role` (`id` / `feature` / `target`),
  `dtype` (`numeric` / `category` / …), `unit` (**may be absent** — absent in task 2),
  `description_ru`.
- `sample_submission.csv` values are placeholders, not answers (zeros for task 1,
  `Normal` for task 2).

Public tasks:

| | Task 1 `site_eui` | Task 2 `attack_cat` |
|---|---|---|
| Type | regression | multiclass (10 string labels) |
| Metric | RMSE, original scale | Macro F1 (`f1_score(average="macro")`); Accuracy and Weighted F1 are reference only |
| train / test rows | 53 308 / 22 449 | 87 670 / 41 166 |
| id column | `id` (last column in train) | `id` (first column) |
| Split | **temporal**: train `Year_Factor` 1–5, test `Year_Factor` 6 | train = 50% of official UNSW-NB15 train; test = stratified 50% of official hold-out (class priors differ) |

## 4. Output

From the "Формат результата" section of both `task.md` files:

- File: **`submission.csv` in the directory of the task Markdown file**.
- UTF-8, `,` separator, **no index**, **exactly** the sample-submission columns
  (`id,<target>`), header included.
- One row per `id` of `test.csv`, **same order as `test.csv`**, no duplicate ids.
- Regression: one finite number per row (no NaN / inf).
- Classification: one string label per row, from the set of allowed classes.
- Only the file present on disk when the process ends (or is killed at 3 h) is scored
  (PDF p.7, p.17).

Agent rules derived from this:
1. Columns, order and ids are copied from `sample_submission.csv` / `test.csv`; values are
   validated (finite, dtype, label vocabulary) before every write.
2. Every write is atomic (`tmp` + `os.replace`) in the same directory.
3. If `task.md` names a different output file, write that one **and** `submission.csv`.

## 5. Exit behaviour

- Baseline: prints the LLM's final message and returns (exit 0); raises
  `RuntimeError` (exit 1) after 100 agent steps. Nothing indicates that the exit code is
  scored — the file is.
- Ours: top-level `try/except` around everything; always exit 0 once a validated
  submission exists; if something fails before that, write the constant fallback, then
  exit 0.

## 6. Resources and rules

- 8 CPU, 32 GB RAM, no GPU (PDF p.7). Agent targets peak RSS < 28 GB and reads its real
  limits (cgroup / psutil) at start-up.
- 3 h wall clock per task, counted from process start; killed afterwards (PDF p.7, p.17).
- No internet from the container; no pretrained models; train only on the task data.
- LLM: only Qwen3.8-27B via the organizers' API (env vars above). 600 RUB/day per key,
  shared by everyone using it; once spent, the proxy answers "Budget has been exceeded"
  until 03:00 MSK → per-run caps on calls and tokens, and the client disables itself on
  that error (key instructions).
- ZIP up to 100 MB (key instructions; ours is under 100 KB).
- Each task runs in isolation; nothing persists between tasks; no feedback from the
  hidden test.
- 3 submissions per team per day (reset 00:00 MSK).

## 7. LLM endpoint

- OpenAI-compatible Chat Completions (`client.chat.completions.create`). The baseline
  uses tool calling with `parallel_tool_calls=False`; we use plain chat completions with
  a JSON-only reply and strict validation, which every OpenAI-compatible server supports.
- Measured on the organizers' proxy (2026-10-22, model `qwen/qwen3.8-27b`): thinking is
  on by default and its text arrives in a separate `reasoning_content` field; `/no_think`
  and `chat_template_kwargs` are ignored; `reasoning_effort: "none"` switches it off
  (0 reasoning tokens, about 2× faster). The client sends `reasoning_effort` together
  with `chat_template_kwargs` (for a bare vLLM server), drops `reasoning_effort` on an
  HTTP 400, then both. A reply whose reasoning hit `max_tokens` is never parsed as the
  answer (one retry with a 3× budget); `<think>…</think>` in the content is stripped.

## 8. Libraries

Two `requirements.txt` variants exist:

- `requirements.txt` — linked from the PDF (has `requests`, `aiohttp`; no `langchain-openai`).
- `requirements_drive_folder.txt` — in the data folder on Drive (has `langchain-openai`;
  no `requests`, `aiohttp`).

**Rule: the agent imports only packages present in both files** (enforced by a test).
`mcp`, which `baseline.py` imports, is in neither and must not be used.

Relevant for us (identical pins in both files):

| Purpose | Packages |
|---|---|
| Data | numpy 1.26.4, pandas 2.2.3, scipy 1.12.0, pyarrow 19.0.1, polars 1.26.0, joblib 1.4.2, threadpoolctl 3.6.0 |
| GBDT | lightgbm 4.6.0, catboost 1.2.8, xgboost 2.1.4 |
| Classic ML | scikit-learn 1.5.2, imbalanced-learn 0.12.4, category-encoders 2.6.4, feature-engine 1.8.3, statsmodels 0.14.0 |
| Tuning | optuna 4.2.1 |
| NN | torch 2.5.1 (CPU) |
| AutoML | autogluon.tabular[lightgbm,catboost,xgboost] 1.3.1, lightautoml 0.4.2, flaml[automl] 2.3.6 |
| LLM | openai 1.68.2, httpx 0.28.1, tenacity 9.0.0 (langchain / langgraph available, not needed) |
| Utils | pydantic 2.10.6, jsonschema 4.23.0, orjson 3.10.15, PyYAML 6.0.2, psutil 6.1.1, cloudpickle 3.1.1, tqdm 4.67.1 |

Verified locally: the full set resolves and imports on CPython 3.12.14.
