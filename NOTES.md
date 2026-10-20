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
