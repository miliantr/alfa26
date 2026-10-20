#!/usr/bin/env bash
# Build the platform ZIP: solution.py at the root + the agent/ package, nothing else.
#
#   ./make_submission.sh            -> dist/submission_<git-sha>.zip
#   ./make_submission.sh --verify   -> also unzip into a clean dir and run it on a task
#                                      in the offline Docker image (short budget)
#
# Deliberately excluded: data, runs, tests, harness, any pyproject.toml / uv.lock
# (`uv run solution.py` would try to resolve dependencies, and the container has no
# internet).
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; fi
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; fi

SHA="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
DIRTY="$(git status --porcelain -- solution.py agent 2>/dev/null | head -1)"
[ -n "$DIRTY" ] && SHA="${SHA}-dirty"
OUT="dist/submission_${SHA}.zip"
mkdir -p dist

"$PY" - "$OUT" <<'EOF'
import sys, zipfile, pathlib
out = pathlib.Path(sys.argv[1])
files = [pathlib.Path("solution.py")] + sorted(p for p in pathlib.Path("agent").rglob("*.py"))
for bad in ("pyproject.toml", "uv.lock", "requirements.txt"):
    assert not pathlib.Path("agent", bad).exists(), bad
src = pathlib.Path("solution.py").read_text(encoding="utf-8")
assert "# /// script" not in src, "inline script metadata would make uv resolve deps"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f.as_posix())
with zipfile.ZipFile(out) as z:
    names = z.namelist()
assert "solution.py" in names, "solution.py must be at the ZIP root"
print(f"{out}: {len(names)} files, {out.stat().st_size / 1024:.0f} KB")
EOF

if [ "${1:-}" = "--verify" ]; then
  TASK="${2:-data/task2_attacks/task.md}"
  BUDGET="${3:-300}"
  VERIFY="dist/verify_${SHA}"
  rm -rf "$VERIFY" && mkdir -p "$VERIFY"
  "$PY" -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$OUT" "$VERIFY"
  echo "unzipped to $VERIFY:"; ls "$VERIFY"
  "$PY" run_local.py --task "$TASK" --holdout --docker --solution-dir "$VERIFY" \
      --time-budget "$BUDGET" --llm off --baselines ""
fi
