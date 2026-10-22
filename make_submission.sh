#!/usr/bin/env bash
# Build the platform ZIP. The platform copies only the ZIP's root-level .py files and
# requirements.txt, so everything ships flat: solution.py, agent_*.py, requirements.txt.
#
#   ./make_submission.sh            -> dist/submission_<git-sha>.zip
#   ./make_submission.sh --verify   -> also unzip into a clean dir and run it on a task
#                                      in the offline Docker image (short budget)
#
# Deliberately excluded: data, runs, tests, harness, run_local.py, baseline.py, any
# pyproject.toml / uv.lock (`uv run solution.py` would try to resolve dependencies).
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; fi
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; fi

SHA="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
DIRTY="$(git status --porcelain -- solution.py "agent_*.py" requirements_drive_folder.txt 2>/dev/null | head -1)"
[ -n "$DIRTY" ] && SHA="${SHA}-dirty"
OUT="dist/submission_${SHA}.zip"
mkdir -p dist

"$PY" - "$OUT" <<'EOF'
import ast, sys, zipfile, pathlib
from harness.ship import local_modules, zip_members
out = pathlib.Path(sys.argv[1])
members = zip_members()
src = pathlib.Path("solution.py").read_text(encoding="utf-8")
assert "# /// script" not in src, "inline script metadata would make uv resolve deps"
local = local_modules()
for name, path in members.items():
    if name.endswith(".py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                assert not node.level, f"relative import in {name}"
                mods = [node.module or ""]
            else:
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else []
            for m in mods:
                top = m.split(".")[0]
                assert not top.startswith("agent") or top in local, f"{name} imports {m}, which is not shipped"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for name, path in members.items():
        z.write(path, name)
with zipfile.ZipFile(out) as z:
    names = z.namelist()
assert "solution.py" in names and "requirements.txt" in names, names
assert all("/" not in n and "\\" not in n for n in names), "the platform ignores files in subfolders"
assert not {"run_local.py", "baseline.py"} & set(names)
size = out.stat().st_size
assert size < 100 * 1024 ** 2, "ZIP limit is 100 MB"
print(f"{out}: {len(names)} files at the root, {size / 1024:.0f} KB")
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
