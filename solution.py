"""Platform entry point.

    OPENAI_API_KEY=... OPENAI_MODEL=... OPENAI_BASE_URL=... python solution.py task.md

Reads the task Markdown file, solves the task autonomously and writes the predictions
file named in it (``submission.csv`` next to ``task.md``). Whatever happens, a valid
submission is on disk before this process exits.
"""
import os
import sys
from pathlib import Path


def _task_path(argv: list[str]) -> Path:
    args = [a for a in argv[1:] if not a.startswith("--")]
    if args:
        return Path(args[0])
    default = Path("task.md")
    if default.exists():
        return default
    mds = sorted(Path.cwd().glob("*.md"))
    return mds[0] if mds else default


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    task = _task_path(sys.argv).resolve()
    try:
        from agent_runner import run

        run(task)
    except BaseException as exc:  # noqa: BLE001 - nothing may escape without a submission
        try:
            from agent_emergency import ensure_submission

            ensure_submission(task, reason=repr(exc))
        except BaseException:
            pass
    else:
        try:
            from agent_emergency import ensure_submission

            ensure_submission(task, reason="post-run check")
        except BaseException:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    # Hard exit: lingering threads or child processes must not keep us alive.
    os._exit(0)


if __name__ == "__main__":
    main()
