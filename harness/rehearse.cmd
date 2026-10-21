@echo off
rem Detached 3-hour dress rehearsal for one task (local, no Docker).
rem   harness\rehearse.cmd <task_dir_name> <out_dir_name> <max_cpus>
rem Started through WMI by the operator so it survives the terminal/app closing.
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" run_local.py --task "data\%1\task.md" --holdout --time-budget 10800 --llm mock --env AGENT_MAX_CPUS=%3 --out "runs\%2" > "runs\%2_harness.txt" 2>&1
echo exit=%ERRORLEVEL% >> "runs\%2_harness.txt"
