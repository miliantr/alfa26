# Replica of the platform runtime: CPython 3.12 + the organizers' requirements, CPU only.
# Run it with the network disabled and the platform limits, in the platform's layout
# (the ZIP's root files copied into the task directory), e.g.
#   docker run --rm --network none --cpus=8 --memory=32g -v "$PWD/runs/x/task:/task" -w /task \
#     alfa-automl-platform:py312 timeout 10800 python3 solution.py task.md
# run_local.py --docker does this with docker create + cp (no bind mounts).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp: OpenMP runtime for LightGBM / XGBoost / CatBoost wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements_drive_folder.txt /tmp/req/

# Install the union of both requirement variants (the agent imports only their
# intersection; tests enforce that). The CPU-only torch index keeps the image small.
RUN cat /tmp/req/requirements.txt /tmp/req/requirements_drive_folder.txt \
      | grep -vE '^\s*(#|$)' | sort -u > /tmp/req/all.txt \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/req/all.txt \
    && python -c "import lightgbm, catboost, xgboost, sklearn, optuna, torch, openai; print('ok')"

RUN useradd --create-home runner
USER runner
WORKDIR /tmp
CMD ["python", "--version"]
