"""Run settings. Platform runs set no AGENT_* variables, so defaults are platform values."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    # Wall-clock budget per task, counted from process start (platform: 3 h).
    time_budget: float = 10800.0
    # "real" uses OPENAI_* env vars; "mock" is a deterministic offline stub; "off" disables.
    llm_mode: str = "real"
    run_dir: str | None = None
    seed: int = 42
    n_folds: int = 5
    # LLM budget per task run.
    llm_max_calls: int = 30
    llm_max_tokens: int = 160_000
    llm_timeout: float = 120.0
    # Fraction of the detected memory limit the agent allows its workers to use.
    mem_fraction: float = 0.80
    max_cpus: int = 8
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def load_settings() -> Settings:
    s = Settings()
    s.time_budget = _env_float("AGENT_TIME_BUDGET", s.time_budget)
    mode = os.environ.get("AGENT_LLM_MODE", "").strip().lower()
    if mode in ("real", "mock", "off"):
        s.llm_mode = mode
    elif not (os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_BASE_URL")):
        # The platform sets the key and the proxy address; OPENAI_MODEL is optional.
        s.llm_mode = "off"
    s.run_dir = os.environ.get("AGENT_RUN_DIR") or None
    s.seed = _env_int("AGENT_SEED", s.seed)
    s.n_folds = max(2, _env_int("AGENT_N_FOLDS", s.n_folds))
    s.llm_max_calls = _env_int("AGENT_LLM_MAX_CALLS", s.llm_max_calls)
    s.llm_max_tokens = _env_int("AGENT_LLM_MAX_TOKENS", s.llm_max_tokens)
    s.max_cpus = max(1, _env_int("AGENT_MAX_CPUS", s.max_cpus))
    return s
