"""Global deadline manager.

The clock starts at *process start* (``psutil`` create time), because the platform counts
the 3 hours from launch, including interpreter start-up and imports.
"""
from __future__ import annotations

import time


def _process_start_time() -> float:
    try:
        import psutil

        t = psutil.Process().create_time()
        now = time.time()
        if 0 <= now - t < 3600:  # sanity: ignore clocks that make no sense
            return t
    except Exception:
        pass
    return time.time()


class Deadline:
    """Time budget split into research / final-assembly / safety phases.

    ``research_end``: no new experiments are *started* after it.
    ``final_end``: ensembling and the final write must be done by then.
    The remaining safety margin absorbs slow I/O and process teardown.
    """

    def __init__(self, total_seconds: float, start_time: float | None = None):
        self.total = float(total_seconds)
        self.start = _process_start_time() if start_time is None else float(start_time)
        self.safety = max(30.0, min(300.0, 0.03 * self.total))
        self.final_reserve = max(60.0, min(900.0, 0.07 * self.total))

    # -- clock ---------------------------------------------------------------------
    def now(self) -> float:
        return time.time()

    def elapsed(self) -> float:
        return self.now() - self.start

    def remaining(self) -> float:
        """Seconds until the hard end of the budget (the kill)."""
        return self.total - self.elapsed()

    def fraction_used(self) -> float:
        return self.elapsed() / self.total if self.total > 0 else 1.0

    # -- phases --------------------------------------------------------------------
    @property
    def final_end_at(self) -> float:
        return self.start + self.total - self.safety

    @property
    def research_end_at(self) -> float:
        return self.final_end_at - self.final_reserve

    def research_remaining(self) -> float:
        return self.research_end_at - self.now()

    def final_remaining(self) -> float:
        return self.final_end_at - self.now()

    def can_start(self, expected_seconds: float, phase: str = "research") -> bool:
        """True if a job of ``expected_seconds`` would finish inside ``phase``."""
        left = self.research_remaining() if phase == "research" else self.final_remaining()
        return expected_seconds <= left

    def job_timeout(self, expected_seconds: float, factor: float = 3.0, phase: str = "research",
                    minimum: float = 20.0) -> float:
        """Hard timeout for a job: generous vs its estimate, never past the phase end."""
        left = self.research_remaining() if phase == "research" else self.final_remaining()
        return max(0.0, min(max(minimum, expected_seconds * factor), left))

    def status(self) -> dict:
        return {
            "elapsed_s": round(self.elapsed(), 1),
            "remaining_s": round(self.remaining(), 1),
            "research_remaining_s": round(self.research_remaining(), 1),
            "final_remaining_s": round(self.final_remaining(), 1),
            "total_s": self.total,
        }
