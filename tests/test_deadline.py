import time

from agent.deadline import Deadline


def test_phases_ordered_and_budget_respected():
    d = Deadline(10800, start_time=time.time())
    assert d.start < d.research_end_at < d.final_end_at < d.start + d.total
    # Research must stop around 2h40m for the 3 h platform budget.
    research_min = (d.research_end_at - d.start) / 60
    assert 150 <= research_min <= 170


def test_short_budget_scales():
    d = Deadline(600, start_time=time.time())
    assert d.research_remaining() > 300
    assert d.final_remaining() > d.research_remaining()


def test_job_timeout_never_past_phase_end():
    d = Deadline(600, start_time=time.time() - 500)  # 100 s left in total
    assert d.job_timeout(1000) <= max(0.0, d.research_remaining()) + 1e-6
    assert not d.can_start(60)


def test_expired_deadline():
    d = Deadline(100, start_time=time.time() - 200)
    assert d.remaining() < 0 and d.job_timeout(10) == 0.0 and not d.can_start(1)
