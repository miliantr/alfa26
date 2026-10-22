"""CPU / memory limits as seen *inside* the container (cgroup-aware).

``os.cpu_count()`` in Docker reports the host's cores, not the ``--cpus`` quota, so we
read cgroup files first to avoid oversubscribing threads.
"""
from __future__ import annotations

import os
from pathlib import Path


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except Exception:
        return None


def cgroup_cpu_limit() -> float | None:
    v2 = _read("/sys/fs/cgroup/cpu.max")
    if v2:
        parts = v2.split()
        if parts and parts[0] != "max":
            try:
                return float(parts[0]) / float(parts[1] if len(parts) > 1 else 100000)
            except ValueError:
                pass
    quota = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    try:
        if quota and period and float(quota) > 0:
            return float(quota) / float(period)
    except ValueError:
        pass
    return None


def detect_cpus(cap: int = 8) -> int:
    counts = [os.cpu_count() or 1]
    try:
        counts.append(len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except Exception:
        pass
    q = cgroup_cpu_limit()
    if q:
        counts.append(max(1, int(q)))
    return max(1, min(min(counts), cap))


def cgroup_memory_limit() -> int | None:
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        v = _read(p)
        if v and v != "max":
            try:
                n = int(v)
                if 0 < n < 1 << 50:  # v1 reports a huge number when unlimited
                    return n
            except ValueError:
                pass
    return None


def detect_memory_bytes() -> int:
    limits = []
    try:
        import psutil

        limits.append(int(psutil.virtual_memory().total))
    except Exception:
        pass
    cg = cgroup_memory_limit()
    if cg:
        limits.append(cg)
    return min(limits) if limits else 8 * 1024 ** 3


def available_memory_bytes() -> int:
    try:
        import psutil

        avail = int(psutil.virtual_memory().available)
    except Exception:
        avail = detect_memory_bytes()
    cg = cgroup_memory_limit()
    if cg:
        used = None
        for p in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
            v = _read(p)
            if v:
                try:
                    used = int(v)
                    break
                except ValueError:
                    pass
        if used is not None:
            avail = min(avail, cg - used)
    return max(0, avail)


def rss_tree(pid: int) -> int:
    """Resident memory of a process and all its children, in bytes."""
    try:
        import psutil

        p = psutil.Process(pid)
        total = p.memory_info().rss
        for c in p.children(recursive=True):
            try:
                total += c.memory_info().rss
            except Exception:
                pass
        return int(total)
    except Exception:
        return 0


def kill_tree(pid: int) -> None:
    try:
        import psutil

        p = psutil.Process(pid)
        procs = p.children(recursive=True) + [p]
        for c in procs:
            try:
                c.kill()
            except Exception:
                pass
        psutil.wait_procs(procs, timeout=10)
    except Exception:
        pass


def snapshot() -> dict:
    return {
        "cpus": detect_cpus(cap=1024),
        "cgroup_cpu": cgroup_cpu_limit(),
        "mem_total_gb": round(detect_memory_bytes() / 1024 ** 3, 2),
        "mem_available_gb": round(available_memory_bytes() / 1024 ** 3, 2),
    }
