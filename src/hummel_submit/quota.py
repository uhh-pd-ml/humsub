"""Preflight estimate of the SSD footprint of a submission.

Every running chain job occupies the SSD (``cache_dir``) with an unpacked ``cmsexec``-style runtime
and, when the bulk scratch is placed on the SSD, with its branch scratch.  Submitting many chains
without a concurrency cap can therefore exceed the SSD quota within minutes; this check refuses such
a submission up front.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from .config import ConfigError

_UNITS = {"B": 1, "KIB": 1 << 10, "MIB": 1 << 20, "GIB": 1 << 30, "TIB": 1 << 40}
_QUOTA_RE = re.compile(
    r"^(?P<mount>\S+)\s+\d+\s*%\s*\(\s*(?P<used>[\d.]+)\s*(?P<uu>[A-Za-z]+)\s*/\s*(?P<total>[\d.]+)\s*(?P<tu>[A-Za-z]+)\s*\)"
)
SAFETY_FRACTION = 0.8


def parse_rrz_quota(text: str, path: Path) -> tuple[int, int] | None:
    """(used bytes, quota bytes) of the filesystem line of ``rrz-quota`` that contains *path*."""
    best: tuple[int, int, int] | None = None
    for line in text.splitlines():
        match = _QUOTA_RE.match(line.strip())
        if not match:
            continue
        mount = match.group("mount")
        if not str(path).startswith(mount.rstrip("/") + "/") and str(path) != mount:
            continue
        try:
            used = float(match.group("used")) * _UNITS[match.group("uu").upper()]
            total = float(match.group("total")) * _UNITS[match.group("tu").upper()]
        except (KeyError, ValueError):
            continue
        if best is None or len(mount) > best[0]:
            best = (len(mount), int(used), int(total))
    return None if best is None else (best[1], best[2])


def quota_usage(path: Path) -> tuple[int, int] | None:
    """Best-effort (used, quota) of the filesystem holding *path*; ``None`` if unknown."""
    exe = shutil.which("rrz-quota")
    if exe:
        try:
            out = subprocess.run([exe], check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=60).stdout
            parsed = parse_rrz_quota(out, path)
            if parsed:
                return parsed
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        st = os.statvfs(path if path.exists() else path.parent)
    except OSError:
        return None
    total = st.f_blocks * st.f_frsize
    return (total - st.f_bavail * st.f_frsize, total) if total else None


def concurrent_chains(*, chains: int, max_concurrent: int, wait: bool, parallel_jobs: int) -> int:
    limits = [chains]
    if max_concurrent > 0:
        limits.append(max_concurrent)
    if wait and parallel_jobs > 0:
        limits.append(parallel_jobs)
    return max(1, min(limits))


def projected_fast_bytes(config: dict[str, Any], *, concurrency: int, scratch_kind: str, scratch_bytes_per_branch: int) -> int:
    per_job = int(config["execution"]["fast_bytes_per_job"])
    if scratch_kind == "ssd":
        per_job += scratch_bytes_per_branch
    return concurrency * per_job


def check_fast_quota(
    config: dict[str, Any],
    *,
    branches: int,
    tasks_per_job: int,
    max_concurrent: int,
    wait: bool,
    parallel_jobs: int,
    scratch_kind: str,
    scratch_bytes_per_branch: int = 0,
    usage: tuple[int, int] | None | str = "auto",
) -> str:
    """Raise ConfigError if the projected SSD footprint would not fit; return a one-line summary."""
    chains = math.ceil(branches / max(tasks_per_job, 1))
    concurrency = concurrent_chains(chains=chains, max_concurrent=max_concurrent, wait=wait, parallel_jobs=parallel_jobs)
    projected = projected_fast_bytes(
        config, concurrency=concurrency, scratch_kind=scratch_kind, scratch_bytes_per_branch=scratch_bytes_per_branch
    )
    cache_dir = Path(config["execution"]["cache_dir"])
    if usage == "auto":
        usage = quota_usage(cache_dir)
    gib = 1 << 30
    if not usage:
        return f"SSD footprint: ~{projected / gib:.1f} GiB for up to {concurrency} concurrent chain(s) (quota unknown, not checked)"
    used, total = usage
    free = max(total - used, 0)
    summary = (
        f"SSD footprint: ~{projected / gib:.1f} GiB for up to {concurrency} concurrent chain(s); "
        f"{free / gib:.1f} GiB free of {total / gib:.0f} GiB"
    )
    if projected > SAFETY_FRACTION * free:
        raise ConfigError(
            summary + f" -> exceeds {int(SAFETY_FRACTION * 100)}% of the free SSD quota. Lower the concurrency "
            "(--max-concurrent N, or --wait --parallel-jobs N), put the bulk scratch on BeeGFS (--scratch beegfs), "
            "or override with --ignore-quota-check."
        )
    return summary
