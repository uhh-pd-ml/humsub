"""Per-branch scratch directories, payload retry policy and leftover handling.

Two scratch tiers exist for every branch:

* ``HUMSUB_SCRATCH_FAST`` always lives on the SSD (``cache_dir``); small, fast, tiny quota.
* ``HUMSUB_SCRATCH`` is the *bulk* scratch.  It lives on BeeGFS unless ``[execution].scratch = "ssd"``
  (or ``--scratch ssd`` / a manifest hint) selects the SSD, in which case both variables point to
  the same directory.

Both directories are removed when the branch ends, also when it is interrupted by the pre-timeout
signal.  Only a few small text files are kept for diagnosis (:func:`keep_small_files`); anything
left behind by a killed job (SIGKILL, node failure) is swept by ``humsub resume`` / ``humsub gc``
(:func:`stale_scratch_dirs`).
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable

KEEP_MAX_FILE_BYTES = 1 << 20
KEEP_MAX_FILES = 20
RETRY_DURATION_SLACK = 1.2

_SCRATCH_NAME_RE = re.compile(r"^(?P<job>[0-9]+|local)-(?P<branch>[0-9]+)$")


@dataclass(frozen=True)
class ScratchDirs:
    bulk: Path
    fast: Path

    @property
    def all(self) -> list[Path]:
        return [self.fast] if self.bulk == self.fast else [self.bulk, self.fast]


def effective_scratch_kind(config: dict[str, Any], hint: dict[str, Any] | None = None, override: str | None = None) -> str:
    """Precedence: command line > manifest hint > configuration."""
    if override:
        return override
    if hint and hint.get("scratch"):
        return str(hint["scratch"])
    return str(config["execution"]["scratch"])


def scratch_dirs(config: dict[str, Any], submission_id: str, job_id: str, branch_id: int, kind: str) -> ScratchDirs:
    name = f"{job_id}-{branch_id}"
    fast = Path(config["execution"]["cache_dir"]) / "payload-work" / submission_id / name
    if kind == "ssd":
        return ScratchDirs(bulk=fast, fast=fast)
    bulk = Path(config["execution"]["bulk_scratch_dir"]) / submission_id / name
    return ScratchDirs(bulk=bulk, fast=fast)


def keep_small_files(dirs: ScratchDirs, destination: Path) -> list[Path]:
    """Copy a few small text-like files out of the scratch before it is removed."""
    kept: list[Path] = []
    for root in dirs.all:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if len(kept) >= KEEP_MAX_FILES:
                return kept
            try:
                if not path.is_file() or path.is_symlink() or path.stat().st_size > KEEP_MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            if path.suffix.lower() not in {".json", ".log", ".txt", ".out", ".err", ".py", ".cfg"}:
                continue
            target = destination / root.name / path.relative_to(root)
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                kept.append(target)
            except OSError:
                continue
    return kept


def remove_scratch(dirs: ScratchDirs) -> None:
    for path in dirs.all:
        shutil.rmtree(path, ignore_errors=True)


def reset_scratch(dirs: ScratchDirs) -> None:
    remove_scratch(dirs)
    for path in dirs.all:
        path.mkdir(parents=True, exist_ok=True)


def retry_allowed(
    *,
    failed_attempts: int,
    max_retries: int,
    elapsed: float | None,
    usable: float | None,
    attempt_durations: list[float],
    slack: float = RETRY_DURATION_SLACK,
) -> tuple[bool, str]:
    """Decide whether another payload attempt may start.

    A retry is skipped when it cannot finish in the time the hop has left: the previous attempts
    are taken as the best estimate for the duration of the next one (longest one, plus *slack*).
    With equal attempt durations this reproduces the intuition "after 50 % of the walltime the
    second retry is skipped, after 66 % the third", because the n-th retry needs n/(n+1) at most.
    """
    if failed_attempts > max_retries:
        return False, f"retry budget of {max_retries} exhausted"
    if elapsed is None or usable is None or usable <= 0:
        return True, "no time information; retry allowed"
    estimate = (max(attempt_durations) if attempt_durations else 0.0) * slack
    remaining = usable - elapsed
    if estimate > remaining:
        return False, (
            f"retry skipped: next attempt needs ~{estimate:.0f}s (longest previous {max(attempt_durations):.0f}s "
            f"x {slack}) but only {max(remaining, 0):.0f}s of the {usable:.0f}s usable hop time are left"
        )
    return True, f"retry allowed: ~{estimate:.0f}s needed, {remaining:.0f}s left"


def hop_timing(environ: dict[str, str] | None = None, now: Callable[[], float] | None = None) -> tuple[float | None, float | None]:
    """(elapsed seconds in this hop, usable seconds of this hop) from the chain worker's environment."""
    import time
    env = os.environ if environ is None else environ
    clock = time.time if now is None else now
    try:
        start = float(env["HUMSUB_HOP_START"])
        usable = float(env["HUMSUB_HOP_USABLE_SECONDS"])
    except (KeyError, ValueError):
        return None, None
    return max(clock() - start, 0.0), usable


def stale_scratch_dirs(
    roots: list[Path],
    live_jobs: set[str],
    submission_ids: set[str] | None = None,
) -> list[Path]:
    """Scratch directories ``<root>/<submission>/<jobid>-<branch>`` whose Slurm job is not alive."""
    stale: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            if submission_ids is not None and sub.name not in submission_ids:
                continue
            for entry in sorted(sub.iterdir()):
                match = _SCRATCH_NAME_RE.match(entry.name)
                if entry.is_dir() and match and match.group("job") not in live_jobs:
                    stale.append(entry)
    return stale
