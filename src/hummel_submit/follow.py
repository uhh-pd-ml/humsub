from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import TextIO

from .chain import query_chain
from .slurm import query_jobs, slurm_log_path
from .state import load_state


_TERMINAL_RAW_PREFIXES = (
    "failed",
    "stopped-",
    "cancelled",
    "cleaned-up",
)


def _raw_terminal(status: str) -> bool:
    return status == "completed" or status.startswith(_TERMINAL_RAW_PREFIXES)


def _tail_start(path: Path, lines: int) -> int:
    """Return a byte offset corresponding approximately to ``tail -n lines``."""
    if lines <= 0:
        return path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(0, 2)
        end = handle.tell()
        if end == 0:
            return 0
        pos = end
        chunks: list[bytes] = []
        newlines = 0
        while pos > 0 and newlines <= lines:
            step = min(8192, pos)
            pos -= step
            handle.seek(pos)
            chunk = handle.read(step)
            chunks.append(chunk)
            newlines += chunk.count(b"\n")
        data = b"".join(reversed(chunks))
        split = data.splitlines(keepends=True)
        keep = split[-lines:] if len(split) > lines else split
        return end - sum(len(part) for part in keep)


def _drain(path: Path, offset: int, out: TextIO) -> int:
    if not path.exists():
        return offset
    try:
        size = path.stat().st_size
    except OSError:
        return offset
    if size < offset:
        offset = 0
    if size == offset:
        return offset
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
        offset = handle.tell()
    if data:
        out.write(data.decode("utf-8", errors="replace"))
        out.flush()
    return offset


def _pick_current_job(state: dict) -> str | None:
    current = state.get("current_job_id")
    if current:
        return str(current)

    jobs = list(map(str, state.get("jobs", [])))
    if not jobs:
        return None

    statuses = query_jobs(jobs)
    for job_id in jobs:
        status = statuses.get(job_id)
        if status and status.running:
            return job_id
    for job_id in jobs:
        status = statuses.get(job_id)
        if status and status.pending:
            return job_id

    # For a terminal chain, the last known job is the useful log to show.
    if _raw_terminal(str(state.get("status", ""))):
        return jobs[-1]
    return None


def follow_chain(
    state_path: Path,
    *,
    initial_lines: int = 10,
    poll_interval: float = 1.0,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> int:
    """Follow the current Slurm log and automatically cross continuation hops.

    The first attached log follows ``tail -n <initial_lines> -f`` semantics.  A
    successor hop is read from its beginning so no startup output is missed.
    The function returns 0 for a successful chain and the chain's non-zero
    failure code (or 1) for a failed chain.
    """
    if initial_lines < 0:
        raise ValueError("initial_lines must be >= 0")
    if poll_interval <= 0:
        raise ValueError("poll_interval must be > 0")

    out = out or sys.stdout
    err = err or sys.stderr

    active_job: str | None = None
    active_log: Path | None = None
    offset = 0
    first_attachment = True
    terminal_stable_polls = 0
    last_terminal_size: int | None = None

    while True:
        state = load_state(state_path)
        current_job = _pick_current_job(state)

        if current_job and current_job != active_job:
            if active_log is not None:
                offset = _drain(active_log, offset, out)
            active_job = current_job
            active_log = slurm_log_path(state, current_job)
            if first_attachment and active_log.exists():
                offset = _tail_start(active_log, initial_lines)
            else:
                offset = 0
            first_attachment = False
            print(f"[follow] job {current_job}: {active_log}", file=err, flush=True)

        if active_log is not None:
            offset = _drain(active_log, offset, out)

        raw_status = str(state.get("status", ""))
        if _raw_terminal(raw_status):
            # The chain state can become terminal a moment before the Slurm
            # wrapper has emitted its final lines.  Wait until the current job
            # is no longer active and the log size has been stable twice.
            active_now = False
            if active_job:
                status = query_jobs([active_job]).get(active_job)
                active_now = bool(status and (status.running or status.pending))

            if not active_now:
                size = -1
                if active_log is not None and active_log.exists():
                    try:
                        size = active_log.stat().st_size
                    except OSError:
                        size = -1
                if size == last_terminal_size:
                    terminal_stable_polls += 1
                else:
                    terminal_stable_polls = 0
                    last_terminal_size = size
                if terminal_stable_polls >= 2:
                    if active_log is not None:
                        offset = _drain(active_log, offset, out)
                    effective = query_chain(Path(state["output_dir"]), state["chain_id"])
                    if effective.state == "finished":
                        return 0
                    code = effective.code or int(state.get("exit_code") or 1)
                    return code if 1 <= code <= 255 else 1
        else:
            terminal_stable_polls = 0
            last_terminal_size = None

        time.sleep(poll_interval)
