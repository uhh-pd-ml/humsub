"""Event-triggered passes: when a chain ends, the submission's next supervisor pass is brought forward.

The supervisor runs a pass every ``supervisor_interval`` seconds; anything that becomes possible in between (a finalize
step, a retry) would wait up to a whole interval.  A chain's end is the event that makes such things possible, and its
worker knows about it at once, so it requests a pass (:func:`request_pass`):

* it leaves a marker, ``<submission>/pass-requested``;
* if it gets the submission's control lock (no supervisor runs right now), it brings the pending supervisor job forward
  to ``now + trigger delay`` with ``scontrol update StartTime=`` -- never later than planned.  The delay (default 60 s,
  ``--supervisor-trigger-delay``) gathers the ends of many chains into one pass;
* if a supervisor holds the lock, it is running a pass: when it queues its successor it finds the marker (newer than
  its pass) and queues the successor after the trigger delay instead of the full interval.

Schedule changes only happen under the control lock, so this never creates a second supervisor.  Everything here is
best effort: if it fails, the supervisor's interval still applies (it is the heartbeat for chains that end without
running any code, e.g. after a node failure).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess
import time
from typing import Any

from .locks import ControlLock
from .submission import load_submission_spec

MARKER = "pass-requested"
DEFAULT_TRIGGER_DELAY = 60


def trigger_delay(opts: dict[str, Any]) -> int:
    """Seconds between a chain's end and the pass it triggers: never more than the supervisor interval."""
    interval = int(opts.get("supervisor_interval", 1800))
    return max(1, min(int(opts.get("supervisor_trigger_delay") or DEFAULT_TRIGGER_DELAY), interval))


def request_pass(spec_path: Path) -> str:
    """Called by a chain's worker when its chain has ended; returns what it did (for the chain's log)."""
    spec = load_submission_spec(spec_path)
    opts = spec.get("submit_options", {})
    if not opts.get("supervisor_job"):
        return "not supervised"
    (spec_path.parent / MARKER).touch()
    lock = ControlLock(spec_path.parent, "pass trigger")
    if not lock.try_acquire():
        holder = lock.holder() or {}
        return f"marker left for the running {holder.get('purpose', 'pass')}"
    try:
        jobs = [str(j) for j in spec.get("supervisor_job_ids", [])]
        if not jobs:
            return "marker left (no supervisor job recorded)"
        return bring_forward(jobs[-1], trigger_delay(opts))
    finally:
        lock.release()


def _job_fields(job_id: str) -> dict[str, str] | None:
    proc = subprocess.run(["scontrol", "-o", "show", "job", job_id], text=True, capture_output=True, timeout=60)
    if proc.returncode != 0:
        return None
    return dict(token.split("=", 1) for token in proc.stdout.split() if "=" in token)


def _epoch(value: str | None) -> float | None:
    try:
        return datetime.strptime(str(value), "%Y-%m-%dT%H:%M:%S").timestamp()   # scontrol prints local time
    except ValueError:
        return None


def bring_forward(job_id: str, delay: int, *, now: float | None = None) -> str:
    """Start the pending job *job_id* in *delay* seconds if it would otherwise start later; never postpone it."""
    fields = _job_fields(job_id)
    if fields is None:
        return f"supervisor job {job_id} is unknown to Slurm (marker left)"
    state = fields.get("JobState", "?")
    if state != "PENDING":
        return f"supervisor job {job_id} is {state} (marker left)"
    now = time.time() if now is None else now
    start = _epoch(fields.get("StartTime"))
    if start is not None and start <= now + delay + 1:
        return f"supervisor job {job_id} starts within {delay} s anyway"
    proc = subprocess.run(["scontrol", "update", f"JobId={job_id}", f"StartTime=now+{delay}"],
                          text=True, capture_output=True, timeout=60)
    if proc.returncode != 0:
        return f"could not bring supervisor job {job_id} forward: {(proc.stderr or proc.stdout).strip()} (marker left)"
    return f"supervisor job {job_id} brought forward to start in {delay} s"


def consume_marker(submission_dir: Path, pass_start: float) -> None:
    """A pass handles every request made before it started."""
    marker = Path(submission_dir) / MARKER
    try:
        if marker.stat().st_mtime < pass_start:
            marker.unlink(missing_ok=True)
    except FileNotFoundError:
        pass


def requested_since(submission_dir: Path, pass_start: float) -> bool:
    """Did a chain end (and request a pass) while this pass ran?"""
    try:
        return (Path(submission_dir) / MARKER).stat().st_mtime >= pass_start
    except FileNotFoundError:
        return False
