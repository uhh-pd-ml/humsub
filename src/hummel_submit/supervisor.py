"""Supervisor job: a tiny Slurm job that drives a submission's law task graph without a login-node process.

Every ``supervisor_interval`` seconds it runs one *pass* (:mod:`hummel_submit.tick`): law polls the workflow once,
resubmits failed jobs within ``retries`` and submits what is pending.  It stops when the top law task is complete,
and gives up when law reports jobs that failed with no retries left.

At most one supervisor of a submission ever runs: it holds the submission's control lock (:mod:`hummel_submit.locks`)
from start to end, queues its successor while holding it, and a supervisor that finds another one holding the lock
ends at once without a successor.  Supervisor jobs can only be queued under that lock.
"""
from __future__ import annotations

from pathlib import Path
import signal
import subprocess
import sys
import time

from .locks import ControlLock, LockBusy, describe, held_here
from .overview import compute_overview, finalize_state, format_overview
from .tick import run_tick
from .trigger import consume_marker, requested_since, trigger_delay
from .slurm import SlurmError, slurm_command
from .state import write_worker_snapshot
from .submission import append_spec_list, load_submission_spec

SUPERVISOR_TIME = "00:30:00"
SUPERVISOR_PURPOSE = "supervisor"
LOCK_WAIT_SECONDS = 20 * 60     # a `humsub tick`/`resume` holding the lock takes minutes; the job has 30
MAX_CHECKS = 2000


def log(message: str) -> None:
    import os
    import time
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [supervisor:{os.environ.get('SLURM_JOB_ID', 'local')}] {message}", flush=True)


def supervisor_command(spec_path: Path, round_no: int, checks: int = 1, delay: int | None = None) -> list[str]:
    spec = load_submission_spec(spec_path)
    opts = spec.get("submit_options", {})
    slurm = spec["config"]["slurm"]
    sdir = spec_path.parent / "supervisor"
    sdir.mkdir(exist_ok=True)
    snapshot = sdir / "hummel-submit-worker.zip"
    package_dir = Path(__file__).resolve().parent
    if not snapshot.exists():
        write_worker_snapshot(package_dir, snapshot)
    script = sdir / "supervisor.sh"
    if not script.exists():
        script.write_text((package_dir / "supervisor.sh").read_text(encoding="utf-8"), encoding="utf-8")
        script.chmod(0o755)
    interval = int(opts.get("supervisor_interval", 1800)) if delay is None else delay
    args = [
        slurm_command("sbatch"),
        "--parsable",
        f"--job-name={slurm['job_name']}-supervisor",
        f"--account={slurm.get('supervisor_account') or slurm['account']}",
        f"--partition={slurm.get('supervisor_partition') or slurm['partition']}",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        f"--time={SUPERVISOR_TIME}",
        "--export=NONE",
        f"--chdir={spec['project_dir']}",
        f"--output={Path(spec['output_dir']) / 'logs'}/%x_%j.log",
        f"--begin=now+{interval}",
    ]
    # Deliberately NOT niced: it is a 1-CPU, 30-minute watchdog.  With the chains' nice penalty it can starve for hours on a busy
    # partition (seen on Hummel-2: pending > 6 h), which defeats its purpose as a safety net.
    args += [str(script), spec["python_executable"], str(snapshot), str(spec_path), f"{round_no}:{checks}"]
    return args


def submit_supervisor(spec_path: Path, round_no: int, checks: int = 1, delay: int | None = None) -> str:
    """Queue a supervisor job; only allowed while holding the submission's control lock, so a supervisor can only ever
    be queued by the running supervisor itself, or by a submit/resume while no supervisor runs."""
    if not held_here(spec_path):
        raise RuntimeError("internal error: a supervisor may only be queued while holding the submission's control lock")
    cmd = supervisor_command(spec_path, round_no, checks, delay)
    try:
        proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SlurmError(f"supervisor sbatch failed: {getattr(exc, 'stderr', '') or exc}") from exc
    job_id = proc.stdout.strip().split(";", 1)[0]
    append_spec_list(spec_path, "supervisor_job_ids", job_id)   # so that `humsub stats` and `humsub prune` can attribute its logs
    return job_id


def decide(*, tick_state: str, checks: int) -> str:
    """'done' | 'wait' | 'give-up' after one pass."""
    if tick_state == "complete":
        return "done"
    if tick_state == "failed":
        return "give-up"
    return "wait" if checks < MAX_CHECKS else "give-up"


def acquire_control(spec_path: Path, *, wait: float = LOCK_WAIT_SECONDS, poll: float = 10) -> ControlLock | None:
    """The submission's control lock for this supervisor, or None if another supervisor holds it (then this one is a
    duplicate and must end without a successor).  A `humsub tick`/`resume` holding it is waited for."""
    lock = ControlLock(spec_path.parent, SUPERVISOR_PURPOSE)
    deadline = time.time() + wait
    announced = False
    while not lock.try_acquire():
        holder = lock.holder() or {}
        if holder.get("purpose") == SUPERVISOR_PURPOSE:
            log(f"another supervisor holds the submission ({describe(holder)}): this one is a duplicate and ends "
                "without queuing a successor")
            return None
        if time.time() >= deadline:
            raise LockBusy(holder)
        if not announced:
            log(f"waiting for the submission: {describe(holder)}")
            announced = True
        time.sleep(poll)
    return lock


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: python -m hummel_submit.supervisor SUBMISSION_JSON ROUND[:CHECKS]", file=sys.stderr)
        return 2
    spec_path = Path(argv[0]).resolve()
    parts = argv[1].split(":")
    round_no = int(parts[0])  # kept in the job arguments for older frozen supervisor scripts; law counts retries now
    checks = int(parts[1]) if len(parts) > 1 else 1
    # Slurm's SIGTERM (time limit, scancel) must unwind through `finally`, which releases the control lock; a lock left
    # behind by SIGKILL or a node failure is released by its staleness (hummel_submit.locks).
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    try:
        lock = acquire_control(spec_path)
    except LockBusy as exc:
        log(f"giving up: {exc} for {LOCK_WAIT_SECONDS // 60} minutes; `humsub resume` queues a supervisor again")
        return 1
    if lock is None:
        return 0
    try:
        return _supervise(spec_path, round_no, checks)
    finally:
        lock.release()


def _supervise(spec_path: Path, round_no: int, checks: int) -> int:
    spec = load_submission_spec(spec_path)
    output_dir = Path(spec["output_dir"])
    log(f"submission {spec['submission_id']} (run {spec['run_name']}, project {spec.get('project_dir')})")

    pass_start = time.time()
    consume_marker(spec_path.parent, pass_start)     # this pass handles every chain that ended before it
    result = run_tick(spec_path, SUPERVISOR_PURPOSE)
    spec = load_submission_spec(spec_path)
    for line in format_overview(compute_overview(output_dir, spec), hint=False):
        log(line)
    finalize = finalize_state(spec)
    if finalize:
        log(f"finalize:   {finalize}")
    action = decide(tick_state=result.state, checks=checks)
    log(f"check {checks}: pass {result.state} -> {action}")
    if action == "done":
        return 0
    if action == "give-up":
        for error in result.errors[:20]:
            log(f"  {error}")
        log("giving up: law has no retries left (or too many checks); inspect with `humsub submission-status`, "
            f"fix the cause and run `humsub resume {spec['submission_id']}` (fresh retries, supervisor queued again)")
        return 1
    opts = spec.get("submit_options", {})
    if requested_since(spec_path.parent, pass_start):
        delay = trigger_delay(opts)
        why = "a chain ended during this pass"
    else:
        delay = int(opts.get("supervisor_interval", 1800))
        why = "the interval"
    job = submit_supervisor(spec_path, round_no, checks + 1, delay)
    log(f"queued next check as job {job}, in {delay} s ({why}; ending chains bring it forward)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
