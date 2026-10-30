"""Supervisor job: a tiny Slurm job that drives a submission's law task graph without a login-node process.

Every ``supervisor_interval`` seconds it runs one *pass* (:mod:`hummel_submit.tick`): law polls the workflow once,
resubmits failed jobs within ``retries`` and submits what is pending.  It stops when the top law task is complete,
and gives up when law reports jobs that failed with no retries left.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from .lineage import compute_overview, finalize_state, format_overview
from .tick import run_tick
from .slurm import SlurmError, slurm_command
from .state import write_worker_snapshot
from .submission import append_spec_list, load_submission_spec

SUPERVISOR_TIME = "00:30:00"
MAX_CHECKS = 2000


def log(message: str) -> None:
    import os
    import time
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [supervisor:{os.environ.get('SLURM_JOB_ID', 'local')}] {message}", flush=True)


def supervisor_command(spec_path: Path, round_no: int, checks: int = 1) -> list[str]:
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
    interval = int(opts.get("supervisor_interval", 1800))
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


def submit_supervisor(spec_path: Path, round_no: int, checks: int = 1) -> str:
    cmd = supervisor_command(spec_path, round_no, checks)
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


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: python -m hummel_submit.supervisor SUBMISSION_JSON ROUND[:CHECKS]", file=sys.stderr)
        return 2
    spec_path = Path(argv[0])
    parts = argv[1].split(":")
    round_no = int(parts[0])  # kept in the job arguments for older frozen supervisor scripts; law counts retries now
    checks = int(parts[1]) if len(parts) > 1 else 1
    spec = load_submission_spec(spec_path)
    output_dir = Path(spec["output_dir"])

    result = run_tick(spec_path)
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
            "fix the cause and run `humsub tick` or `humsub resume`")
        return 1
    job = submit_supervisor(spec_path, round_no, checks + 1)
    log(f"queued next check as job {job}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
