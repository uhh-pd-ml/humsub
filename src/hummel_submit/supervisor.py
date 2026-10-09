"""Supervisor job: a tiny Slurm job that periodically resumes missing branches without a login-node process.

It checks the lineage of a submission every ``supervisor_interval`` seconds.  While chains are alive it
only re-queues itself; when nothing is alive and branches are still missing it runs ``resume`` (at most
``supervisor_max_rounds`` times, so one broken input cannot consume jobs forever) and queues itself once
more to verify the outcome.  It stops when every output exists.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from .lineage import compute_overview, format_overview, resume
from .slurm import SlurmError, slurm_command
from .state import write_worker_snapshot
from .submission import load_submission_spec

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
    if slurm.get("nice", 0) > 0:
        args.append(f"--nice={slurm['nice']}")
    args += [str(script), spec["python_executable"], str(snapshot), str(spec_path), f"{round_no}:{checks}"]
    return args


def submit_supervisor(spec_path: Path, round_no: int, checks: int = 1) -> str:
    cmd = supervisor_command(spec_path, round_no, checks)
    try:
        proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SlurmError(f"supervisor sbatch failed: {getattr(exc, 'stderr', '') or exc}") from exc
    return proc.stdout.strip().split(";", 1)[0]


def decide(*, missing: int, active_branches: int, round_no: int, max_rounds: int, checks: int) -> str:
    """'done' | 'wait' | 'resume' | 'give-up'."""
    if missing == 0 and active_branches == 0:
        return "done"
    if active_branches > 0:
        return "wait" if checks < MAX_CHECKS else "give-up"
    if round_no > max_rounds:
        return "give-up"
    return "resume"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: python -m hummel_submit.supervisor SUBMISSION_JSON ROUND[:CHECKS]", file=sys.stderr)
        return 2
    spec_path = Path(argv[0])
    parts = argv[1].split(":")
    round_no = int(parts[0])
    checks = int(parts[1]) if len(parts) > 1 else 1
    spec = load_submission_spec(spec_path)
    output_dir = Path(spec["output_dir"])
    max_rounds = int(spec.get("submit_options", {}).get("supervisor_max_rounds", 3))

    ov = compute_overview(output_dir, spec)
    for line in format_overview(ov):
        log(line)
    action = decide(missing=len(ov.missing), active_branches=ov.active, round_no=round_no, max_rounds=max_rounds, checks=checks)
    log(f"round {round_no}/{max_rounds}, check {checks}: {action}")
    if action == "done":
        return 0
    if action == "give-up":
        log(f"giving up: {len(ov.missing)} branch(es) still missing after {max_rounds} resume round(s); inspect with `humsub submission-status`")
        return 1
    if action == "resume":
        resume(output_dir, spec_path, from_supervisor=True)
        round_no += 1
    job = submit_supervisor(spec_path, round_no, checks + 1)
    log(f"queued next check as job {job}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
