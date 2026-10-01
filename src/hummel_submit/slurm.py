from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Any


class SlurmError(RuntimeError):
    pass


@dataclass(frozen=True)
class SlurmJobStatus:
    job_id: str
    state: str
    exit_code: int | None = None

    @property
    def pending(self) -> bool:
        return self.state in {"CONFIGURING", "PENDING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED"}

    @property
    def running(self) -> bool:
        return self.state in {"RUNNING", "COMPLETING", "STAGE_OUT"}

    @property
    def finished(self) -> bool:
        return self.state == "COMPLETED" and (self.exit_code in (None, 0))

    @property
    def failed(self) -> bool:
        return not (self.pending or self.running or self.finished)


def slurm_command(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    fallback = Path("/syssw/slurm/current/bin") / name
    return str(fallback)


def base_sbatch_args(state: dict[str, Any]) -> list[str]:
    cfg = state["config"]
    slurm = cfg["slurm"]
    output_dir = Path(state["output_dir"])
    args = [
        "--parsable",
        f"--job-name={slurm['job_name']}",
        f"--account={slurm['account']}",
        f"--partition={slurm['partition']}",
        f"--nodes={slurm['nodes']}",
        f"--time={slurm['time_limit']}",
        "--export=NONE",
        f"--chdir={state['project_dir']}",
        f"--output={output_dir / 'logs' / '%x_%j.log'}",
    ]
    if slurm["gpus"] > 0:
        args.append(f"--gpus={slurm['gpus']}")
    if state.get("resubmit", True) and slurm.get("max_hops", 2) > 1:
        args.append(f"--signal=B:USR1@{slurm['signal_seconds']}")
    if slurm["mail"]:
        args += [f"--mail-user={slurm['mail']}", "--mail-type=FAIL"]
    if slurm["reservation"]:
        args.append(f"--reservation={slurm['reservation']}")
    args.extend(slurm["extra_args"])
    return args


def sbatch_command(state: dict[str, Any], state_path: Path, hop: int, dependency: str | None = None) -> list[str]:
    args = [slurm_command("sbatch"), *base_sbatch_args(state)]
    if dependency:
        args.append(f"--dependency=afterany:{dependency}")
    args += [
        state["worker_script"],
        state["python_executable"],
        state["snapshot_path"],
        str(state_path),
        str(hop),
    ]
    return args


def submit(state: dict[str, Any], state_path: Path, hop: int, dependency: str | None = None) -> str:
    cmd = sbatch_command(state, state_path, hop, dependency)
    try:
        proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise SlurmError(f"sbatch failed: {detail.strip()}") from exc
    job_id = proc.stdout.strip().split(";", 1)[0]
    if not job_id:
        raise SlurmError("sbatch returned no job id")
    return job_id


def cancel_jobs(job_ids: list[str]) -> None:
    if not job_ids:
        return
    subprocess.run([slurm_command("scancel"), *job_ids], check=False)


def queue_status(job_ids: list[str]) -> str:
    if not job_ids:
        return ""
    proc = subprocess.run(
        [slurm_command("squeue"), "-h", "-j", ",".join(job_ids), "-o", "%i %T %M %R"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return proc.stdout.rstrip()


def _parse_exit_code(raw: str) -> int | None:
    first = raw.split(":", 1)[0].strip()
    try:
        return int(first)
    except ValueError:
        return None


def parse_squeue_status(text: str) -> dict[str, SlurmJobStatus]:
    result: dict[str, SlurmJobStatus] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 1)
        if len(parts) != 2:
            continue
        job_id, state = parts
        result[job_id.strip()] = SlurmJobStatus(job_id.strip(), state.strip().split()[0])
    return result


def parse_sacct_status(text: str, requested: set[str] | None = None) -> dict[str, SlurmJobStatus]:
    result: dict[str, SlurmJobStatus] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        job_id, state, exit_code = (part.strip() for part in parts[:3])
        # sacct also returns .batch/.extern steps.  The chain tracks top-level
        # Slurm jobs only, so ignore scheduler steps here.
        if "." in job_id:
            continue
        if requested is not None and job_id not in requested:
            continue
        result[job_id] = SlurmJobStatus(job_id, state.split()[0], _parse_exit_code(exit_code))
    return result


def query_jobs(job_ids: list[str]) -> dict[str, SlurmJobStatus]:
    """Query current and accounting state for top-level Slurm jobs."""
    if not job_ids:
        return {}

    requested = set(map(str, job_ids))
    result: dict[str, SlurmJobStatus] = {}

    squeue = subprocess.run(
        [slurm_command("squeue"), "-h", "-j", ",".join(job_ids), "-o", "%i|%T"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if squeue.returncode == 0:
        result.update(parse_squeue_status(squeue.stdout))

    missing = requested - set(result)
    if missing:
        sacct = subprocess.run(
            [
                slurm_command("sacct"),
                "-n",
                "-X",
                "-P",
                "-j",
                ",".join(sorted(missing)),
                "-o",
                "JobIDRaw,State,ExitCode",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if sacct.returncode == 0:
            result.update(parse_sacct_status(sacct.stdout, requested=missing))

    return result
