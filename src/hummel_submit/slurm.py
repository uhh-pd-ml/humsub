from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Any


class SlurmError(RuntimeError):
    pass


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
        f"--gpus={slurm['gpus']}",
        f"--time={slurm['time_limit']}",
        "--export=NONE",
        f"--signal=B:USR1@{slurm['signal_seconds']}",
        f"--chdir={state['project_dir']}",
        f"--output={output_dir / 'logs' / '%x_%j.log'}",
    ]
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
