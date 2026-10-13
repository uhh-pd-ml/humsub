"""``humsub doctor``: can this setup run a production?  Checks on the frontend and, with ``--probe``, from a real batch node.

The questions that are cheap to answer *before* thousands of jobs are submitted and expensive afterwards:

* does the configuration resolve; are ``output_dir`` / ``cache_dir`` usable; does the scheduler answer; how full is the SSD quota?
* (probe) can a batch node write to the output locations — including group folders that are not the user's own BeeGFS space?
  can a batch job call ``sbatch`` (needed by ``--supervisor-job``)?  are ``$HOME``/``$USW`` read-only as the rules say?
  can the batch node import humsub and law with the interpreter the chains will use?  is the account/partition(/GPU) combination accepted?

The probe is one tiny job with the account, partition and GPU request of the configuration, queued with ``--nice 0`` unless
``--keep-nice`` (it should answer within minutes; it holds one CPU for a few seconds).
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

from .pathcheck import PathCheckError, check_compute_writable
from .quota import parse_rrz_quota, quota_usage
from .slurm import SlurmError, base_sbatch_args, cancel_jobs, query_jobs, slurm_command, slurm_log_path


@dataclass
class Check:
    status: str      # ok | fail | warn | info
    name: str
    detail: str = ""

    def line(self) -> str:
        return f"[{self.status:>4}] {self.name}" + (f": {self.detail}" if self.detail else "")


def local_checks(config: dict[str, Any], extra_paths: list[Path]) -> list[Check]:
    checks: list[Check] = []
    exe, slurm = config["execution"], config["slurm"]
    checks.append(Check("info", "configuration", f"account={slurm['account']} partition={slurm['partition']} gpus={slurm['gpus']} nice={slurm['nice']} "
                                                f"time_limit={slurm['time_limit']} signal={slurm['signal_seconds']}s grace={slurm['grace_seconds']}s"))
    for label, path, shared in (("output_dir", exe["output_dir"], True), ("cache_dir", exe["cache_dir"], False)):
        try:
            res = check_compute_writable(Path(path), f"execution.{label}", must_be_shared=shared)
            checks.append(Check("warn" if res.status == "warning" else "ok", label, f"{res.resolved} ({res.detail})"))
        except PathCheckError as exc:
            checks.append(Check("fail", label, str(exc)))
    for path in extra_paths:
        try:
            res = check_compute_writable(path, "path", must_be_shared=True)
            checks.append(Check("ok", f"path {path}", res.detail))
        except PathCheckError as exc:
            checks.append(Check("fail", f"path {path}", str(exc)))
    try:
        proc = subprocess.run([slurm_command("squeue"), "-h", "-u", os.environ.get("USER", ""), "-o", "%i"], check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        checks.append(Check("ok", "scheduler", f"squeue answers ({len(proc.stdout.split())} of your jobs queued or running)"))
    except (OSError, subprocess.SubprocessError) as exc:
        checks.append(Check("fail", "scheduler", f"squeue failed: {exc}"))
    usage = quota_usage(Path(exe["cache_dir"]))
    if usage:
        used, total = usage
        frac = used / total if total else 0
        checks.append(Check("warn" if frac > 0.5 else "ok", "SSD quota", f"{used / 2**30:.1f} of {total / 2**30:.0f} GiB used ({100 * frac:.0f} %)"))
    else:
        checks.append(Check("info", "SSD quota", "not determinable here"))
    try:
        import law
        checks.append(Check("ok", "law", f"{getattr(law, '__version__', '?')} (python {sys.version.split()[0]} at {sys.executable})"))
    except ImportError as exc:
        checks.append(Check("fail", "law", f"not importable: {exc}"))
    if slurm["nice"] == 0:
        checks.append(Check("warn", "nice", "0: the priority penalty is disabled; chains compete with other users' jobs"))
    return checks


def probe_script(paths: list[Path], python: str, account: str, partition: str, gpus: int) -> str:
    """The batch script of the probe; prints ``DOCTOR <status> <name>: <detail>`` lines."""
    lines = [
        "#!/bin/bash",
        "source /sw/batch/init.sh 2>/dev/null",
        'echo "DOCTOR info batch node: $(hostname), $(date -Is), job $SLURM_JOB_ID"',
    ]
    for p in paths:
        q = shlex.quote(str(p))
        lines += [
            f'mkdir -p {q} 2>/dev/null; f={q}/.humsub-doctor-$SLURM_JOB_ID',
            f'if ( : > "$f" ) 2>/dev/null; then rm -f "$f"; echo "DOCTOR ok writable from the batch node: {p}";'
            f' else echo "DOCTOR fail writable from the batch node: {p}: cannot create a file there (not shared, read-only or no quota)"; fi',
        ]
    for var in ("HOME", "USW"):
        lines.append(
            f'if [ -n "${{{var}:-}}" ]; then if ( : > "${var}/.humsub-doctor-$SLURM_JOB_ID" ) 2>/dev/null; then rm -f "${var}/.humsub-doctor-$SLURM_JOB_ID";'
            f' echo "DOCTOR info {var} is writable in batch jobs (expected read-only on Hummel-2)"; else echo "DOCTOR ok {var} is read-only in batch jobs, as expected"; fi; fi'
        )
    gpu_arg = f"--gpus={gpus}" if gpus else ""
    lines += [
        f'j=$(sbatch --parsable --hold --account={shlex.quote(account)} --partition={shlex.quote(partition)} {gpu_arg} --time=1 --nice=1000000 '
        f'--output=/dev/null --wrap=true 2>&1)',
        'if [[ "$j" =~ ^[0-9]+ ]]; then scancel "${j%%;*}"; echo "DOCTOR ok nested sbatch: a batch job can submit jobs (--supervisor-job works); test job ${j%%;*} held and cancelled";'
        ' else echo "DOCTOR fail nested sbatch: a batch job cannot submit jobs ($j); --supervisor-job will not work"; fi',
        f'if {shlex.quote(python)} -c "import law, hummel_submit; print(\'DOCTOR ok humsub importable in batch: law\', getattr(law, \'__version__\', \'?\'), \'humsub\', hummel_submit.__version__)" 2>/dev/null;'
        f' then :; else echo "DOCTOR fail humsub/law not importable in batch with {python}"; fi',
        'echo "DOCTOR info probe finished"',
    ]
    return "\n".join(lines) + "\n"


def parse_probe_log(text: str) -> list[Check]:
    out = []
    for line in text.splitlines():
        if not line.startswith("DOCTOR "):
            continue
        parts = line.split(" ", 2)
        if len(parts) < 3:
            continue
        status, rest = parts[1], parts[2]
        name, _, detail = rest.partition(": ")
        out.append(Check(status if status in ("ok", "fail", "warn", "info") else "info", "probe: " + name, detail))
    return out


def run_probe(config: dict[str, Any], output_dir: Path, paths: list[Path], timeout: int, keep_nice: bool = False,
              cpu_only: bool = False) -> list[Check]:
    import copy
    cfg = copy.deepcopy(config)
    cfg["slurm"]["time_limit"] = "00:05:00"
    cfg["slurm"]["max_hops"] = 1
    if not keep_nice:
        cfg["slurm"]["nice"] = 0
    if cpu_only:
        cfg["slurm"]["gpus"] = 0
    state = {"config": cfg, "project_dir": str(Path.cwd()), "output_dir": str(output_dir), "resubmit": False}
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    script = output_dir / "logs" / f".doctor-probe-{os.getpid()}.sh"
    script.write_text(probe_script(paths, sys.executable, cfg["slurm"]["account"], cfg["slurm"]["partition"], int(cfg["slurm"]["gpus"])))
    script.chmod(0o755)
    cmd = [slurm_command("sbatch"), *base_sbatch_args(state), f"--job-name=humsub-doctor", str(script)]
    try:
        proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        script.unlink(missing_ok=True)
        return [Check("fail", "probe: submit", f"sbatch refused the probe job: {exc.stderr.strip()} (account/partition/gpus combination wrong?)")]
    job = proc.stdout.strip().split(";", 1)[0]
    deadline = time.time() + timeout
    status = None
    while time.time() < deadline:
        st = query_jobs([job]).get(job)
        if st is not None and not st.pending and not st.running:
            status = st
            break
        time.sleep(4)
    script.unlink(missing_ok=True)
    if status is None:
        cancel_jobs([job])
        return [Check("warn", "probe: queue", f"job {job} did not finish within {timeout} s (cluster busy?); cancelled. Retry with --timeout or look at `squeue`")]
    log = output_dir / "logs" / f"humsub-doctor_{job}.log"
    time.sleep(1)
    checks = parse_probe_log(log.read_text(errors="replace")) if log.exists() else []
    if not checks:
        checks.append(Check("fail", "probe: log", f"job {job} ended {status.state} without output (see {log})"))
    elif not status.finished:
        checks.append(Check("fail", "probe: job", f"ended {status.state}"))
    log.unlink(missing_ok=True)
    return checks


def manifest_paths(manifest: Path) -> list[Path]:
    import json
    data = json.loads(manifest.read_text())
    outs = [o for b in data["branches"] for o in b["outputs"] + b.get("extra_outputs", [])]
    return [Path(os.path.commonpath(outs))] if outs else []
