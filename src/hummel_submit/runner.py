from __future__ import annotations

import glob
import os
from pathlib import Path
import shutil
import shlex
import signal
import subprocess
import time
from typing import Any

from .envfile import load_env_file


def log(message: str) -> None:
    job = os.environ.get("SLURM_JOB_ID", "worker")
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] [job:{job}] {message}", flush=True)


def newest_checkpoint(state: dict[str, Any]) -> Path | None:
    pattern = state["config"]["execution"]["checkpoint_glob"]
    if not pattern:
        return None
    run_dir = Path(state["run_dir"])
    matches = [Path(p) for p in glob.glob(str(run_dir / pattern), recursive=True)]
    files = [p for p in matches if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def detect_ngpu() -> int:
    raw = os.environ.get("SLURM_GPUS_ON_NODE", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        proc = subprocess.run([nvidia, "-L"], check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        count = sum(1 for line in proc.stdout.splitlines() if line.strip())
        if count:
            return count
    return 0


def render_auto_args(auto_args: list[str], user_args: list[str], values: dict[str, str]) -> list[str]:
    user_keys = {arg.split("=", 1)[0] for arg in user_args if arg.startswith("--")}
    rendered: list[str] = []
    for token in auto_args:
        for key, value in values.items():
            token = token.replace("{" + key + "}", value)
        option_key = token.split("=", 1)[0] if token.startswith("--") else None
        if option_key and option_key in user_keys:
            continue
        rendered.append(token)
    return rendered


def build_command(state: dict[str, Any], *, resuming: bool) -> tuple[list[str], Path | None, int, str]:
    exe = state["config"]["execution"]
    ngpu = detect_ngpu()
    strategy = "ddp" if ngpu > 1 else "auto"
    ckpt = newest_checkpoint(state) if resuming else None
    values = {
        "RUN": state["run_name"],
        "RUN_DIR": state["run_dir"],
        "NGPU": str(ngpu),
        "STRATEGY": strategy,
        "CKPT": str(ckpt) if ckpt else "null",
    }
    auto = render_auto_args(exe["auto_args"], state["user_args"], values)
    return [*exe["command"], *auto, *state["user_args"]], ckpt, ngpu, strategy


def _runtime_env(state: dict[str, Any], job_id: str) -> tuple[dict[str, str], dict[str, str]]:
    exe = state["config"]["execution"]
    env_file = Path(exe["env_file"])
    loaded = load_env_file(env_file)
    env = os.environ.copy()
    env.update(loaded)
    cache = f"/tmp/{os.environ.get('USER', 'user')}-cache-{job_id}"
    runtime = {
        "SLURM_JOB_NAME": "interactive",
        "MPLCONFIGDIR": f"{cache}/mpl",
        "TRITON_CACHE_DIR": f"{cache}/triton",
        "TORCHINDUCTOR_CACHE_DIR": f"{cache}/inductor",
        "PYTHONPATH": state["project_dir"] + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        "HUMMEL_RUN_NAME": state["run_name"],
        "HUMMEL_RUN_DIR": state["run_dir"],
        "HUMMEL_CHAIN_ID": state["chain_id"],
    }
    env.update(runtime)
    return env, {**loaded, **runtime}


def make_process_command(state: dict[str, Any], command: list[str], ngpu: int) -> tuple[list[str], dict[str, str]]:
    exe = state["config"]["execution"]
    job_id = os.environ.get("SLURM_JOB_ID", "unknown")
    env, container_env = _runtime_env(state, job_id)
    if exe["image"] == "none":
        return command, env

    apptainer = exe.get("apptainer") or shutil.which("apptainer") or "/sw/env/system-gcc/apptainer/1.4.5/bin/apptainer"
    proc_cmd = [apptainer, "exec"]
    if exe.get("nv", True):
        proc_cmd.append("--nv")
    if exe["binds"]:
        proc_cmd += ["-B", ",".join(exe["binds"])]
    proc_cmd += [exe["image"], *command]

    for key, value in container_env.items():
        env[f"APPTAINERENV_{key}"] = value
    return proc_cmd, env


def run_payload(state: dict[str, Any], state_path: Path, hop: int) -> tuple[int, bool, Path | None]:
    resuming = hop > 0
    command, ckpt, ngpu, strategy = build_command(state, resuming=resuming)
    if ckpt:
        log(f"resuming from {ckpt}")
    elif resuming and state["config"]["execution"]["checkpoint_glob"]:
        log("WARNING: no checkpoint found in this run directory; starting this hop without one")
    log(f"GPUs visible={ngpu}, strategy={strategy}")
    log("running: " + shlex.join(command))

    proc_cmd, env = make_process_command(state, command, ngpu)
    timed_out = False
    child: subprocess.Popen[str] | None = None

    def on_usr1(signum: int, frame: object) -> None:
        nonlocal timed_out, child
        timed_out = True
        (state_path.parent / f"continue-{hop}").touch()
        log("time limit approaching: marked chain for continuation and sending SIGTERM to the payload")
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    old_handler = signal.signal(signal.SIGUSR1, on_usr1)
    try:
        child = subprocess.Popen(
            proc_cmd,
            cwd=state["project_dir"],
            env=env,
            text=True,
            start_new_session=True,
        )
        rc = child.wait()
    finally:
        signal.signal(signal.SIGUSR1, old_handler)

    log(f"payload finished with exit code {rc} (pre-timeout signal received: {timed_out})")
    return rc, timed_out, newest_checkpoint(state)
