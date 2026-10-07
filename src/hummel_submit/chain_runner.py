from __future__ import annotations

import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
from typing import Any


def log(message: str) -> None:
    job = os.environ.get("SLURM_JOB_ID", "worker")
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] [job:{job}] {message}", flush=True)


def run_chain_payload(
    state: dict[str, Any],
    state_path: Path,
    hop: int,
) -> tuple[int, bool]:
    """Execute one immutable law-generated remote job script for a chain hop.

    The chain wrapper owns only scheduler lifetime.  The payload script owns the
    actual workflow semantics (normally a law remote job).  On the Hummel
    pre-timeout signal, the wrapper records the explicit continuation marker and
    terminates the payload process group so framework-specific code gets a chance
    to checkpoint via its normal signal handling.
    """
    payload = Path(state["payload_script"])
    cwd = Path(state["payload_cwd"])
    if not payload.is_file():
        raise FileNotFoundError(f"chain payload script is missing: {payload}")
    if not cwd.is_dir():
        raise FileNotFoundError(f"chain payload working directory is missing: {cwd}")

    env = os.environ.copy()
    env.update({
        "HUMSUB_CHAIN_ID": state["chain_id"],
        "HUMSUB_HOP": str(hop),
        "HUMSUB_STATE_PATH": str(state_path),
        "HUMMEL_RUN_NAME": state["run_name"],
    })

    command = ["bash", str(payload)]
    log("running law payload: " + shlex.join(command))

    timed_out = False
    child: subprocess.Popen[str] | None = None

    def on_usr1(signum: int, frame: object) -> None:
        nonlocal timed_out
        timed_out = True
        (state_path.parent / f"continue-{hop}").touch()
        log("time limit approaching: marked chain for continuation and sending SIGTERM to the law payload")
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    old_handler = signal.signal(signal.SIGUSR1, on_usr1)
    try:
        child = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            text=True,
            start_new_session=True,
        )
        rc = child.wait()
    finally:
        signal.signal(signal.SIGUSR1, old_handler)

    log(f"law payload finished with exit code {rc} (pre-timeout signal received: {timed_out})")
    return rc, timed_out
