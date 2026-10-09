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


def hop_timing_env(state: dict[str, Any], now: float | None = None) -> dict[str, str]:
    """Start time and usable seconds of this hop, for the payload's time-aware retry policy."""
    from .config import slurm_time_seconds
    slurm = state["config"]["slurm"]
    limit = slurm_time_seconds(slurm["time_limit"])
    if limit is None:
        return {"HUMSUB_HOP_START": str(time.time() if now is None else now)}
    signals = state.get("resubmit", True) and slurm.get("max_hops", 2) + state.get("failure_budget", 0) > 1
    usable = limit - (slurm["signal_seconds"] if signals else 0)
    return {
        "HUMSUB_HOP_START": str(time.time() if now is None else now),
        "HUMSUB_HOP_USABLE_SECONDS": str(max(usable, 0)),
    }


def wait_group_gone(pgid: int, timeout: float = 90.0, poll: float = 0.25) -> bool:
    """Wait until no process of group *pgid* is left; SIGKILL the group after *timeout* seconds.

    The leader (the law job script) can exit before the processes it started: the law/Python process of
    a branch needs a moment after SIGTERM to stop its payload and remove the scratch.  The Slurm job must
    not end before that, or the cleanup is cut short.  Returns False if the group had to be killed.
    """
    deadline = time.time() + timeout
    while True:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        if time.time() > deadline:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return False
        time.sleep(poll)


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
    env.update(hop_timing_env(state))
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
        if not wait_group_gone(child.pid):
            log("processes of the law payload outlived the grace period and were killed")
    finally:
        signal.signal(signal.SIGUSR1, old_handler)

    log(f"law payload finished with exit code {rc} (pre-timeout signal received: {timed_out})")
    return rc, timed_out
