from __future__ import annotations

import os
from pathlib import Path
import sys

from .runner import log, run_payload
from .slurm import SlurmError, cancel_jobs, submit
from .state import append_job, atomic_write_json, continue_marker, done_marker, load_state


def _mark_done(state_path: Path, state: dict, reason: str) -> None:
    done_marker(state_path).write_text(reason + "\n", encoding="utf-8")
    state["status"] = reason
    atomic_write_json(state_path, state)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("internal worker usage: python -m hummel_submit.worker STATE_PATH HOP", file=sys.stderr)
        return 2

    state_path = Path(argv[0]).resolve()
    hop = int(argv[1])
    state = load_state(state_path)
    cfg = state["config"]
    slurm = cfg["slurm"]
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        print("hummel-submit worker must run inside a SLURM job", file=sys.stderr)
        return 2

    state = append_job(state_path, job_id)

    if done_marker(state_path).exists():
        log("chain is already marked done; exiting")
        return 0

    if hop > 0 and not continue_marker(state_path, hop - 1).exists():
        log("previous hop did not explicitly request continuation; stopping chain")
        _mark_done(state_path, state, "stopped-no-continuation-marker")
        return 0

    max_hops = slurm["max_hops"]
    log(f"run={state['run_name']} chain={state['chain_id']} job {hop + 1} of at most {max_hops}")

    next_job: str | None = None
    if state["resubmit"] and hop + 1 < max_hops:
        try:
            next_job = submit(state, state_path, hop + 1, dependency=job_id)
            state = append_job(state_path, next_job)
            log(f"queued follower {next_job}; it will run only if this hop explicitly requests continuation")
        except SlurmError as exc:
            log(f"WARNING: could not queue follower: {exc}")
    elif state["resubmit"]:
        log(f"reached the configured limit of {max_hops} jobs; no follower queued")

    try:
        rc, timed_out, checkpoint = run_payload(state, state_path, hop)
    except Exception as exc:
        log(f"payload setup failed: {exc}")
        if next_job:
            cancel_jobs([next_job])
        _mark_done(state_path, load_state(state_path), "failed-to-start")
        return 2

    state = load_state(state_path)
    if rc == 0 and not timed_out:
        log("run completed normally; stopping chain")
        if next_job:
            cancel_jobs([next_job])
        _mark_done(state_path, state, "completed")
        return 0

    if timed_out:
        if next_job:
            log("time slice ended; follower will continue the chain")
        else:
            log("time slice ended but no follower is queued; chain stops here")
            _mark_done(state_path, state, "stopped-no-follower")
        # An intentional time-slice handoff is a successful batch job, avoiding
        # spurious SLURM failure mail for every slice.
        return 0

    if slurm["retry_on_failure"] and checkpoint is not None and next_job:
        continue_marker(state_path, hop).touch()
        log(f"payload failed with exit code {rc}, but retry_on_failure=true and checkpoint exists; follower will retry")
        return rc if 0 <= rc <= 255 else 1

    log(f"payload failed with exit code {rc}; stopping chain")
    if next_job:
        cancel_jobs([next_job])
    _mark_done(state_path, state, f"failed-{rc}")
    return rc if 0 <= rc <= 255 else 1


if __name__ == "__main__":
    raise SystemExit(main())
