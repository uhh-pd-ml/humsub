from __future__ import annotations

import os
from pathlib import Path
import sys

from .chain_runner import log, run_chain_payload
from .slurm import SlurmError, cancel_jobs, submit
from .state import append_job, continue_marker, done_marker, load_state, mark_status


def should_queue_follower(hop: int, max_hops: int) -> bool:
    return hop + 1 < max_hops


def _mark_done(state_path: Path, reason: str, **fields: object) -> None:
    done_marker(state_path).write_text(reason + "\n", encoding="utf-8")
    mark_status(state_path, reason, **fields)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("internal worker usage: python -m hummel_submit.worker STATE_PATH HOP", file=sys.stderr)
        return 2

    state_path = Path(argv[0]).resolve()
    hop = int(argv[1])
    state = load_state(state_path)
    slurm = state["config"]["slurm"]
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        print("humsub worker must run inside a SLURM job", file=sys.stderr)
        return 2

    state = append_job(state_path, job_id)

    if done_marker(state_path).exists():
        log("chain is already marked done; exiting")
        return 0

    mark_status(state_path, f"running-hop-{hop + 1}", current_job_id=job_id, current_hop=hop)

    if hop > 0 and not continue_marker(state_path, hop - 1).exists():
        log("previous hop did not explicitly request continuation; stopping chain")
        _mark_done(state_path, "stopped-no-continuation-marker")
        return 0

    budget = int(state.get("failure_budget", 0))
    max_hops = slurm["max_hops"] + budget
    log(f"run={state['run_name']} chain={state['chain_id']} job {hop + 1} of at most {max_hops}")

    next_job: str | None = None
    if state["resubmit"] and should_queue_follower(hop, max_hops):
        try:
            next_job = submit(state, state_path, hop + 1, dependency=job_id)
            append_job(state_path, next_job)
            mark_status(state_path, f"running-hop-{hop + 1}", current_job_id=job_id, next_job_id=next_job)
            log(f"queued follower {next_job}; it will run only if this hop explicitly requests continuation")
        except SlurmError as exc:
            log(f"WARNING: could not queue follower: {exc}")
    elif state["resubmit"]:
        log(f"reached the configured limit of {max_hops} jobs; no follower queued")

    try:
        rc, timed_out = run_chain_payload(state, state_path, hop)
    except Exception as exc:
        log(f"law payload setup failed: {exc}")
        if next_job:
            cancel_jobs([next_job])
        _mark_done(state_path, "failed-to-start", error=str(exc), exit_code=2)
        return 2

    if rc == 0:
        # Also when the soft-stop notice had been given: law exits 0 only if every branch of the job is complete,
        # i.e. the payload finished inside the grace window and there is nothing left to continue.
        log("law payload completed normally; stopping chain")
        if next_job:
            cancel_jobs([next_job])
        _mark_done(state_path, "completed", exit_code=0)
        return 0

    if timed_out:
        if next_job:
            mark_status(
                state_path,
                f"continuing-after-hop-{hop + 1}",
                current_job_id=job_id,
                next_job_id=next_job,
            )
            log("time slice ended; follower will continue the chain")
            # Intentional handoff is successful from Slurm's point of view.  To
            # law, the stable chain id remains RUNNING until a later hop either
            # completes or fails.
            return 0

        log("time slice ended but no follower is queued; chain cannot continue")
        _mark_done(state_path, "stopped-no-follower", exit_code=1)
        return 1

    used = int(state.get("failures_used", 0))
    if next_job and used < budget:
        # Safety margin: the follower hop (already queued with afterany) re-runs the chain; law skips
        # every branch whose outputs exist, so only the failed and not-yet-run branches are repeated.
        (state_path.parent / f"continue-{hop}").touch()
        mark_status(
            state_path,
            f"retrying-after-failure-hop-{hop + 1}",
            current_job_id=job_id,
            next_job_id=next_job,
            failures_used=used + 1,
            last_failure_exit_code=rc,
        )
        log(f"law payload failed with exit code {rc}; failure budget {used + 1}/{budget} used, follower hop will retry")
        return rc if 0 < rc <= 255 else 1

    log(f"law payload failed with exit code {rc}; stopping chain")
    if budget and used >= budget:
        log(f"failure budget of {budget} retry hop(s) exhausted")
    if next_job:
        cancel_jobs([next_job])
    _mark_done(state_path, f"failed-{rc}", exit_code=rc)
    return rc if 0 <= rc <= 255 else 1


if __name__ == "__main__":
    raise SystemExit(main())
