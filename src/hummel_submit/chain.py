from __future__ import annotations

import contextlib
from dataclasses import dataclass
import fcntl
from pathlib import Path
import shutil

from .slurm import cancel_jobs, query_jobs, submit
from .state import create_chain_state, done_marker, load_state, mark_status, state_path
from .submission import append_submission_chain, load_submission_spec


@dataclass(frozen=True)
class ChainStatus:
    chain_id: str
    state: str
    code: int | None = None
    error: str | None = None

    @property
    def terminal(self) -> bool:
        return self.state in {"finished", "failed"}


def lane_dependency(spec: dict[str, object]) -> str | None:
    """Slurm job id this chain must wait for to keep at most ``max_concurrent`` chains in flight.

    Chains are assigned round-robin to ``max_concurrent`` lanes; a chain's first hop starts after the
    most recent job of the previous chain in its lane.  Continuation/retry hops of that chain can
    briefly overlap with the next chain, so the cap is "N plus chains currently in a later hop".
    """
    n = int(spec.get("max_concurrent", 0) or 0)
    chain_ids = list(spec.get("chain_ids", []))
    if n < 1 or len(chain_ids) < n:
        return None
    try:
        previous = load_state(state_path(Path(str(spec["output_dir"])), chain_ids[-n]))
    except FileNotFoundError:
        return None
    jobs = previous.get("jobs", [])
    return str(jobs[-1]) if jobs else None


@contextlib.contextmanager
def _submission_lock(submission_spec_path: Path):
    """Serialise chain creation: law submits from several threads, but lane assignment
    (``--max-concurrent``) must see every previously created chain."""
    with (submission_spec_path.parent / ".submit-chain.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def submit_chain(job_file: Path, submission_spec_path: Path) -> str:
    """Create and launch an autonomous chain for one law-generated remote job."""
    with _submission_lock(submission_spec_path):
        return _submit_chain_locked(job_file, submission_spec_path)


def _submit_chain_locked(job_file: Path, submission_spec_path: Path) -> str:
    spec = load_submission_spec(submission_spec_path)
    lane_after = lane_dependency(spec)
    output_dir = Path(spec["output_dir"])
    project_dir = Path(spec["project_dir"])
    package_dir = Path(__file__).resolve().parent

    state, chain_state_path = create_chain_state(
        output_dir=output_dir,
        project_dir=project_dir,
        config=spec["config"],
        run_name=spec["run_name"],
        resubmit=bool(spec["resubmit"]),
        python_executable=spec["python_executable"],
        package_dir=package_dir,
        payload_script=job_file,
        submission_id=spec["submission_id"],
        failure_budget=int(spec.get("failure_budget", 0)),
        lane_after=lane_after,
    )

    try:
        inner_job_id = submit(state, chain_state_path, 0)
    except Exception:
        # No scheduler job became live, so remove the unused frozen chain state.
        # law's own submission retry logic can then make a clean attempt without
        # leaving orphaned chain directories behind.
        shutil.rmtree(chain_state_path.parent, ignore_errors=True)
        raise

    from .state import append_job

    append_job(chain_state_path, inner_job_id)
    append_submission_chain(submission_spec_path, state["chain_id"])
    return state["chain_id"]


def cancel_chain(output_dir: Path, chain_id: str, reason: str = "cancelled-by-law") -> None:
    path = state_path(output_dir, chain_id)
    state = load_state(path)
    done_marker(path).write_text(reason + "\n", encoding="utf-8")
    mark_status(path, reason, exit_code=1)
    cancel_jobs(list(map(str, state.get("jobs", []))))


def query_chain(output_dir: Path, chain_id: str) -> ChainStatus:
    path = state_path(output_dir, chain_id)
    try:
        state = load_state(path)
    except FileNotFoundError:
        return ChainStatus(chain_id, "failed", code=1, error="humsub chain state not found")

    raw_status = str(state.get("status", ""))
    code = state.get("exit_code")
    error = state.get("error")

    if raw_status == "completed":
        return ChainStatus(chain_id, "finished", code=0)

    terminal_failures = (
        done_marker(path).exists()
        or raw_status.startswith("failed")
        or raw_status.startswith("stopped-")
        or raw_status.startswith("cancelled")
        or raw_status.startswith("cleaned-up")
    )
    if terminal_failures:
        return ChainStatus(chain_id, "failed", code=int(code or 1), error=error or raw_status)

    jobs = list(map(str, state.get("jobs", [])))
    statuses = query_jobs(jobs)
    if any(status.running for status in statuses.values()):
        return ChainStatus(chain_id, "running")
    if any(status.pending for status in statuses.values()):
        return ChainStatus(chain_id, "pending")

    # During a handoff, state is updated before the successor starts.  Treat
    # this as running even if accounting visibility briefly lags behind.
    if raw_status.startswith(("running-hop-", "continuing-after-hop-", "retrying-after-failure-hop-")):
        return ChainStatus(chain_id, "running")
    if raw_status in {"submitted", "queued"}:
        # A just-submitted job may not be visible in squeue/sacct immediately.
        return ChainStatus(chain_id, "pending")

    if jobs and statuses and all(status.finished for status in statuses.values()):
        # All inner Slurm jobs completed but the chain worker never recorded a
        # successful terminal state.  This indicates an incomplete/corrupt
        # handoff rather than a successful law job.
        return ChainStatus(
            chain_id,
            "failed",
            code=1,
            error=f"all inner Slurm jobs finished while chain state remained {raw_status!r}",
        )

    if jobs and not statuses:
        return ChainStatus(
            chain_id,
            "failed",
            code=1,
            error="no known inner Slurm job can be found in squeue or sacct",
        )

    return ChainStatus(chain_id, "pending")
