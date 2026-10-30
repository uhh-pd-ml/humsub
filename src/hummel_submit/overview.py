"""Where a submission stands, read from law's jobs files (the source of truth) and the chain states.

law keeps one jobs file per workflow (``<law_dir>/control/slurm_jobs_*.json``; the finalize workflow's under
``control/finalize``): job number -> current chain id, branches, last polled status, and the retries used
(``attempts``).  A resubmitted job points at its new chain, so older chains of the same job are history only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .chain import query_chain
from .cleanup import iter_submission_specs
from .config import ConfigError
from .manifest import load_manifest
from .state import chain_root, load_state
from .submission import load_submission_spec, submission_root


@dataclass
class LawJob:
    num: int
    chain_id: str | None
    branches: list[int]
    attempts: int


@dataclass
class Overview:
    submission_id: str
    total: int
    done: int
    active: int                     # incomplete branches whose current chain is pending/running
    waiting: int                    # incomplete branches law has not submitted yet (e.g. --parallel-jobs)
    missing: list[int]              # incomplete branches whose current chain ended: a pass retries them
    failed_jobs: list[tuple[LawJob, str]] = field(default_factory=list)   # (job, chain error) with incomplete branches
    chain_states: dict[str, int] = field(default_factory=dict)            # every chain ever started, incl. retried ones
    hops: int = 0


def find_submission(output_dir: Path, ident: str) -> Path:
    """Resolve a submission id (or prefix, or unique run name) to its submission.json."""
    direct = submission_root(output_dir) / ident / "submission.json"
    if direct.is_file():
        return direct
    matches = []
    for path in iter_submission_specs(output_dir):
        try:
            spec = load_submission_spec(path)
        except (OSError, json.JSONDecodeError):
            continue
        if spec.get("run_name") == ident or spec["submission_id"].startswith(ident):
            matches.append(path)
    if not matches:
        raise ConfigError(f"no submission or run named {ident!r} under {submission_root(output_dir)}")
    if len(matches) > 1:
        raise ConfigError(f"{ident!r} is ambiguous: " + ", ".join(p.parent.name for p in matches))
    return matches[0]


def submission_manifest(spec: dict[str, Any]) -> dict[str, Any]:
    return load_manifest(Path(spec["manifest_workflow"]["manifest"]))


def jobs_files(spec: dict[str, Any], *, finalize: bool = False) -> list[Path]:
    control = Path(spec.get("law_dir", "")) / "control"
    return sorted((control / "finalize" if finalize else control).glob("slurm_jobs_*.json"))


def jobs_in_file(path: Path) -> tuple[list[LawJob], int]:
    """(submitted jobs, number of branches not submitted yet) of one law jobs file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], 0
    attempts = {int(k): int(v) for k, v in data.get("attempts", {}).items()}
    jobs = [
        LawJob(int(num), job.get("job_id"), [int(b) for b in job.get("branches", [])], attempts.get(int(num), 0))
        for num, job in data.get("jobs", {}).items() if isinstance(job, dict)
    ]
    return jobs, sum(len(b) for b in data.get("unsubmitted_jobs", {}).values())


def law_jobs(spec: dict[str, Any], *, finalize: bool = False) -> tuple[list[LawJob], int]:
    """(submitted jobs, number of branches law has not submitted yet) of the branch (or finalize) workflow."""
    jobs: list[LawJob] = []
    unsubmitted = 0
    for path in jobs_files(spec, finalize=finalize):
        found, waiting = jobs_in_file(path)
        jobs += found
        unsubmitted += waiting
    return jobs, unsubmitted


def compute_overview(output_dir: Path, spec: dict[str, Any]) -> Overview:
    manifest = submission_manifest(spec)
    outputs = {int(b["id"]): b["outputs"] for b in manifest["branches"]}
    done = {bid for bid, outs in outputs.items() if all(Path(o).exists() for o in outs)}

    jobs, _ = law_jobs(spec)
    submitted: set[int] = set()
    active: set[int] = set()
    ended: set[int] = set()
    failed: list[tuple[LawJob, str]] = []
    for job in jobs:
        todo = [b for b in job.branches if b not in done]
        if not job.chain_id:
            continue
        submitted.update(job.branches)
        if not todo:
            continue
        status = query_chain(output_dir, job.chain_id)
        if status.state in {"pending", "running"}:
            active.update(todo)
        else:
            ended.update(todo)
            failed.append((job, status.error or status.state))

    states: dict[str, int] = {}
    hops = 0
    for chain_id in spec.get("chain_ids", []):
        state = query_chain(output_dir, chain_id).state
        states[state] = states.get(state, 0) + 1
        try:
            hops += len(load_state(chain_root(output_dir) / chain_id / "state.json").get("jobs", []))
        except OSError:
            pass
    return Overview(
        submission_id=spec["submission_id"],
        total=len(outputs),
        done=len(done),
        active=len(active),
        waiting=len(set(outputs) - done - submitted),
        missing=sorted(ended),
        failed_jobs=failed,
        chain_states=states,
        hops=hops,
    )


def finalize_state(spec: dict[str, Any]) -> str | None:
    """One line on the manifest's finalize step (None without one), from its outputs and law's jobs file."""
    fin = submission_manifest(spec).get("finalize")
    if not fin:
        return None
    if all(Path(o).exists() for o in fin["outputs"]):
        return "done"
    jobs, _ = law_jobs(spec, finalize=True)
    if not jobs:
        return "waiting for the branches (the first law pass that finds them all done submits it)"
    job = jobs[0]
    status = query_chain(Path(spec["output_dir"]), job.chain_id).state if job.chain_id else "submitted"
    return f"chain {job.chain_id} {status} ({job.attempts} retr{'y' if job.attempts == 1 else 'ies'} used)"


def format_overview(ov: Overview, *, hint: bool = True) -> list[str]:
    counts = f"{ov.total} total | {ov.done} done | {ov.active} in flight"
    if ov.waiting:
        counts += f" | {ov.waiting} not yet submitted"
    lines = [
        f"branches:   {counts} | {len(ov.missing)} failed",
        "chains:     " + (", ".join(f"{n} {s}" for s, n in sorted(ov.chain_states.items())) or "none")
        + f" ({ov.hops} Slurm job(s) used; a retried job's earlier chains count here too)",
    ]
    for job, error in ov.failed_jobs[:20]:
        branches = [b for b in job.branches if b in ov.missing]
        # (no "  <word>: <word>" shape: tools parse the chain lines of submission-status by that pattern)
        lines.append(f"  law job {job.num} failed (chain {job.chain_id}, {error}, {job.attempts} retr"
                     f"{'y' if job.attempts == 1 else 'ies'} used): branches {branches}")
    if len(ov.failed_jobs) > 20:
        lines.append(f"  ... and {len(ov.failed_jobs) - 20} more failed job(s)")
    if ov.missing and hint:
        lines.append(f"next step:  a law pass retries them while retries are left (supervisor or `humsub tick "
                     f"{ov.submission_id}`); after fixing the cause, `humsub resume {ov.submission_id}` grants new retries")
    return lines
