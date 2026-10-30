"""``humsub resume``: continue a submission after its law passes gave up (or with changed settings).

Everything stays in the same submission and the same law workflow; law's passes do the actual resubmitting.  Resume

* applies changes for future chains: Slurm settings (e.g. a longer ``--time``), ``--retries``, ``--parallel-jobs``,
  and stages given with ``--stage NAME=PATH`` (e.g. a renewed proxy; stages removed by ``humsub cleanup`` are restored
  from their recorded sources, all others stay frozen);
* grants every failed law job (branch or finalize workflow) a fresh retry budget (law's ``attempts`` reset to 0);
* removes the scratch of Slurm jobs that are gone, runs one pass right away (which resubmits the failed jobs), and
  queues the supervisor again if the submission is supervised and none is alive.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .config import ConfigError, validate_config
from .overview import compute_overview, format_overview, jobs_files, jobs_in_file, submission_manifest
from .chain import query_chain
from .scratch import sweep_stale_scratch
from .staging import stage_inputs
from .state import atomic_write_json
from .submission import load_submission_spec, update_submission_spec
from .locks import ControlLock, LockBusy
from .tick import TickResult, run_tick


def _failed_jobs(output_dir: Path, spec: dict[str, Any]) -> dict[Path, list[int]]:
    """law job numbers, per jobs file, whose chain ended while their outputs are still missing."""
    manifest = submission_manifest(spec)
    outputs = {int(b["id"]): b["outputs"] for b in manifest["branches"]}
    found: dict[Path, list[int]] = {}
    for finalize in (False, True):
        if finalize and not manifest.get("finalize"):
            continue
        for path in jobs_files(spec, finalize=finalize):
            for job in jobs_in_file(path)[0]:
                outs = manifest["finalize"]["outputs"] if finalize else [o for b in job.branches for o in outputs[b]]
                if not job.chain_id or all(Path(o).exists() for o in outs):
                    continue
                if query_chain(output_dir, job.chain_id).state in {"finished", "failed"}:
                    found.setdefault(path, []).append(job.num)
    return found


def _reset_attempts(path: Path, nums: list[int]) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    attempts = data.setdefault("attempts", {})
    for num in nums:
        attempts[str(num)] = 0
    atomic_write_json(path, data)


def resume(
    output_dir: Path,
    spec_path: Path,
    *,
    config_overrides: dict[str, dict[str, Any]] | None = None,
    retries: int | None = None,
    parallel_jobs: int | None = None,
    stage_overrides: dict[str, Path] | None = None,
    supervisor: bool = False,
    supervisor_interval: int | None = None,
    dry_run: bool = False,
) -> TickResult | None:
    spec = load_submission_spec(spec_path)
    if "law_task" not in spec:
        raise ConfigError("this submission predates law passes (humsub < 0.5); resubmit its manifest instead")
    for line in format_overview(compute_overview(output_dir, spec), hint=False):
        print(f"[resume] {line}")

    config = copy.deepcopy(spec["config"])
    for section, values in (config_overrides or {}).items():
        config.setdefault(section, {}).update(values)
    validate_config(config, require_command=False)

    workflow = spec["manifest_workflow"]
    sources = {k: Path(v) for k, v in workflow.get("stage_sources", {}).items()}
    unknown = sorted(set(stage_overrides or {}) - set(sources))
    if unknown:
        raise ConfigError(f"unknown stage(s) {', '.join(unknown)}; the submission has: {', '.join(sorted(sources)) or 'none'}")
    sources.update(stage_overrides or {})
    restage = {n: p for n, p in sources.items()
               if n in (stage_overrides or {}) or not Path(workflow["stages"][n]).exists()}

    from .manifest_submit import check_concurrency, min_supervisor_interval

    if parallel_jobs is not None:
        check_concurrency(parallel_jobs, int(spec.get("max_concurrent", 0) or 0))
    if supervisor_interval is not None and supervisor_interval < min_supervisor_interval():
        raise ConfigError(f"--supervisor-interval must be >= {min_supervisor_interval()} s")
    law_retries = int(spec["law_task"]["params"].get("retries", 0)) if retries is None else retries
    if law_retries < 1:
        law_retries = 1  # a reset budget of 0 retries would resubmit nothing
    failed = _failed_jobs(output_dir, spec)
    n_failed = sum(len(v) for v in failed.values())

    print(f"[resume] {n_failed} failed law job(s) get a fresh budget of {law_retries} retr{'y' if law_retries == 1 else 'ies'}")
    for name, path in restage.items():
        print(f"[resume] stage {name} re-copied from {path}")
    if config_overrides:
        print(f"[resume] settings for new chains: {json.dumps(config_overrides)}")
    swept, swept_bytes = sweep_stale_scratch(config, {spec["submission_id"]}, apply=not dry_run)
    if swept:
        print(f"[resume] {'would remove' if dry_run else 'removed'} scratch of {swept} dead job(s) ({swept_bytes / 1e6:.0f} MB)")
    if dry_run:
        return None

    def change(data: dict[str, Any]) -> None:
        data["config"] = config
        data["manifest_workflow"]["stage_sources"] = {k: str(v) for k, v in sources.items()}
        tasks = [data["law_task"]] + ([data["manifest_workflow"]["branch_task"]]
                                      if "branch_task" in data["manifest_workflow"] else [])
        for task in tasks:
            task["params"]["retries"] = law_retries
        if parallel_jobs is not None:
            tasks[-1]["params"]["parallel_jobs"] = parallel_jobs  # the branch workflow
        opts = data.setdefault("submit_options", {})
        if supervisor:
            opts["supervisor_job"] = True
        if supervisor_interval:
            opts["supervisor_interval"] = supervisor_interval

    # Everything that changes the submission, runs a pass or queues a supervisor happens under the control lock: no
    # supervisor pass runs meanwhile, and no second supervisor can be queued next to a running one.
    lock = ControlLock(spec_path.parent, "humsub resume")
    try:
        lock.acquire()
    except LockBusy as exc:
        raise ConfigError(f"{exc}; try again when it is done") from exc
    try:
        failed = _failed_jobs(output_dir, spec)      # again: a supervisor pass may have ended just before we locked
        if restage:
            stage_inputs(restage, cache_dir=Path(config["execution"]["cache_dir"]), submission_id=spec["submission_id"],
                         excludes={n: e for n, e in workflow.get("stage_excludes", {}).items() if n in restage})
        spec = update_submission_spec(spec_path, change)
        for path, nums in failed.items():
            _reset_attempts(path, nums)

        result = run_tick(spec_path, "humsub resume")
        print(f"[resume] pass: {result.state}")
        for error in result.errors:
            print(f"[resume]   {error}")

        if spec.get("submit_options", {}).get("supervisor_job") and result.state != "complete":
            from .slurm import live_job_ids
            from .supervisor import submit_supervisor

            alive = live_job_ids() & {str(j) for j in spec.get("supervisor_job_ids", [])}
            if alive:
                print(f"[resume] supervisor job(s) {', '.join(sorted(alive))} still queued; it picks the changes up")
            else:
                print(f"[resume] supervisor queued again as job {submit_supervisor(spec_path, round_no=1)}")
        return result
    finally:
        lock.release()
