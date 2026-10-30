"""Submit a manifest workflow (``submit-manifest``); law passes (:mod:`hummel_submit.tick`) take it from there."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

from .config import ConfigError, check_signal_window
from .manifest import freeze_manifest_workflow, load_manifest
from .pathcheck import check_compute_writable
from .quota import check_fast_quota
from .scratch import effective_scratch_kind
from .staging import stage_inputs
from .state import atomic_write_json, chain_root
from .submission import create_submission_spec, load_submission_spec
from .cleanup import OWNER_MARKER
from .tick import manifest_law_task


def min_supervisor_interval() -> int:
    """60 s: a supervisor job per minute is the most a production should cost the scheduler.  The end-to-end tests set
    HUMSUB_MIN_SUPERVISOR_INTERVAL to run passes every few seconds (Slurm adds ~15 s to a delayed start anyway)."""
    try:
        return max(1, int(os.environ.get("HUMSUB_MIN_SUPERVISOR_INTERVAL", "60")))
    except ValueError:
        return 60


def check_concurrency(parallel_jobs: int, max_concurrent: int) -> None:
    """--parallel-jobs J limits how many chains exist in Slurm (law, per pass), --max-concurrent N how many run
    (Slurm dependencies, at once).  Together they only make sense with J > N; otherwise the lanes never hold back."""
    if parallel_jobs and max_concurrent and parallel_jobs <= max_concurrent:
        raise ConfigError(
            f"--parallel-jobs {parallel_jobs} <= --max-concurrent {max_concurrent}: --parallel-jobs limits how many "
            "chains are in the queue, --max-concurrent how many of them run, so the lanes would never hold anything "
            "back.  Use only one of them, or a --parallel-jobs well above --max-concurrent (see README: Concurrency)"
        )


@dataclass
class SubmitOptions:
    tasks_per_job: int = 1
    wait: bool = False
    retries: int | None = None      # None: not given (the supervisor then defaults to supervisor_max_rounds)
    parallel_jobs: int = 0
    retry_payload: int = 0
    safety_margin: float = 0.0
    max_concurrent: int = 0
    resubmit: bool = True
    scratch: str | None = None
    supervisor_job: bool = False
    supervisor_interval: int = 1800
    supervisor_trigger_delay: int = 60      # a chain's end brings the next pass forward to this many seconds
    supervisor_max_rounds: int = 3
    ignore_quota_check: bool = False

    def validate(self) -> None:
        retries = self.retries or 0
        if retries < 0 or self.parallel_jobs < 0 or self.retry_payload < 0 or self.max_concurrent < 0:
            raise ConfigError("retries, retry-payload, parallel-jobs and max-concurrent must be >= 0")
        if self.tasks_per_job < 1:
            raise ConfigError("tasks_per_job must be >= 1")
        if self.safety_margin < 0:
            raise ConfigError("--safety-margin must be >= 0")
        if self.safety_margin > 0 and self.retry_payload:
            raise ConfigError(
                "--safety-margin (retry hops queued in Slurm) is mutually exclusive with --retry-payload: "
                "choose one in-chain retry mechanism (law --retries can be combined with either)"
            )
        if self.safety_margin > 0 and not self.resubmit:
            raise ConfigError("--safety-margin needs follower hops; it cannot be combined with --no-resubmit")
        if not self.wait and not self.supervisor_job and retries:
            raise ConfigError(
                "--retries requires --wait or --supervisor-job: something must run law again to resubmit failed jobs"
            )
        check_concurrency(self.parallel_jobs, self.max_concurrent)
        if self.scratch not in (None, "beegfs", "ssd"):
            raise ConfigError("--scratch must be 'beegfs' or 'ssd'")
        floor = min_supervisor_interval()
        if self.supervisor_job and (self.supervisor_interval < floor or self.supervisor_max_rounds < 1):
            raise ConfigError(f"--supervisor-interval must be >= {floor} s and --supervisor-rounds >= 1")
        if self.supervisor_job and self.supervisor_trigger_delay < floor:
            raise ConfigError(f"--supervisor-trigger-delay must be >= {floor} s")

    def law_retries(self) -> int:
        """law's resubmissions per failed job.  The supervisor's passes let law retry (``--supervisor-rounds`` times
        unless ``--retries`` says otherwise).  They nest with the in-chain mechanisms: law only sees a chain fail
        once its --safety-margin hops (or --retry-payload attempts) are used up, and then starts a new chain."""
        if self.retries is not None or self.wait or not self.supervisor_job:
            return self.retries or 0
        return self.supervisor_max_rounds

    def failure_budget(self) -> int:
        return 0 if self.safety_margin <= 0 else max(1, math.ceil(self.safety_margin * self.tasks_per_job))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubmitOptions":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def all_outputs_exist(manifest: dict[str, Any]) -> bool:
    return all(Path(output).exists() for branch in manifest["branches"] for output in branch["outputs"])


def discard_preparation(staged_root: Path | None, spec_path: Path | None, spec: dict[str, Any] | None) -> None:
    """Remove the frozen submission, staged inputs and run directory of a submission that started no chain."""
    if staged_root is not None:
        shutil.rmtree(staged_root, ignore_errors=True)
    if spec_path is not None:
        shutil.rmtree(spec_path.parent, ignore_errors=True)
    if spec is not None:
        shutil.rmtree(Path(spec["run_dir"]), ignore_errors=True)


def print_manifest_submission(spec: dict[str, Any], output_dir: Path, *, wait: bool) -> None:
    print(f"[submit] submission  {spec['submission_id']}")
    print(f"[submit] run         {spec['run_name']}")
    print(f"[submit] run dir     {spec['run_dir']}")
    chain_ids = list(spec.get("chain_ids", []))
    if not chain_ids:
        raise ConfigError("law returned successfully but the Hummel backend recorded no chain id")
    for chain_id in chain_ids:
        print(f"[submit] chain       {chain_id}")
        print(f"[submit] state       {chain_root(output_dir) / chain_id / 'state.json'}")
    if not wait:
        print("[submit] law returned after submission; autonomous chains continue independently")


def describe_policy(config: dict[str, Any], opts: SubmitOptions, scratch_kind: str) -> list[str]:
    nice = config["slurm"]["nice"]
    lines = [
        f"[submit] nice        {nice}" + ("" if nice > 0 else "  (priority penalty explicitly disabled)"),
        f"[submit] scratch     bulk on {scratch_kind}, fast on ssd",
        f"[submit] packing     {opts.tasks_per_job} branch(es) per chain",
    ]
    if opts.max_concurrent:
        lines.append(f"[submit] concurrency at most {opts.max_concurrent} chain lane(s) (continuation hops may overlap briefly)")
    else:
        lines.append("[submit] concurrency not capped (use --max-concurrent N or --parallel-jobs N with --supervisor-job)")
    if opts.retry_payload:
        lines.append(f"[submit] retry       payload up to {opts.retry_payload}x inside a job (time-aware)")
    if opts.safety_margin:
        lines.append(f"[submit] retry       safety margin {opts.safety_margin}: {opts.failure_budget()} extra retry hop(s) per chain")
    retries = opts.law_retries()
    if retries:
        driver = "the waiting controller" if opts.wait else "the supervisor's law passes"
        lines.append(f"[submit] retry       law resubmits a failed job up to {retries}x ({driver})")
    if opts.supervisor_job:
        lines.append(f"[submit] supervisor  a law pass every {opts.supervisor_interval}s until done (or no retries left); "
                     f"a chain's end brings the next one forward to {min(opts.supervisor_trigger_delay, opts.supervisor_interval)}s")
    return lines


def submit_manifest_run(
    *,
    project_dir: Path,
    config: dict[str, Any],
    sources: list[Path],
    run_name: str,
    manifest_source: Path,
    payload_source: Path,
    stages: dict[str, Path],
    stage_excludes: dict[str, list[str]],
    opts: SubmitOptions,
    skip_path_checks: bool = False,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Freeze and submit a manifest workflow; returns the submission spec (``None`` for dry runs/no-ops)."""
    manifest = load_manifest(manifest_source)
    branches = len(manifest["branches"])
    finalize = manifest.get("finalize")
    if finalize and not opts.wait and not opts.supervisor_job:
        # the finalize workflow is submitted by a law pass once every branch is done: something must run those passes
        # (switched on before validating: e.g. --retries is valid because of it)
        opts.supervisor_job = True
        print("[submit] note: the manifest has a finalize step, so the supervisor job (law passes) is switched on")
    opts.validate()
    check_signal_window(config, resubmit=opts.resubmit)
    output_dir = Path(config["execution"]["output_dir"])
    cache_dir = Path(config["execution"]["cache_dir"])

    if not opts.wait and not opts.supervisor_job and opts.parallel_jobs and branches > opts.parallel_jobs:
        raise ConfigError(
            "manifest contains more branches than --parallel-jobs, but neither --wait nor --supervisor-job was set; "
            "use one of them for rolling submission or --parallel-jobs=0 to submit all branches immediately"
        )

    if not skip_path_checks:
        check_compute_writable(output_dir, "execution.output_dir", must_be_shared=True)
        check_compute_writable(cache_dir, "execution.cache_dir")
        for branch in manifest["branches"]:
            for output in branch["outputs"]:
                check_compute_writable(Path(output), f"branch {branch['id']} output", must_be_shared=True)
            for output in branch.get("extra_outputs", []):
                check_compute_writable(Path(output), f"branch {branch['id']} extra output", must_be_shared=True)
        for output in (finalize or {}).get("outputs", []):
            check_compute_writable(Path(output), "finalize output", must_be_shared=True)
    else:
        print("[submit] WARNING: filesystem/path validation disabled by --skip-path-checks", file=sys.stderr)

    hint = manifest.get("humsub") or {}
    scratch_kind = effective_scratch_kind(config, hint, opts.scratch)
    quota_line = None
    if not opts.ignore_quota_check:
        quota_line = check_fast_quota(
            config,
            branches=branches,
            tasks_per_job=opts.tasks_per_job,
            max_concurrent=opts.max_concurrent,
            wait=opts.wait,
            parallel_jobs=opts.parallel_jobs,
            scratch_kind=scratch_kind,
            scratch_bytes_per_branch=int(hint.get("scratch_bytes_per_branch", 0)),
        )

    print(f"[submit] manifest    {manifest_source}")
    print(f"[submit] payload     {payload_source}")
    print(f"[submit] branches    {branches}")
    print(f"[submit] run         {run_name}")
    print(f"[submit] output      {output_dir}")
    for name, path in stages.items():
        print(f"[submit] stage       {name}={path}")
        for pattern in stage_excludes.get(name, []):
            print(f"[submit] exclude     {name}={pattern}")
    if sources:
        print(f"[submit] config      {', '.join(map(str, sources))}")
    for line in describe_policy(config, opts, scratch_kind):
        print(line)
    if finalize:
        print(f"[submit] finalize    a law workflow after all branches ({len(finalize['outputs'])} output(s)), "
              "submitted by the first pass that finds every branch done")
    if quota_line:
        print(f"[submit] {quota_line}")
    if dry_run:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    spec: dict[str, Any] | None = None
    spec_path: Path | None = None
    staged_root: Path | None = None
    try:
        spec, spec_path = create_submission_spec(
            output_dir=output_dir,
            project_dir=project_dir,
            config=config,
            run_name=run_name,
            user_args=[],
            resubmit=opts.resubmit,
            python_executable=sys.executable,
            extra={
                "max_concurrent": opts.max_concurrent,
                "failure_budget": opts.failure_budget(),
                "submit_options": opts.to_dict(),
            },
        )
        staged_root = cache_dir / "stages" / spec["submission_id"]
        frozen_stages = stage_inputs(
            stages, cache_dir=cache_dir, submission_id=spec["submission_id"], excludes=stage_excludes
        )
        # gc only knows the submissions of the *current* project's output_dir, but cache_dir is
        # shared by default; record the owner so gc never treats another project's staging as orphan.
        staged_root.mkdir(parents=True, exist_ok=True)
        (staged_root / OWNER_MARKER).write_text(str(spec_path) + "\n", encoding="utf-8")
        spec = freeze_manifest_workflow(
            spec_path,
            manifest_source=manifest_source,
            payload_source=payload_source,
            stages=frozen_stages,
            workflow_extra={
                "retry_payload": opts.retry_payload,
                "scratch": opts.scratch,
                "stage_sources": {k: str(v) for k, v in stages.items()},
                "stage_excludes": stage_excludes,
            },
        )
        # The law tasks: the branch workflow and the top task that every supervisor pass re-instantiates identically
        # (hummel_submit.tick); with a finalize step the top task is the finalize workflow, which requires the branches.
        law_opts = dict(opts.to_dict(), retries=opts.law_retries())
        branch_task = manifest_law_task(spec_path, law_opts)
        spec["manifest_workflow"]["branch_task"] = branch_task
        spec["law_task"] = (
            manifest_law_task(spec_path, dict(law_opts, tasks_per_job=1, parallel_jobs=0), "ManifestFinalizeWorkflow")
            if finalize else branch_task
        )
        atomic_write_json(spec_path, spec)

        try:
            import luigi
            from .tick import instantiate
        except ImportError as exc:
            raise ConfigError(
                "law (master) is required for manifest submission; reinstall hummel-submit with its dependencies"
            ) from exc

        # Without --wait only the branches can start now; the passes take it from there.  A waiting controller
        # polls through the whole graph (finalize included).
        task = instantiate(spec["law_task"] if opts.wait else branch_task)
        task.no_poll = not opts.wait
        success = luigi.build([task], local_scheduler=True, workers=1)
        if not success:
            raise ConfigError("law failed to prepare/submit the manifest workflow")
        spec = load_submission_spec(spec_path)
        finalize_pending = bool(finalize) and not all(Path(o).exists() for o in finalize["outputs"])
        if not spec.get("chain_ids") and all_outputs_exist(manifest) and not finalize_pending:
            discard_preparation(staged_root, spec_path, spec)
            print(f"[submit] all {branches} branches already have their outputs; nothing to submit")
            return None
    except Exception:
        has_chains = False
        if spec_path and spec_path.exists():
            try:
                has_chains = bool(load_submission_spec(spec_path).get("chain_ids"))
            except Exception:
                pass
        if not has_chains:
            discard_preparation(staged_root, spec_path, spec)
        raise

    if spec.get("chain_ids") or not finalize:
        print_manifest_submission(spec, output_dir, wait=opts.wait)
    else:
        print(f"[submit] submission  {spec['submission_id']}")
        print(f"[submit] all {branches} branches already have their outputs; only the finalize step is left")
    if opts.supervisor_job:
        from .locks import control_lock
        from .supervisor import submit_supervisor
        with control_lock(spec_path, "submit-manifest"):     # supervisors are only ever queued under the lock
            job_id = submit_supervisor(spec_path, round_no=1)
        print(f"[submit] supervisor  job {job_id} (first check in {opts.supervisor_interval}s)")
    return spec
