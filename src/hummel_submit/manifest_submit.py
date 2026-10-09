"""Submit a manifest workflow: shared by ``submit-manifest``, ``resume`` and the supervisor job."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
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


@dataclass
class SubmitOptions:
    tasks_per_job: int = 1
    wait: bool = False
    retries: int = 0
    parallel_jobs: int = 0
    retry_payload: int = 0
    safety_margin: float = 0.0
    max_concurrent: int = 0
    resubmit: bool = True
    scratch: str | None = None
    supervisor_job: bool = False
    supervisor_interval: int = 1800
    supervisor_max_rounds: int = 3
    ignore_quota_check: bool = False

    def validate(self) -> None:
        if self.retries < 0 or self.parallel_jobs < 0 or self.retry_payload < 0 or self.max_concurrent < 0:
            raise ConfigError("retries, retry-payload, parallel-jobs and max-concurrent must be >= 0")
        if self.tasks_per_job < 1:
            raise ConfigError("tasks_per_job must be >= 1")
        if self.safety_margin < 0:
            raise ConfigError("--safety-margin must be >= 0")
        if self.safety_margin > 0 and (self.retry_payload or self.retries):
            raise ConfigError(
                "--safety-margin (retry hops queued in Slurm) is mutually exclusive with --retry-payload and --retries: "
                "choose one retry mechanism"
            )
        if self.safety_margin > 0 and not self.resubmit:
            raise ConfigError("--safety-margin needs follower hops; it cannot be combined with --no-resubmit")
        if not self.wait and self.retries:
            raise ConfigError("--retries requires --wait; non-polling law cannot perform controller-side retries")
        if self.scratch not in (None, "beegfs", "ssd"):
            raise ConfigError("--scratch must be 'beegfs' or 'ssd'")
        if self.supervisor_job and (self.supervisor_interval < 60 or self.supervisor_max_rounds < 1):
            raise ConfigError("--supervisor-interval must be >= 60 s and --supervisor-rounds >= 1")

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
        lines.append("[submit] concurrency not capped (use --max-concurrent N or --wait --parallel-jobs N)")
    if opts.retry_payload:
        lines.append(f"[submit] retry       payload up to {opts.retry_payload}x inside a job (time-aware)")
    if opts.safety_margin:
        lines.append(f"[submit] retry       safety margin {opts.safety_margin}: {opts.failure_budget()} extra retry hop(s) per chain")
    if opts.retries:
        lines.append(f"[submit] retry       law retries {opts.retries} (controller must stay alive)")
    if opts.supervisor_job:
        lines.append(f"[submit] supervisor  every {opts.supervisor_interval}s, at most {opts.supervisor_max_rounds} resume round(s)")
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
    lineage: dict[str, Any] | None = None,
    parent_submission: str | None = None,
    allow_supervisor: bool = True,
) -> dict[str, Any] | None:
    """Freeze and submit a manifest workflow; returns the submission spec (``None`` for dry runs/no-ops)."""
    opts.validate()
    check_signal_window(config, resubmit=opts.resubmit)
    output_dir = Path(config["execution"]["output_dir"])
    cache_dir = Path(config["execution"]["cache_dir"])
    manifest = load_manifest(manifest_source)
    branches = len(manifest["branches"])

    if not opts.wait and opts.parallel_jobs and branches > opts.parallel_jobs:
        raise ConfigError(
            "manifest contains more branches than --parallel-jobs, but --wait was not set; "
            "use --wait for rolling submission or --parallel-jobs=0 to submit all branches immediately"
        )

    if not skip_path_checks:
        check_compute_writable(output_dir, "execution.output_dir", must_be_shared=True)
        check_compute_writable(cache_dir, "execution.cache_dir")
        for branch in manifest["branches"]:
            for output in branch["outputs"]:
                check_compute_writable(Path(output), f"branch {branch['id']} output", must_be_shared=True)
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
                "parent_submission": parent_submission,
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
        spec["lineage"] = lineage or {
            "id": spec["submission_id"],
            "manifest": spec["manifest_workflow"]["manifest"],
            "payload": spec["manifest_workflow"]["payload"],
        }
        atomic_write_json(spec_path, spec)

        try:
            import luigi
            from .manifest_workflow import ManifestPayloadWorkflow
        except ImportError as exc:
            raise ConfigError(
                "law (master) is required for manifest submission; reinstall hummel-submit with its dependencies"
            ) from exc

        task = ManifestPayloadWorkflow(
            humsub_spec=str(spec_path),
            workflow="slurm",
            no_poll=not opts.wait,
            retries=opts.retries,
            tasks_per_job=opts.tasks_per_job,
            parallel_jobs=opts.parallel_jobs,
            job_workers=1,
        )
        success = luigi.build([task], local_scheduler=True, workers=1)
        if not success:
            raise ConfigError("law failed to prepare/submit the manifest workflow")
        spec = load_submission_spec(spec_path)
        if not spec.get("chain_ids") and all_outputs_exist(manifest):
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

    print_manifest_submission(spec, output_dir, wait=opts.wait)
    if opts.supervisor_job and allow_supervisor:
        from .supervisor import submit_supervisor
        job_id = submit_supervisor(spec_path, round_no=1)
        print(f"[submit] supervisor  job {job_id} (first check in {opts.supervisor_interval}s)")
    return spec
