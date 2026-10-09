from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import shutil
import sys
from typing import Any

from . import __version__
from .chain import cancel_chain, query_chain
from .follow import follow_chain
from .cleanup import (
    cleanup_submission_cache,
    inspect_submission_cache,
    iter_submission_specs,
    remove_cache_paths,
    OWNER_MARKER,
    stale_orphan_cache_dirs,
)
from .config import (
    ConfigError,
    PROJECT_CONFIG_NAME,
    USER_CONFIG,
    check_signal_window,
    config_as_json,
    load_config,
    validate_run_name,
)
from .pathcheck import PathCheckError, check_compute_writable, check_payload_args
from .slurm import SlurmError, queue_status, sbatch_command, slurm_log_path
from .state import chain_root, default_run_name, load_state
from .submission import create_submission_spec, load_submission_spec, submission_root
from .manifest import load_manifest
from .manifest_submit import (
    SubmitOptions,
    all_outputs_exist as _all_outputs_exist,
    discard_preparation as _discard_preparation,
    print_manifest_submission as _print_manifest_submission,
    submit_manifest_run,
)
from .lineage import (
    compute_overview,
    find_submission,
    format_overview,
    resume as resume_lineage,
    sweep_stale_scratch,
)
from .staging import parse_stage_exclude, parse_stage_spec, stage_inputs
from .templates import PROJECT_TEMPLATE, USER_TEMPLATE


def _add_slurm_options(parser: argparse.ArgumentParser) -> None:
    """Per-submission overrides of [slurm] settings (shared by submit and submit-manifest)."""
    parser.add_argument("--no-resubmit", action="store_true", help="no continuation hops: the job runs once, in a single Slurm job")
    parser.add_argument("--time", dest="time_limit", help="time limit of each hop, e.g. 04:00:00")
    parser.add_argument("--account", help="Slurm account")
    parser.add_argument("--partition", help="Slurm partition")
    parser.add_argument("--reservation", help="Slurm reservation")
    parser.add_argument("--mail", help="e-mail address for failure notifications")
    parser.add_argument("--max-hops", type=int, help="maximum number of Slurm jobs (hops) in one chain")
    parser.add_argument("--signal-seconds", type=int, dest="signal_seconds",
                        help="the soft-stop notice comes this many seconds before the time limit of a hop")
    parser.add_argument("--grace-seconds", type=int, dest="grace_seconds",
                        help="seconds between the soft-stop notice and the hard SIGTERM (-1 automatic, 0 = no soft stop)")
    parser.add_argument(
        "--nice", type=int, dest="nice",
        help="Slurm --nice of every chain job incl. continuation hops (default [slurm].nice = 1000000: other users' jobs "
             "always go first, so chains fill idle time and may wait long between hops - intended). "
             "--nice 0 deliberately removes the penalty when you need results now",
    )
    parser.add_argument("--sbatch-arg", action="append", default=[], help="extra sbatch option (repeatable), e.g. --sbatch-arg=--cpus-per-task=8")
    parser.add_argument("--skip-path-checks", action="store_true", help="skip Hummel filesystem/path validation (escape hatch)")


def _add_manifest_policy_options(parser: argparse.ArgumentParser) -> None:
    """Retry, concurrency, scratch and supervision policy of a manifest submission (also used by resume)."""
    parser.add_argument(
        "--retry-payload", type=int, default=None, metavar="N",
        help="re-run a failing payload up to N times inside the job; a retry is skipped when it cannot finish "
             "in the time left of the hop",
    )
    parser.add_argument(
        "--safety-margin", type=float, default=None, metavar="F",
        help="queue retry hops for failed chains: max(1, ceil(F x tasks-per-job)) extra hops per chain, "
             "no controller needed; mutually exclusive with --retry-payload and --retries",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=None, metavar="N",
        help="at most N chains in flight (round-robin lanes enforced by Slurm dependencies, no controller needed)",
    )
    parser.add_argument(
        "--scratch", choices=["beegfs", "ssd"], default=None,
        help="location of HUMSUB_SCRATCH (bulk per-branch scratch); default [execution].scratch = beegfs",
    )
    parser.add_argument("--supervisor-job", action="store_true", default=None,
                        help="queue a tiny Slurm job that periodically resumes missing branches (no login-node process)")
    parser.add_argument("--supervisor-interval", type=int, default=None, metavar="SECONDS",
                        help="seconds between supervisor checks (default 1800)")
    parser.add_argument("--supervisor-rounds", type=int, default=None, metavar="R",
                        help="maximum resume rounds of the supervisor (default 3)")
    parser.add_argument("--ignore-quota-check", action="store_true", default=None,
                        help="skip the preflight estimate of the SSD footprint")


def _option_overrides(args: argparse.Namespace) -> dict[str, Any]:
    mapping = {
        "retry_payload": "retry_payload",
        "safety_margin": "safety_margin",
        "max_concurrent": "max_concurrent",
        "scratch": "scratch",
        "supervisor_job": "supervisor_job",
        "supervisor_interval": "supervisor_interval",
        "supervisor_rounds": "supervisor_max_rounds",
        "ignore_quota_check": "ignore_quota_check",
        "tasks_per_job": "tasks_per_job",
        "wait": "wait",
        "retries": "retries",
        "parallel_jobs": "parallel_jobs",
    }
    out: dict[str, Any] = {}
    for attr, key in mapping.items():
        value = getattr(args, attr, None)
        if value is not None and value is not False:
            out[key] = value
    if getattr(args, "no_resubmit", False):
        out["resubmit"] = False
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="humsub",
        description="law-backed Hummel-2 submission helper: run a command or a manifest of independent branches "
        "as autonomous Slurm job chains (see README.md)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    init = sub.add_parser("init", help="create editable user and project config templates without overwriting existing files")
    mode = init.add_mutually_exclusive_group()
    mode.add_argument("--project-only", action="store_true")
    mode.add_argument("--user-only", action="store_true")

    show = sub.add_parser("config", help="show the resolved effective configuration")
    show.add_argument("--json", action="store_true", help="print machine-readable JSON")

    submit_p = sub.add_parser("submit", help="submit a law workflow for the configured payload")
    submit_p.add_argument("--dry-run", action="store_true", help="validate and print the sbatch command without submitting")
    submit_p.add_argument("--run-name", help="unique run name (default: run-<timestamp>-<id>); names the run directory")
    _add_slurm_options(submit_p)
    submit_p.add_argument("args", nargs=argparse.REMAINDER, help="arguments passed to [execution].command (after --)")

    manifest = sub.add_parser(
        "submit-manifest",
        help="submit a generic manifest of independent branches to an executable payload",
    )
    manifest.add_argument("--manifest", type=Path, required=True, help="JSON manifest (schema 1)")
    manifest.add_argument("--payload", type=Path, required=True, help="executable invoked once per branch")
    manifest.add_argument(
        "--stage", action="append", default=[], metavar="NAME=PATH",
        help="freeze a file or directory for all branches; repeat as needed",
    )
    manifest.add_argument(
        "--stage-exclude", action="append", default=[], metavar="NAME=PATTERN",
        help="exclude a pattern while freezing a directory stage; repeat as needed",
    )
    manifest.add_argument("--run-name", help="unique run name (default: run-<timestamp>-<id>)")
    manifest.add_argument("--wait", action="store_true", help="keep law alive to poll and retry remote jobs")
    manifest.add_argument("--retries", type=int, default=0, help="law failure retries per branch (requires --wait)")
    manifest.add_argument(
        "--tasks-per-job", type=int, default=1,
        help="branches run one after another in one Slurm chain (default 1: one chain per branch)",
    )
    manifest.add_argument(
        "--parallel-jobs", type=int, default=0,
        help="maximum chains active at once (needs --wait); 0 = submit all immediately (default)",
    )
    manifest.add_argument("--dry-run", action="store_true", help="validate the manifest, stages and paths; submit nothing")
    _add_slurm_options(manifest)
    _add_manifest_policy_options(manifest)

    sub.add_parser("help", help="show this help message")

    status = sub.add_parser("status", help="show saved chain state, queue state and log paths of one chain")
    status.add_argument("chain", help="chain id (or any of its Slurm job ids)")

    follow = sub.add_parser("follow", help="follow the active chain log across continuation hops")
    follow.add_argument("chain", help="chain id (or any of its Slurm job ids)")
    follow.add_argument("-n", "--lines", type=int, default=10, help="initial lines to show from the current log (default: 10)")
    follow.add_argument("--poll-interval", type=float, default=1.0, help=argparse.SUPPRESS)

    resume = sub.add_parser(
        "resume",
        help="re-submit only the branches of a submission (or its resumes) whose outputs are still missing",
    )
    resume.add_argument("submission", help="submission id or run name of any submission of the lineage")
    resume.add_argument("--dry-run", action="store_true", help="show what would be resubmitted; submit and delete nothing")
    resume.add_argument("--include-active", action="store_true",
                        help="also resubmit branches owned by chains that are still pending/running (normally skipped)")
    resume.add_argument("--stage", action="append", default=[], metavar="NAME=PATH",
                        help="override a stage source (e.g. a renewed proxy); stages are re-copied from their sources anyway")
    resume.add_argument("--wait", action="store_true", default=None, help="keep law alive to poll and retry")
    resume.add_argument("--retries", type=int, default=None, help="law retries (requires --wait)")
    resume.add_argument("--tasks-per-job", type=int, default=None, help="branches per chain for the resubmission")
    resume.add_argument("--parallel-jobs", type=int, default=None, help="rolling submission width (needs --wait)")
    _add_slurm_options(resume)
    _add_manifest_policy_options(resume)

    submission_status = sub.add_parser(
        "submission-status", help="summarize all autonomous chains belonging to one submission"
    )
    submission_status.add_argument("submission", help="submission id printed by submit-manifest")

    cleanup = sub.add_parser(
        "cleanup",
        help="remove SSD runtime caches for one submission while preserving persistent state and logs",
    )
    cleanup.add_argument("submission", help="submission id")
    cleanup.add_argument("--dry-run", action="store_true", help="only report what would be removed")
    cleanup.add_argument("--force", action="store_true", help="allow cleanup even when chains are still active")

    gc = sub.add_parser("gc", help="garbage-collect old SSD submission caches")
    gc.add_argument("--apply", action="store_true", help="actually delete; without this flag gc is a dry run")
    gc.add_argument("--stale-scratch", action="store_true",
                    help="also remove branch scratch (SSD and bulk) of Slurm jobs that no longer exist")
    gc.add_argument("--successful-after-hours", type=float, default=24.0)
    gc.add_argument("--failed-after-hours", type=float, default=168.0)
    gc.add_argument(
        "--orphans-after-hours", type=float, default=None,
        help="also delete unreferenced cache directories older than this many hours",
    )

    cancel = sub.add_parser("cancel", help="cancel a chain: scancel all its Slurm jobs and mark it terminal")
    cancel.add_argument("chain", help="chain id (or any of its Slurm job ids)")

    return parser


def _write_if_missing(path: Path, content: str) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def cmd_init(args: argparse.Namespace) -> int:
    project = Path.cwd() / PROJECT_CONFIG_NAME
    do_project = not args.user_only
    do_user = not args.project_only
    if do_project:
        print(("created " if _write_if_missing(project, PROJECT_TEMPLATE) else "exists  ") + str(project))
    if do_user:
        print(("created " if _write_if_missing(USER_CONFIG, USER_TEMPLATE) else "exists  ") + str(USER_CONFIG))
    return 0


def _cli_overrides(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    slurm: dict[str, Any] = {}
    for attr in ("account", "partition", "reservation", "mail", "max_hops", "nice", "signal_seconds", "grace_seconds"):
        value = getattr(args, attr, None)
        if value is not None:
            slurm[attr] = value
    if getattr(args, "time_limit", None) is not None:
        slurm["time_limit"] = args.time_limit
    if getattr(args, "sbatch_arg", None):
        slurm["extra_args"] = args.sbatch_arg
    out: dict[str, dict[str, Any]] = {"slurm": slurm} if slurm else {}
    if getattr(args, "scratch", None):
        out["execution"] = {"scratch": args.scratch}
    return out


def _print_config(config: dict[str, Any], sources: list[Path]) -> None:
    print("configuration sources (low -> high precedence):")
    if sources:
        for path in sources:
            print(f"  {path}")
    else:
        print("  built-in defaults only")
    print(config_as_json(config))


def cmd_config(args: argparse.Namespace) -> int:
    config, sources = load_config(Path.cwd(), require_command=False)
    if args.json:
        print(config_as_json(config))
    else:
        _print_config(config, sources)
    return 0


def _prepare_submission(args: argparse.Namespace, dry_run: bool) -> tuple[dict[str, Any], list[Path], str, list[str]]:
    project_dir = Path.cwd().resolve()
    config, sources = load_config(project_dir, _cli_overrides(args))
    check_signal_window(config, resubmit=not args.no_resubmit)
    exe = config["execution"]
    if exe["image"] != "none" and not Path(exe["image"]).is_file():
        raise ConfigError(f"container image does not exist: {exe['image']}")
    env_file = Path(exe["env_file"])
    if not env_file.exists():
        print(f"[submit] note: no env file at {env_file}", file=sys.stderr)

    if exe["image"] != "none":
        for bind in exe["binds"]:
            if not Path(bind).exists():
                raise ConfigError(f"Apptainer bind path does not exist on the submission node: {bind}")

    user_args = list(args.args)
    if user_args and user_args[0] == "--":
        user_args = user_args[1:]
    run_name = default_run_name() if args.run_name is None else validate_run_name(args.run_name)

    if not args.skip_path_checks:
        checks = [
            check_compute_writable(Path(exe["output_dir"]), "execution.output_dir", must_be_shared=True),
            check_compute_writable(Path(exe["cache_dir"]), "execution.cache_dir"),
        ]
        run_dir = str(Path(exe["output_dir"]) / "runs" / run_name)
        auto_args = [
            token.replace("{RUN}", run_name).replace("{RUN_DIR}", run_dir)
            for token in exe["auto_args"]
        ]
        checks.extend(check_payload_args(
            auto_args + user_args,
            project_dir,
            extra_writable=config["validation"]["writable_args"],
        ))
        for check in checks:
            prefix = "WARNING" if check.status == "warning" else "path ok"
            stream = sys.stderr if check.status == "warning" else sys.stdout
            print(f"[submit] {prefix}: {check.label}: {check.resolved} ({check.detail})", file=stream)
    else:
        print("[submit] WARNING: filesystem/path validation disabled by --skip-path-checks", file=sys.stderr)

    return config, sources, run_name, user_args


def _fake_chain_state(
    *,
    config: dict[str, Any],
    project_dir: Path,
    output_dir: Path,
    run_name: str,
    resubmit: bool,
) -> tuple[dict[str, Any], Path]:
    fake_chain = "DRY-RUN"
    fake_dir = output_dir / ".hummel-submit" / "chains" / fake_chain
    state = {
        "chain_id": fake_chain,
        "project_dir": str(project_dir),
        "output_dir": str(output_dir),
        "run_name": run_name,
        "config": config,
        "resubmit": resubmit,
        "python_executable": sys.executable,
        "snapshot_path": str(fake_dir / "hummel-submit-worker.zip"),
        "worker_script": str(fake_dir / "worker.sh"),
        "payload_script": str(fake_dir / "law-job.sh"),
    }
    return state, fake_dir / "state.json"


def _run_law_submission(spec_path: Path) -> None:
    try:
        import luigi
        from .law_payload import PayloadWorkflow
    except ImportError as exc:
        raise ConfigError(
            "law (master) is required for submission; reinstall hummel-submit with its dependencies"
        ) from exc

    task = PayloadWorkflow(
        humsub_spec=str(spec_path),
        workflow="slurm",
        no_poll=True,
        retries=0,
        tasks_per_job=1,
        job_workers=1,
    )
    success = luigi.build([task], local_scheduler=True, workers=1)
    if not success:
        raise ConfigError("law failed to prepare/submit the Hummel workflow")


def cmd_submit(args: argparse.Namespace) -> int:
    config, sources, run_name, user_args = _prepare_submission(args, args.dry_run)
    project_dir = Path.cwd().resolve()
    output_dir = Path(config["execution"]["output_dir"])

    print(f"[submit] project     {project_dir}")
    print(f"[submit] run         {run_name}")
    print(f"[submit] image       {config['execution']['image']}")
    print(f"[submit] output      {output_dir}")
    print(f"[submit] account     {config['slurm']['account']}")
    print(f"[submit] reservation {config['slurm']['reservation'] or 'none (may still be pulled in magnetically)'}")
    print(f"[submit] time limit  {config['slurm']['time_limit']} per hop")
    print("[submit] middleware  law (master) / Hummel chain backend")
    print(f"[submit] config      {', '.join(map(str, sources)) if sources else 'built-in defaults only'}")

    if args.dry_run:
        fake_state, fake_state_path = _fake_chain_state(
            config=config,
            project_dir=project_dir,
            output_dir=output_dir,
            run_name=run_name,
            resubmit=not args.no_resubmit,
        )
        cmd = sbatch_command(fake_state, fake_state_path, 0)
        print("[submit] law would render one remote-job script; the Hummel backend would start its chain with:")
        print("  " + shlex.join(cmd))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    spec, spec_path = create_submission_spec(
        output_dir=output_dir,
        project_dir=project_dir,
        config=config,
        run_name=run_name,
        user_args=user_args,
        resubmit=not args.no_resubmit,
        python_executable=sys.executable,
    )

    try:
        _run_law_submission(spec_path)
        spec = load_submission_spec(spec_path)
    except Exception:
        # Once a chain exists, preserve all state for diagnosis.  Before that
        # point, remove the empty submission/run directories just like the old
        # frontend did for a failed initial sbatch.
        try:
            current = load_submission_spec(spec_path)
        except Exception:
            current = spec
        if not current.get("chain_ids"):
            shutil.rmtree(spec_path.parent, ignore_errors=True)
            shutil.rmtree(Path(spec["run_dir"]), ignore_errors=True)
        raise

    chain_ids = list(spec.get("chain_ids", []))
    if not chain_ids:
        raise ConfigError("law returned successfully but the Hummel backend recorded no chain id")

    for chain_id in chain_ids:
        print(f"[submit] chain id    {chain_id}")
        print(f"[submit] state       {chain_root(output_dir) / chain_id / 'state.json'}")
        print(f"[submit] cancel with humsub cancel {chain_id}")
    print(f"[submit] law state   {spec['law_dir']}")
    return 0



def _stage_map(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, path = parse_stage_spec(value)
        if name in result:
            raise ConfigError(f"duplicate stage name {name!r}")
        result[name] = path
    return result


def _stage_exclude_map(values: list[str], stages: dict[str, Path]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for value in values:
        name, pattern = parse_stage_exclude(value)
        if name not in stages:
            raise ConfigError(f"stage exclude references unknown stage {name!r}")
        result.setdefault(name, []).append(pattern)
    return result


def cmd_submit_manifest(args: argparse.Namespace) -> int:
    project_dir = Path.cwd().resolve()
    config, sources = load_config(project_dir, _cli_overrides(args), require_command=False)
    run_name = default_run_name() if args.run_name is None else validate_run_name(args.run_name)
    manifest_source = args.manifest.expanduser().absolute()
    payload_source = args.payload.expanduser().absolute()
    if not manifest_source.is_file():
        raise ConfigError(f"manifest does not exist: {manifest_source}")
    if not payload_source.is_file():
        raise ConfigError(f"payload does not exist: {payload_source}")
    stages = _stage_map(args.stage)
    stage_excludes = _stage_exclude_map(args.stage_exclude, stages)
    opts = SubmitOptions.from_dict(_option_overrides(args))
    submit_manifest_run(
        project_dir=project_dir,
        config=config,
        sources=sources,
        run_name=run_name,
        manifest_source=manifest_source,
        payload_source=payload_source,
        stages=stages,
        stage_excludes=stage_excludes,
        opts=opts,
        skip_path_checks=args.skip_path_checks,
        dry_run=args.dry_run,
    )
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    config, _ = load_config(Path.cwd(), require_command=False)
    output_dir = Path(config["execution"]["output_dir"])
    spec_path = find_submission(output_dir, args.submission)
    stage_overrides = _stage_map(args.stage) if args.stage else None
    resume_lineage(
        output_dir,
        spec_path,
        config_overrides=_cli_overrides(args),
        option_overrides=_option_overrides(args),
        stage_overrides=stage_overrides,
        dry_run=args.dry_run,
        include_active=args.include_active,
    )
    return 0


def cmd_submission_status(args: argparse.Namespace) -> int:
    config, _ = load_config(Path.cwd(), require_command=False)
    output_dir = Path(config["execution"]["output_dir"])
    spec_path = submission_root(output_dir) / args.submission / "submission.json"
    if not spec_path.is_file():
        raise ConfigError(f"submission {args.submission!r} not found under {submission_root(output_dir)}")
    spec = load_submission_spec(spec_path)
    print(f"submission: {spec['submission_id']}")
    print(f"run:        {spec['run_name']}")
    print(f"chains:     {len(spec.get('chain_ids', []))}")
    for chain_id in spec.get("chain_ids", []):
        effective = query_chain(output_dir, chain_id)
        state_path = chain_root(output_dir) / chain_id / "state.json"
        state = load_state(state_path)
        print(f"  {chain_id}: {effective.state} ({state['status']})")
    print()
    for line in format_overview(compute_overview(output_dir, spec)):
        print(line)
    return 0


def _find_chain(chain: str) -> Path:
    config, _ = load_config(Path.cwd(), require_command=False)
    root = chain_root(Path(config["execution"]["output_dir"]))
    direct = root / chain / "state.json"
    if direct.exists():
        return direct
    if not root.exists():
        raise ConfigError(f"no chain state directory at {root}")
    for state_path in root.glob("*/state.json"):
        try:
            state = load_state(state_path)
        except Exception:
            continue
        if chain in state.get("jobs", []):
            return state_path
    raise ConfigError(f"chain or job id {chain!r} not found under {root}")


def cmd_help(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    parser.print_help()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    path = _find_chain(args.chain)
    state = load_state(path)
    effective = query_chain(Path(state["output_dir"]), state["chain_id"])
    print(f"chain:   {state['chain_id']}")
    print(f"run:     {state['run_name']}")
    print(f"status:  {state['status']} (law: {effective.state})")
    print(f"state:   {path}")
    print(f"jobs:    {', '.join(state['jobs']) or '-'}")
    queued = queue_status(state["jobs"])
    if queued:
        print("SLURM:")
        print(queued)
    else:
        print("SLURM:   no known jobs currently in squeue")

    jobs = list(map(str, state.get("jobs", [])))
    if jobs:
        current = str(state.get("current_job_id", ""))
        next_job = str(state.get("next_job_id", ""))
        print("logs:")
        for job_id in jobs:
            log_path = slurm_log_path(state, job_id)
            if job_id == current:
                role = "current"
            elif job_id == next_job:
                role = "next"
            else:
                role = "previous"
            availability = "exists" if log_path.exists() else "not created yet"
            print(f"  {job_id} {role:<8} {log_path} ({availability})")
    else:
        print("logs:    no known Slurm job logs yet")
    return 0


def cmd_follow(args: argparse.Namespace) -> int:
    path = _find_chain(args.chain)
    try:
        return follow_chain(path, initial_lines=args.lines, poll_interval=args.poll_interval)
    except KeyboardInterrupt:
        print("\n[follow] interrupted", file=sys.stderr)
        return 130


def _format_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} TiB"


def _submission_spec_from_current_config(submission: str) -> Path:
    config, _ = load_config(Path.cwd(), require_command=False)
    output_dir = Path(config["execution"]["output_dir"])
    path = submission_root(output_dir) / submission / "submission.json"
    if not path.is_file():
        raise ConfigError(f"submission {submission!r} not found under {submission_root(output_dir)}")
    return path


def _print_cleanup_status(status, *, prefix: str = "") -> None:
    age = "-" if status.terminal_age_hours is None else f"{status.terminal_age_hours:.1f} h"
    total = sum(entry.bytes for entry in status.cache_paths)
    print(f"{prefix}submission {status.submission_id}: {status.state}, terminal age {age}, cache {_format_bytes(total)}")
    if status.cache_paths:
        for entry in status.cache_paths:
            print(f"{prefix}  {entry.label:<12} {_format_bytes(entry.bytes):>10}  {entry.path}")
    else:
        print(f"{prefix}  no SSD runtime caches remain")


def cmd_cleanup(args: argparse.Namespace) -> int:
    spec_path = _submission_spec_from_current_config(args.submission)
    before = inspect_submission_cache(spec_path)
    _print_cleanup_status(before)
    if not before.cache_paths:
        return 0
    if args.dry_run:
        print("cleanup: dry run; nothing removed")
        return 0
    cleanup_submission_cache(spec_path, force=args.force, dry_run=False)
    print("cleanup: removed SSD runtime caches; persistent submission state, chain state, outputs and logs were preserved")
    return 0


def cmd_gc(args: argparse.Namespace) -> int:
    if args.successful_after_hours < 0 or args.failed_after_hours < 0:
        raise ConfigError("gc retention values must be >= 0 hours")
    if args.orphans_after_hours is not None and args.orphans_after_hours < 0:
        raise ConfigError("--orphans-after-hours must be >= 0")

    config, _ = load_config(Path.cwd(), require_command=False)
    output_dir = Path(config["execution"]["output_dir"])
    cache_dir = Path(config["execution"]["cache_dir"])
    specs = list(iter_submission_specs(output_dir))
    known_ids: set[str] = set()
    candidates = []
    skipped_active = 0
    for spec_path in specs:
        try:
            status = inspect_submission_cache(spec_path)
        except Exception as exc:
            print(f"gc: WARNING: cannot inspect {spec_path}: {exc}", file=sys.stderr)
            continue
        known_ids.add(status.submission_id)
        if status.active:
            skipped_active += 1
            continue
        if not status.cache_paths or not status.terminal:
            continue
        age = status.terminal_age_hours or 0.0
        threshold = args.successful_after_hours if status.state == "successful" else args.failed_after_hours
        if age >= threshold:
            candidates.append((spec_path, status))

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"gc: {mode}; output={output_dir}; cache={cache_dir}")
    print(f"gc: scanned {len(specs)} submission(s), skipped {skipped_active} active submission(s)")
    reclaim = 0
    for _, status in candidates:
        _print_cleanup_status(status, prefix="gc: ")
        reclaim += sum(entry.bytes for entry in status.cache_paths)

    orphan_entries = []
    if args.orphans_after_hours is not None:
        orphan_entries = stale_orphan_cache_dirs(
            cache_dir, known_ids, older_than_hours=args.orphans_after_hours
        )
        for entry in orphan_entries:
            print(f"gc: orphan {entry.label:<20} {_format_bytes(entry.bytes):>10}  {entry.path}")
            reclaim += entry.bytes

    if args.stale_scratch:
        count, nbytes = sweep_stale_scratch(config, None, apply=args.apply)
        print(f"gc: stale scratch of dead jobs: {count} dir(s), {_format_bytes(nbytes)}")
        if args.apply:
            reclaim += 0
        else:
            reclaim += nbytes

    print(f"gc: reclaimable {_format_bytes(reclaim)}")
    if not args.apply:
        print("gc: dry run; pass --apply to delete")
        return 0

    for spec_path, _ in candidates:
        cleanup_submission_cache(spec_path, force=False, dry_run=False)
    remove_cache_paths(orphan_entries)
    print("gc: cleanup complete; persistent submission metadata, outputs and logs were preserved")
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    path = _find_chain(args.chain)
    state = load_state(path)
    cancel_chain(Path(state["output_dir"]), state["chain_id"], reason="cancelled-by-user")
    print(f"marked chain {state['chain_id']} done and requested cancellation of {len(state['jobs'])} known job(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"help", "init", "config", "submit", "submit-manifest", "resume", "status", "follow", "submission-status", "cleanup", "gc", "cancel"}
    top_level = {"-h", "--help", "--version"}
    if argv and argv[0] not in commands and argv[0] not in top_level:
        argv.insert(0, "submit")
    args = parser.parse_args(argv)
    try:
        if args.subcommand == "help":
            return cmd_help(args, parser)
        if args.subcommand == "init":
            return cmd_init(args)
        if args.subcommand == "config":
            return cmd_config(args)
        if args.subcommand == "submit":
            return cmd_submit(args)
        if args.subcommand == "submit-manifest":
            return cmd_submit_manifest(args)
        if args.subcommand == "resume":
            return cmd_resume(args)
        if args.subcommand == "status":
            return cmd_status(args)
        if args.subcommand == "follow":
            return cmd_follow(args)
        if args.subcommand == "submission-status":
            return cmd_submission_status(args)
        if args.subcommand == "cleanup":
            return cmd_cleanup(args)
        if args.subcommand == "gc":
            return cmd_gc(args)
        if args.subcommand == "cancel":
            return cmd_cancel(args)
        parser.error("unknown command")
    except (ConfigError, PathCheckError, SlurmError, OSError, RuntimeError, ValueError) as exc:
        print(f"humsub: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
