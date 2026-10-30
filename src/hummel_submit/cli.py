from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import difflib
import os
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
from .submission import create_submission_spec, load_submission_spec, require_owner, submission_root, submission_owner
from .manifest import load_manifest
from .manifest_submit import (
    SubmitOptions,
    all_outputs_exist as _all_outputs_exist,
    discard_preparation as _discard_preparation,
    print_manifest_submission as _print_manifest_submission,
    submit_manifest_run,
)
from .doctor import local_checks, manifest_paths, run_probe
from .make_manifest import build_manifest, find_inputs, json_object, parse_assignments
from .payloads import resolve as resolve_builtin_payload
from .state import atomic_write_json
from .prune import apply_plan, fmt as fmt_size, plan_submission, select_submissions as select_prune_submissions
from .stats import format_stats, stats_for, to_json as stats_json
from .overview import compute_overview, finalize_state, find_submission, format_overview
from .staging import parse_stage_exclude, parse_stage_spec, stage_inputs
from .templates import PROJECT_TEMPLATE, USER_TEMPLATE


def _add_slurm_options(parser: argparse.ArgumentParser) -> None:
    """Per-submission overrides of [slurm] settings (shared by submit and submit-manifest)."""
    parser.add_argument("--no-resubmit", action="store_true", help="no continuation hops: the job runs once, in a single Slurm job")
    _add_slurm_overrides(parser)
    parser.add_argument("--skip-path-checks", action="store_true", help="skip Hummel filesystem/path validation (escape hatch)")


def _add_slurm_overrides(parser: argparse.ArgumentParser) -> None:
    """[slurm] settings that can also be changed for the future chains of a running submission (resume)."""
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


def _add_manifest_policy_options(parser: argparse.ArgumentParser) -> None:
    """Retry, concurrency, scratch and supervision policy of a manifest submission."""
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
                        help="queue a tiny Slurm job that runs a law pass periodically (polls, retries failed jobs, "
                             "submits what is ready; no login-node process)")
    parser.add_argument("--supervisor-interval", type=int, default=None, metavar="SECONDS",
                        help="seconds between supervisor passes (default 1800)")
    parser.add_argument("--supervisor-trigger-delay", type=int, default=None, metavar="SECONDS",
                        help="a chain that ends brings the next supervisor pass forward to this many seconds later "
                             "(default 60: gathers the ends of many chains into one pass)")
    parser.add_argument("--supervisor-rounds", type=int, default=None, metavar="R",
                        help="law retries per failed job when the supervisor runs and --retries is not given (default 3)")
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
        "supervisor_trigger_delay": "supervisor_trigger_delay",
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
    manifest.add_argument(
        "--payload", required=True, metavar="PATH|builtin:NAME",
        help="executable invoked once per branch, or a built-in payload (builtin:container runs the manifest's "
             "command in an Apptainer image)",
    )
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
    manifest.add_argument(
        "--retries", type=int, default=None, metavar="R",
        help="law resubmissions per failed job (needs --wait or --supervisor-job; with the supervisor the default is "
             "--supervisor-rounds, 0 disables retries)",
    )
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

    make = sub.add_parser(
        "make-manifest",
        help="write a manifest from a folder search: one entry per input file, outputs from templates, "
             "inputs packed into branches",
    )
    make.add_argument("--search", type=Path, action="append", required=True, metavar="DIR",
                      help="folder searched recursively (repeatable)")
    make.add_argument("--glob", default="*", help="file name pattern of the inputs (default: *)")
    make.add_argument("--require", metavar="REGEX", help="every file name found must fullmatch this, else stop")
    make.add_argument("--select", metavar="REGEX", help="only file names that fullmatch this become inputs")
    make.add_argument("--output", action="append", required=True, metavar="TEMPLATE",
                      help="declared output per input file (repeatable); fields {stem} {name} {parent} and --set keys")
    make.add_argument("--files-per-branch", type=int, default=1, metavar="K",
                      help="input files per branch (one payload run handles all of them); default 1")
    make.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                      help="string value in common and a template field (repeatable)")
    make.add_argument("--common", type=Path, metavar="JSON_FILE", help="JSON object merged into common")
    make.add_argument("--common-json", metavar="JSON", help="JSON object merged into common")
    make.add_argument("--finalize-output", action="append", default=[], metavar="TEMPLATE",
                      help="output of the finalize step, run once all branches are done (repeatable)")
    make.add_argument("--finalize-data-json", metavar="JSON", help="data of the finalize step (e.g. its own command)")
    make.add_argument("-o", "--manifest", type=Path, required=True, help="manifest file to write")

    tick = sub.add_parser(
        "tick",
        help="one pass of law over a submission: poll its jobs once, resubmit failed ones within --retries, submit "
             "pending ones (what the supervisor job does periodically)",
    )
    tick.add_argument("submission", help="submission id or run name")

    sub.add_parser("help", help="show this help message")

    status = sub.add_parser("status", help="show saved chain state, queue state and log paths of one chain")
    status.add_argument("chain", help="chain id (or any of its Slurm job ids)")

    follow = sub.add_parser("follow", help="follow the active chain log across continuation hops")
    follow.add_argument("chain", help="chain id (or any of its Slurm job ids)")
    follow.add_argument("-n", "--lines", type=int, default=10, help="initial lines to show from the current log (default: 10)")
    follow.add_argument("--poll-interval", type=float, default=1.0, help=argparse.SUPPRESS)

    resume = sub.add_parser(
        "resume",
        help="continue a submission after its passes gave up: fresh retries for failed jobs, optional new settings for "
             "future chains, one pass now, supervisor queued again",
    )
    resume.add_argument("submission", help="submission id or run name")
    resume.add_argument("--dry-run", action="store_true", help="show what would change; change and submit nothing")
    resume.add_argument("--stage", action="append", default=[], metavar="NAME=PATH",
                        help="re-copy a stage from this source (e.g. a renewed proxy); other stages stay frozen")
    resume.add_argument("--retries", type=int, default=None, metavar="R",
                        help="law retries per failed job from now on (default: as submitted, at least 1)")
    resume.add_argument("--parallel-jobs", type=int, default=None, metavar="J", help="at most J branch jobs in flight")
    resume.add_argument("--supervisor-job", action="store_true", help="supervise the submission from now on")
    resume.add_argument("--supervisor-interval", type=int, default=None, metavar="SECONDS",
                        help="seconds between supervisor passes")
    _add_slurm_overrides(resume)

    stats = sub.add_parser("stats", help="what a production cost: jobs, runtimes, core-hours, queue waits, failures, retries, peak concurrency")
    stats.add_argument("submission", help="submission id or run name")
    stats.add_argument("--json", action="store_true", help="machine-readable output")

    prune = sub.add_parser(
        "prune",
        help="delete the provenance (state, chains, run dir, Slurm logs, caches) of selected finished submissions; dry run unless --apply",
    )
    prune.add_argument("--run", action="append", default=[], metavar="NAME", help="run name (repeatable)")
    prune.add_argument("--id", action="append", default=[], metavar="PREFIX", help="submission id or prefix (repeatable)")
    prune.add_argument("--before", metavar="YYYY-MM-DD", help="all submissions created before this date")
    prune.add_argument("--apply", action="store_true", help="actually delete (default: show what would be deleted)")
    prune.add_argument("--force", action="store_true", help="also prune submissions that still look active")

    doctor = sub.add_parser("doctor", help="check that this setup can run a production; --probe also tests from a real batch node")
    doctor.add_argument("--probe", action="store_true", help="run a tiny batch job: writes to the output locations, nested sbatch, humsub importable")
    doctor.add_argument("--check-path", action="append", default=[], type=Path, metavar="PATH", help="also check that a batch node can write here (repeatable)")
    doctor.add_argument("--manifest", type=Path, help="check the common output location of this manifest")
    doctor.add_argument("--timeout", type=int, default=300, help="seconds to wait for the probe job (default 300)")
    doctor.add_argument("--keep-nice", action="store_true", help="queue the probe with the configured nice instead of 0")
    doctor.add_argument("--cpu-only", action="store_true", help="do not request the configured GPUs for the probe")

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

    gc = sub.add_parser(
        "gc",
        help="garbage-collect caches and scratch of your finished submissions: lists them per submission; "
             "--apply asks which to delete, --submission names them, --force deletes all",
    )
    gc.add_argument("--apply", action="store_true",
                    help="delete after asking which of the listed submissions to clean (needs a terminal)")
    gc.add_argument("--submission", action="append", default=[], metavar="ID|RUN",
                    help="clean exactly this submission (id, id prefix or run name; repeatable), without asking")
    gc.add_argument(
        "-f", "--force", action="store_true",
        help="DANGER: delete everything listed, for every submission, without asking.  Caches and scratch cannot be "
             "recovered; failed submissions lose the stages and scratch leftovers you may still need to debug or "
             "resume them.  Only for scripts that know exactly what the output_dir and cache_dir contain",
    )
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
            "law is required for submission; reinstall hummel-submit with its dependencies"
        ) from exc

    task = PayloadWorkflow(
        humsub_spec=str(spec_path),
        workflow="hummel",
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
    print("[submit] middleware  law / Hummel chain backend (workflow type \"hummel\")")
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
    payload_source = resolve_builtin_payload(args.payload) or Path(args.payload).expanduser().absolute()
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


def cmd_make_manifest(args: argparse.Namespace) -> int:
    values = parse_assignments(args.set)
    common: dict[str, Any] = {}
    if args.common:
        common.update(json_object(args.common.read_text(encoding="utf-8"), str(args.common)))
    if args.common_json:
        common.update(json_object(args.common_json, "--common-json"))
    files, found = find_inputs(args.search, args.glob, require=args.require, select=args.select)
    manifest = build_manifest(
        files,
        outputs=args.output,
        files_per_branch=args.files_per_branch,
        common=common,
        values=values,
        finalize_outputs=args.finalize_output,
        finalize_data=json_object(args.finalize_data_json, "--finalize-data-json") if args.finalize_data_json else None,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.manifest, manifest)
    load_manifest(args.manifest)  # the same validation submit-manifest applies
    print(
        f"[make-manifest] {args.manifest}: {len(manifest['branches'])} branch(es), {len(files)} input file(s) "
        f"({found - len(files)} of {found} found left out by --select)"
        + (", with a finalize step" if "finalize" in manifest else "")
    )
    return 0


def cmd_tick(args: argparse.Namespace) -> int:
    from .tick import ensure_lock_generation, run_tick

    config, _ = load_config(Path.cwd(), require_command=False)
    output_dir = Path(config["execution"]["output_dir"])
    spec_path = find_submission(output_dir, args.submission)
    require_owner(spec_path, load_submission_spec(spec_path), "run a pass over")
    ensure_lock_generation(spec_path)
    result = run_tick(spec_path, "humsub tick")
    print(f"[tick] {result.state}")
    for error in result.errors:
        print(f"[tick]   {error}")
    spec = load_submission_spec(spec_path)
    for line in format_overview(compute_overview(output_dir, spec), hint=False):
        print(f"[tick] {line}")
    finalize = finalize_state(spec)
    if finalize:
        print(f"[tick] finalize:   {finalize}")
    return {"complete": 0, "progress": 0, "busy": 3, "failed": 1}[result.state]


def cmd_resume(args: argparse.Namespace) -> int:
    from .resume import resume
    from .tick import ensure_lock_generation

    output_dir, spec_path, spec = _submission_context(args.submission)
    require_owner(spec_path, spec, "resume")
    if not args.dry_run:
        ensure_lock_generation(spec_path)
    result = resume(
        output_dir,
        spec_path,
        config_overrides=_cli_overrides(args),
        retries=args.retries,
        parallel_jobs=args.parallel_jobs,
        stage_overrides=_stage_map(args.stage) if args.stage else None,
        supervisor=args.supervisor_job,
        supervisor_interval=args.supervisor_interval,
        dry_run=args.dry_run,
    )
    return 1 if result is not None and result.state == "failed" else 0


def _submission_context(ident: str) -> tuple[Path, Path, dict[str, Any]]:
    config, _ = load_config(Path.cwd(), require_command=False)
    output_dir = Path(config["execution"]["output_dir"])
    spec_path = find_submission(output_dir, ident)
    return output_dir, spec_path, load_submission_spec(spec_path)


def cmd_stats(args: argparse.Namespace) -> int:
    output_dir, _, spec = _submission_context(args.submission)
    summary = stats_for(output_dir, spec, str(spec["config"]["slurm"]["job_name"]))
    if args.json:
        print(stats_json(summary))
    else:
        print(f"submission {spec['submission_id']} (run {spec['run_name']})")
        for line in format_stats(summary):
            print(line)
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    if not (args.run or args.id or args.before):
        raise ConfigError("select what to prune with --run, --id or --before")
    config, _ = load_config(Path.cwd(), require_command=False)
    output_dir = Path(config["execution"]["output_dir"])
    chosen = select_prune_submissions(output_dir, args.run, args.before, args.id)
    for path in chosen:
        require_owner(path, load_submission_spec(path), "prune")
    if not chosen:
        print("prune: nothing matches")
        return 0
    all_specs = list(iter_submission_specs(output_dir))
    plans = [plan_submission(output_dir, p) for p in chosen]
    total = 0
    refused = 0
    for plan in plans:
        size = plan.size()
        active = plan.state == "active"
        total += 0 if (active and not args.force) else size
        print(f"prune: submission {plan.submission_id} run {plan.run_name} [{plan.state}] {fmt_size(size)} in {len(plan.paths)} path(s)"
              + ("  REFUSED: still active (use --force)" if active and not args.force else ""))
        counts: dict[str, int] = {}
        for label, _ in plan.paths:
            counts[label] = counts.get(label, 0) + 1
        print("prune:   " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
        for note in plan.skipped:
            print(f"prune:   KEPT (safety): {note}")
        refused += int(active and not args.force)
    print(f"prune: {len(plans)} submission(s) selected, {len(all_specs) - len(plans)} other submission(s) of this output_dir are untouched; {fmt_size(total)} to reclaim")
    if not args.apply:
        print("prune: dry run; pass --apply to delete")
        return 1 if refused else 0
    for plan in plans:
        if plan.state == "active" and not args.force:
            continue
        apply_plan(plan)
    print("prune: done (declared outputs of the manifests were never touched)")
    return 1 if refused else 0


def cmd_doctor(args: argparse.Namespace) -> int:
    config, sources = load_config(Path.cwd(), require_command=False)
    extra = list(args.check_path)
    if args.manifest:
        extra += manifest_paths(args.manifest.expanduser().absolute())
    checks = local_checks(config, extra)
    if args.probe:
        output_dir = Path(config["execution"]["output_dir"])
        paths = [output_dir, Path(config["execution"]["cache_dir"]), *extra]
        print("doctor: probe job submitted, waiting for a batch node ...", flush=True)
        checks += run_probe(config, output_dir, paths, args.timeout, keep_nice=args.keep_nice, cpu_only=args.cpu_only)
    for check in checks:
        print(check.line())
    failed = [c for c in checks if c.status == "fail"]
    print(f"doctor: {len(failed)} failed, {sum(c.status == 'warn' for c in checks)} warning(s)" + ("" if args.probe else "; run with --probe for the checks that need a batch node"))
    return 1 if failed else 0


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
    finalize = finalize_state(spec)
    if finalize:
        print(f"finalize:   {finalize}")
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
    require_owner(spec_path, load_submission_spec(spec_path), "clean up")
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
    groups, scanned, skipped_active, skipped_foreign = _gc_groups(config, args)

    print(f"gc: output={output_dir}; cache={cache_dir}")
    print(f"gc: scanned {scanned} submission(s); skipped {skipped_active} active and {skipped_foreign} of other users")
    if not groups:
        print("gc: nothing to clean")
        return 0
    for i, group in enumerate(groups, 1):
        print(f"gc: [{i}] {group.describe()}")
        for kind, path, nbytes in group.entries:
            print(f"gc:       {kind:<8} {_format_bytes(nbytes):>10}  {path}")
    print(f"gc: reclaimable {_format_bytes(sum(g.bytes for g in groups))}")

    if args.force:
        chosen = groups
    elif args.submission:
        chosen = _gc_select_named(groups, args.submission)
    elif not args.apply:
        print("gc: dry run; --apply asks which to clean, --submission ID cleans one, --force cleans all (see --help)")
        return 0
    elif not sys.stdin.isatty():
        print("gc: not a terminal, so nothing is deleted without being named: use --submission ID (repeatable), "
              "or --force for everything listed", file=sys.stderr)
        return 2
    else:
        chosen = _gc_ask(groups, input)
    if not chosen:
        print("gc: nothing selected; nothing deleted")
        return 0
    for group in chosen:
        group.remove()
        print(f"gc: cleaned {group.key} ({_format_bytes(group.bytes)})")
    print("gc: done; persistent submission metadata, outputs and logs were preserved")
    return 0


@dataclass
class _GcGroup:
    """What gc would remove for one submission (or the orphans): its cache after the retention time, and the scratch
    of its dead Slurm jobs."""
    key: str
    label: str
    spec_path: Path | None = None
    cache: bool = False
    entries: list[tuple[str, Path, int]] = field(default_factory=list)
    scratch: list[Path] = field(default_factory=list)
    orphans: list[Any] = field(default_factory=list)

    @property
    def bytes(self) -> int:
        return sum(n for _, _, n in self.entries)

    def describe(self) -> str:
        return f"{self.key}  {self.label}  ({_format_bytes(self.bytes)})"

    def remove(self) -> None:
        if self.cache and self.spec_path is not None:
            cleanup_submission_cache(self.spec_path, force=False, dry_run=False)
        for path in self.scratch:
            shutil.rmtree(path, ignore_errors=True)
        remove_cache_paths(self.orphans)


def _gc_groups(config: dict[str, Any], args: argparse.Namespace) -> tuple[list[_GcGroup], int, int, int]:
    """(groups, scanned, skipped active, skipped of other users): only the current user's submissions and directories
    are ever considered, wherever output_dir and cache_dir are shared."""
    from .cleanup import _dir_size
    from .scratch import scratch_roots, stale_scratch_dirs
    from .slurm import live_job_ids

    output_dir = Path(config["execution"]["output_dir"])
    cache_dir = Path(config["execution"]["cache_dir"])
    groups: dict[str, _GcGroup] = {}
    labels: dict[str, tuple[str, Path]] = {}
    known_ids: set[str] = set()
    scanned = skipped_active = skipped_foreign = 0
    for spec_path in iter_submission_specs(output_dir):
        scanned += 1
        try:
            spec = load_submission_spec(spec_path)
            known_ids.add(spec["submission_id"])
            if submission_owner(spec_path, spec)[0] != os.getuid():
                skipped_foreign += 1
                continue
            labels[spec["submission_id"]] = (f"run={spec.get('run_name')} project={spec.get('project_dir')}", spec_path)
            status = inspect_submission_cache(spec_path)
        except Exception as exc:
            print(f"gc: WARNING: cannot inspect {spec_path}: {exc}", file=sys.stderr)
            continue
        if status.active:
            skipped_active += 1
            continue
        if not status.cache_paths or not status.terminal:
            continue
        age = status.terminal_age_hours or 0.0
        threshold = args.successful_after_hours if status.state == "successful" else args.failed_after_hours
        if age >= threshold:
            group = groups.setdefault(status.submission_id, _GcGroup(status.submission_id, labels[status.submission_id][0], spec_path))
            group.cache = True
            group.entries += [(f"cache:{e.label}", e.path, e.bytes) for e in status.cache_paths]

    if args.stale_scratch:
        for path in stale_scratch_dirs(scratch_roots(config), live_job_ids(all_users=True)):
            sub_id = path.parent.name
            label, spec_path = labels.get(sub_id, ("(submission not in this output_dir)", None))
            group = groups.setdefault(sub_id, _GcGroup(sub_id, label, spec_path))
            group.scratch.append(path)
            group.entries.append(("scratch", path, _dir_size(path)))

    if args.orphans_after_hours is not None:
        mine = [e for e in stale_orphan_cache_dirs(cache_dir, known_ids, older_than_hours=args.orphans_after_hours)
                if _owned_by_me(e.path)]
        if mine:
            group = _GcGroup("orphans", "cache directories no submission of this output_dir refers to")
            group.orphans = mine
            group.entries = [(f"orphan:{e.label}", e.path, e.bytes) for e in mine]
            groups["orphans"] = group
    return sorted(groups.values(), key=lambda g: g.key), scanned, skipped_active, skipped_foreign


def _owned_by_me(path: Path) -> bool:
    try:
        return Path(path).stat().st_uid == os.getuid()
    except OSError:
        return False


def _gc_select_named(groups: list[_GcGroup], names: list[str]) -> list[_GcGroup]:
    chosen: list[_GcGroup] = []
    for name in names:
        matches = [g for g in groups if g.key == name or g.key.startswith(name) or f"run={name} " in g.label + " "]
        if not matches:
            raise ConfigError(f"gc: {name!r} is not among the listed submissions")
        if len(matches) > 1:
            raise ConfigError(f"gc: {name!r} is ambiguous: {', '.join(g.key for g in matches)}")
        if matches[0] not in chosen:
            chosen.append(matches[0])
    return chosen


def _gc_ask(groups: list[_GcGroup], ask) -> list[_GcGroup]:
    """Ask which of the listed groups to clean: numbers ("1,3", "2-4"), "all", or nothing."""
    answer = ask(f"gc: clean which? numbers (e.g. 1,3 or 2-4), 'all', or Enter for none: ").strip().lower()
    if not answer:
        return []
    if answer == "all":
        return list(groups)
    picked: list[_GcGroup] = []
    for part in answer.replace(" ", "").split(","):
        lo, _, hi = part.partition("-")
        try:
            numbers = range(int(lo), int(hi or lo) + 1)
        except ValueError:
            raise ConfigError(f"gc: cannot read {part!r}; nothing deleted") from None
        for n in numbers:
            if not 1 <= n <= len(groups):
                raise ConfigError(f"gc: there is no [{n}]; nothing deleted")
            if groups[n - 1] not in picked:
                picked.append(groups[n - 1])
    return picked


def cmd_cancel(args: argparse.Namespace) -> int:
    path = _find_chain(args.chain)
    state = load_state(path)
    if state.get("submission_id"):
        spec_path = submission_root(Path(state["output_dir"])) / state["submission_id"] / "submission.json"
        if spec_path.is_file():
            require_owner(spec_path, load_submission_spec(spec_path), "cancel chains of")
    cancel_chain(Path(state["output_dir"]), state["chain_id"], reason="cancelled-by-user")
    print(f"marked chain {state['chain_id']} done and requested cancellation of {len(state['jobs'])} known job(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = sorted(
        name for action in parser._subparsers._group_actions for name in action.choices  # type: ignore[union-attr]
    )
    top_level = {"-h", "--help", "--version"}
    if argv and argv[0] not in commands and argv[0] not in top_level:
        if not argv[0].startswith("-"):
            # An unknown or mistyped command (or one this installed version does not have yet) must never turn into a
            # job submission: only `humsub -- ARGS` and `humsub --option ...` are shorthands for `humsub submit`.
            close = difflib.get_close_matches(argv[0], commands, n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            print(
                f"humsub: error: unknown command {argv[0]!r}{hint} (humsub {__version__}; "
                "arguments for the configured command go after `humsub submit --`)",
                file=sys.stderr,
            )
            return 2
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
        if args.subcommand == "make-manifest":
            return cmd_make_manifest(args)
        if args.subcommand == "resume":
            return cmd_resume(args)
        if args.subcommand == "tick":
            return cmd_tick(args)
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
        if args.subcommand == "stats":
            return cmd_stats(args)
        if args.subcommand == "prune":
            return cmd_prune(args)
        if args.subcommand == "doctor":
            return cmd_doctor(args)
        parser.error("unknown command")
    except (ConfigError, PathCheckError, SlurmError, OSError, RuntimeError, ValueError) as exc:
        print(f"humsub: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
