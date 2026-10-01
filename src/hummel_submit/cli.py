from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import shutil
import sys
from typing import Any

from . import __version__
from .chain import cancel_chain, query_chain
from .config import ConfigError, PROJECT_CONFIG_NAME, USER_CONFIG, config_as_json, load_config, validate_run_name
from .pathcheck import PathCheckError, check_compute_writable, check_payload_args
from .slurm import SlurmError, queue_status, sbatch_command
from .state import chain_root, default_run_name, load_state
from .submission import create_submission_spec, load_submission_spec
from .templates import PROJECT_TEMPLATE, USER_TEMPLATE


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="humsub", description="law-backed Hummel-2 submission helper")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    init = sub.add_parser("init", help="create editable user and project config templates without overwriting existing files")
    mode = init.add_mutually_exclusive_group()
    mode.add_argument("--project-only", action="store_true")
    mode.add_argument("--user-only", action="store_true")

    show = sub.add_parser("config", help="show the resolved effective configuration")
    show.add_argument("--json", action="store_true", help="print machine-readable JSON")

    submit_p = sub.add_parser("submit", help="submit a law workflow for the configured payload")
    submit_p.add_argument("--dry-run", action="store_true")
    submit_p.add_argument("--no-resubmit", action="store_true")
    submit_p.add_argument("--run-name")
    submit_p.add_argument("--time", dest="time_limit")
    submit_p.add_argument("--account")
    submit_p.add_argument("--partition")
    submit_p.add_argument("--reservation")
    submit_p.add_argument("--mail")
    submit_p.add_argument("--max-hops", type=int)
    submit_p.add_argument("--sbatch-arg", action="append", default=[], help="append an extra sbatch option; repeat as needed")
    submit_p.add_argument("--skip-path-checks", action="store_true", help="skip Hummel filesystem/path validation (escape hatch)")
    submit_p.add_argument("args", nargs=argparse.REMAINDER)

    status = sub.add_parser("status", help="show saved chain state and current SLURM queue state")
    status.add_argument("chain")

    cancel = sub.add_parser("cancel", help="cancel the autonomous Hummel chain")
    cancel.add_argument("chain")

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
    for attr in ("account", "partition", "reservation", "mail", "max_hops"):
        value = getattr(args, attr, None)
        if value is not None:
            slurm[attr] = value
    if getattr(args, "time_limit", None) is not None:
        slurm["time_limit"] = args.time_limit
    if getattr(args, "sbatch_arg", None):
        slurm["extra_args"] = args.sbatch_arg
    return {"slurm": slurm} if slurm else {}


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
    run_name = validate_run_name(args.run_name) if args.run_name else default_run_name()

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
            "law release_prep is required for submission; reinstall hummel-submit with its dependencies"
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
    print(f"[submit] middleware  law release_prep / Hummel chain backend")
    print(f"[submit] config      {', '.join(map(str, sources)) if sources else 'built-in defaults only'}")

    if config["slurm"].get("retry_on_failure"):
        print(
            "[submit] WARNING: slurm.retry_on_failure is deprecated in the law-backed design; "
            "ordinary failures terminate a chain and should be retried by law",
            file=sys.stderr,
        )

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
    commands = {"init", "config", "submit", "status", "cancel"}
    top_level = {"-h", "--help", "--version"}
    if argv and argv[0] not in commands and argv[0] not in top_level:
        argv.insert(0, "submit")
    args = parser.parse_args(argv)
    try:
        if args.subcommand == "init":
            return cmd_init(args)
        if args.subcommand == "config":
            return cmd_config(args)
        if args.subcommand == "submit":
            return cmd_submit(args)
        if args.subcommand == "status":
            return cmd_status(args)
        if args.subcommand == "cancel":
            return cmd_cancel(args)
        parser.error("unknown command")
    except (ConfigError, PathCheckError, SlurmError, OSError, RuntimeError, ValueError) as exc:
        print(f"humsub: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
