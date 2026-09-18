from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import shutil
import sys
from typing import Any

from . import __version__
from .config import ConfigError, PROJECT_CONFIG_NAME, USER_CONFIG, config_as_json, load_config, validate_run_name
from .slurm import SlurmError, cancel_jobs, queue_status, sbatch_command, submit
from .pathcheck import PathCheckError, check_compute_writable, check_payload_args
from .state import append_job, chain_root, create_state, default_run_name, done_marker, load_state, mark_status
from .templates import PROJECT_TEMPLATE, USER_TEMPLATE


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="humsub", description="Hummel-2 SLURM submission helper")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    init = sub.add_parser("init", help="create editable user and project config templates without overwriting existing files")
    mode = init.add_mutually_exclusive_group()
    mode.add_argument("--project-only", action="store_true")
    mode.add_argument("--user-only", action="store_true")

    show = sub.add_parser("config", help="show the resolved effective configuration")
    show.add_argument("--json", action="store_true", help="print machine-readable JSON")

    submit_p = sub.add_parser("submit", help="submit a job (arguments after -- are passed to the project command)")
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

    cancel = sub.add_parser("cancel", help="mark a chain done and cancel all of its known SLURM jobs")
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
        # Inspect both project-supplied auto arguments and one-off user arguments.
        # Resolve the placeholders known at submit time; leave runtime-only placeholders
        # untouched so the checker does not mistake them for host paths.
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
    print(f"[submit] config      {', '.join(map(str, sources)) if sources else 'built-in defaults only'}")

    if args.dry_run:
        fake_chain = "DRY-RUN"
        fake_dir = output_dir / ".hummel-submit" / "chains" / fake_chain
        fake_state = {
            "chain_id": fake_chain,
            "project_dir": str(project_dir),
            "output_dir": str(output_dir),
            "run_name": run_name,
            "run_dir": str(output_dir / "runs" / run_name),
            "config": config,
            "user_args": user_args,
            "resubmit": not args.no_resubmit,
            "python_executable": sys.executable,
            "snapshot_path": str(fake_dir / "hummel-submit-worker.zip"),
            "worker_script": str(fake_dir / "worker.sh"),
        }
        cmd = sbatch_command(fake_state, fake_dir / "state.json", 0)
        print("[submit] would run:")
        print("  " + shlex.join(cmd))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    package_dir = Path(__file__).resolve().parent
    state, state_path = create_state(
        output_dir=output_dir,
        project_dir=project_dir,
        config=config,
        run_name=run_name,
        user_args=user_args,
        resubmit=not args.no_resubmit,
        python_executable=sys.executable,
        package_dir=package_dir,
    )
    try:
        job_id = submit(state, state_path, 0)
    except Exception:
        # The chain never became live; remove the snapshot/run directory to avoid litter.
        shutil.rmtree(state_path.parent, ignore_errors=True)
        shutil.rmtree(Path(state["run_dir"]), ignore_errors=True)
        raise

    state = append_job(state_path, job_id)
    print(f"[submit] submitted job {job_id}")
    print(f"[submit] chain id    {state['chain_id']}")
    print(f"[submit] state       {state_path}")
    print(f"[submit] cancel with humsub cancel {state['chain_id']}")
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
    print(f"chain:   {state['chain_id']}")
    print(f"run:     {state['run_name']}")
    print(f"status:  {state['status']}")
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
    done_marker(path).write_text("cancelled-by-user\n", encoding="utf-8")
    state = mark_status(path, "cancelled-by-user")
    cancel_jobs(state["jobs"])
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
    except (ConfigError, PathCheckError, SlurmError, OSError, ValueError) as exc:
        print(f"humsub: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
