from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import tomllib
from typing import Any

PROJECT_CONFIG_NAME = ".hummel-submit.toml"
USER_CONFIG = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "hummel-submit" / "config.toml"

DEFAULTS: dict[str, Any] = {
    "execution": {
        "image": "none",
        "output_dir": "${BEEGFS}/jobs",
        "cache_dir": "${SSD}/.hummel-submit/cache",
        "command": [],
        "auto_args": [],
        "checkpoint_glob": "",
        "env_file": ".env",
        "binds": ["${BEEGFS}", "${USW}", "${SSD}"],
        "nv": True,
        "apptainer": "",
    },
    "slurm": {
        "job_name": "job",
        "account": "kasieczka_gpu",
        "partition": "gpu",
        "nodes": 1,
        "gpus": 1,
        "time_limit": "24:00:00",
        "signal_seconds": 600,
        "mail": "",
        "reservation": "",
        "max_hops": 20,
        "retry_on_failure": False,
        "extra_args": [],
    },
    "validation": {
        "writable_args": [],
    },
}

_ALLOWED = {section: set(values) for section, values in DEFAULTS.items()}

_ENV_OVERRIDES = {
    "HUMMEL_IMAGE": ("execution", "image", str),
    "HUMMEL_OUTPUT_DIR": ("execution", "output_dir", str),
    "HUMMEL_CACHE_DIR": ("execution", "cache_dir", str),
    "HUMMEL_ACCOUNT": ("slurm", "account", str),
    "HUMMEL_PARTITION": ("slurm", "partition", str),
    "HUMMEL_TIME_LIMIT": ("slurm", "time_limit", str),
    "HUMMEL_MAIL": ("slurm", "mail", str),
    "HUMMEL_RESERVATION": ("slurm", "reservation", str),
    "HUMMEL_MAX_HOPS": ("slurm", "max_hops", int),
}

_RESERVED_SBATCH_OPTIONS = {
    "--account", "-A",
    "--partition", "-p",
    "--nodes", "-N",
    "--gpus", "--gpus-per-node",
    "--time", "-t",
    "--export",
    "--signal",
    "--chdir", "-D",
    "--output", "-o",
    "--mail-user",
    "--mail-type",
    "--reservation",
    "--dependency", "-d",
}

_FORBIDDEN_MEMORY_OPTIONS = {"--mem", "--mem-per-cpu", "--mem-per-gpu"}
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_UNEXPANDED_ENV_RE = re.compile(r"\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)")


class ConfigError(ValueError):
    pass


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    _validate_keys(data, path)
    return data


def _validate_keys(data: dict[str, Any], path: Path) -> None:
    unknown_sections = set(data) - set(_ALLOWED)
    if unknown_sections:
        raise ConfigError(f"unknown section(s) in {path}: {', '.join(sorted(unknown_sections))}")
    for section, values in data.items():
        if not isinstance(values, dict):
            raise ConfigError(f"[{section}] in {path} must be a table")
        unknown = set(values) - _ALLOWED[section]
        if unknown:
            raise ConfigError(f"unknown option(s) in [{section}] of {path}: {', '.join(sorted(unknown))}")


def _merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for section, values in src.items():
        dst.setdefault(section, {}).update(values)


def _expand_path(value: str, project_dir: Path, *, relative_to_project: bool = True) -> str:
    value = os.path.expanduser(os.path.expandvars(value))
    unresolved = _UNEXPANDED_ENV_RE.search(value)
    if unresolved:
        raise ConfigError(
            f"path contains unresolved environment variable {unresolved.group(0)!r}; "
            "run hummel-submit from a Hummel-2 frontend or define the variable explicitly"
        )
    if not value:
        return value
    path = Path(value)
    if relative_to_project and not path.is_absolute():
        path = project_dir / path
    return str(path.resolve(strict=False))


def load_config(
    project_dir: Path,
    cli_overrides: dict[str, dict[str, Any]] | None = None,
    *,
    require_command: bool = True,
) -> tuple[dict[str, Any], list[Path]]:
    project_dir = project_dir.resolve()
    config = copy.deepcopy(DEFAULTS)
    sources: list[Path] = []

    if USER_CONFIG.exists():
        _merge(config, _load_toml(USER_CONFIG))
        sources.append(USER_CONFIG)

    project_config = project_dir / PROJECT_CONFIG_NAME
    if project_config.exists():
        _merge(config, _load_toml(project_config))
        sources.append(project_config)

    for env_name, (section, key, converter) in _ENV_OVERRIDES.items():
        if env_name in os.environ:
            try:
                config[section][key] = converter(os.environ[env_name])
            except ValueError as exc:
                raise ConfigError(f"invalid {env_name}={os.environ[env_name]!r}") from exc

    if cli_overrides:
        _merge(config, cli_overrides)

    exe = config["execution"]
    exe["output_dir"] = _expand_path(str(exe["output_dir"]), project_dir)
    exe["cache_dir"] = _expand_path(str(exe["cache_dir"]), project_dir)
    if str(exe["image"]).lower() != "none":
        exe["image"] = _expand_path(str(exe["image"]), project_dir)
    else:
        exe["image"] = "none"
    exe["env_file"] = _expand_path(str(exe["env_file"]), project_dir)
    exe["binds"] = [_expand_path(str(p), project_dir, relative_to_project=False) for p in exe["binds"]]
    if exe.get("apptainer"):
        exe["apptainer"] = _expand_path(str(exe["apptainer"]), project_dir)

    validate_config(config, require_command=require_command)
    return config, sources


def validate_config(config: dict[str, Any], *, require_command: bool = True) -> None:
    exe = config["execution"]
    slurm = config["slurm"]
    validation = config["validation"]

    if not isinstance(exe["command"], list) or not all(isinstance(x, str) and x for x in exe["command"]):
        raise ConfigError("[execution].command must be an array of non-empty strings")
    if require_command and not exe["command"]:
        raise ConfigError("[execution].command must not be empty when submitting")
    if not isinstance(exe["auto_args"], list) or not all(isinstance(x, str) for x in exe["auto_args"]):
        raise ConfigError("[execution].auto_args must be an array of strings")
    if not isinstance(exe["binds"], list) or not all(isinstance(x, str) and x for x in exe["binds"]):
        raise ConfigError("[execution].binds must be an array of paths")

    glob = str(exe["checkpoint_glob"])
    if glob:
        gp = Path(glob)
        if gp.is_absolute() or ".." in gp.parts:
            raise ConfigError("[execution].checkpoint_glob must stay inside the run directory (no absolute path or '..')")

    for key in ("nodes", "gpus", "signal_seconds", "max_hops"):
        if not isinstance(slurm[key], int):
            raise ConfigError(f"[slurm].{key} must be an integer")
    if slurm["nodes"] < 1:
        raise ConfigError("[slurm].nodes must be >= 1")
    if slurm["gpus"] < 0:
        raise ConfigError("[slurm].gpus must be >= 0")
    if slurm["signal_seconds"] < 1:
        raise ConfigError("[slurm].signal_seconds must be >= 1")
    if slurm["max_hops"] < 1:
        raise ConfigError("[slurm].max_hops must be >= 1")

    if not isinstance(slurm["retry_on_failure"], bool):
        raise ConfigError("[slurm].retry_on_failure must be true or false")
    if not isinstance(slurm["extra_args"], list) or not all(isinstance(x, str) and x for x in slurm["extra_args"]):
        raise ConfigError("[slurm].extra_args must be an array of argument strings")
    if not isinstance(validation["writable_args"], list) or not all(
        isinstance(x, str) and x.startswith("-") for x in validation["writable_args"]
    ):
        raise ConfigError("[validation].writable_args must be an array of option names such as '--output-dir'")

    _validate_extra_args(slurm["extra_args"])


def _validate_extra_args(args: list[str]) -> None:
    for arg in args:
        option = arg.split("=", 1)[0]
        if option in _FORBIDDEN_MEMORY_OPTIONS:
            raise ConfigError(
                f"{option} is not allowed on Hummel-2: memory is allocated implicitly with virtual nodes; "
                "do not pass SLURM memory request options"
            )
        if option in _RESERVED_SBATCH_OPTIONS:
            raise ConfigError(
                f"{option} is managed by hummel-submit; set the corresponding [slurm] option instead"
            )


def validate_run_name(name: str) -> str:
    if not _RUN_NAME_RE.fullmatch(name):
        raise ConfigError("run name may contain only letters, digits, '.', '_' and '-' and must start alphanumerically")
    return name


def config_as_json(config: dict[str, Any]) -> str:
    return json.dumps(config, indent=2, sort_keys=True)
