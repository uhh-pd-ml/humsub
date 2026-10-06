from __future__ import annotations

from pathlib import Path
import os
import re
import shutil
import subprocess


_STAGE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


def validate_stage_name(name: str) -> str:
    if not _STAGE_NAME_RE.fullmatch(name):
        raise ValueError(
            "stage names must start with a letter and contain only letters, digits, '.', '_' and '-'"
        )
    return name


def parse_stage_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--stage must have the form NAME=PATH")
    name, raw_path = value.split("=", 1)
    validate_stage_name(name)
    if not raw_path:
        raise ValueError(f"stage {name!r} has an empty path")
    expanded = os.path.expandvars(os.path.expanduser(raw_path))
    if "$" in expanded:
        raise ValueError(f"unresolved environment variable in stage path: {raw_path!r}")
    return name, Path(expanded).absolute()


def parse_stage_exclude(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError("--stage-exclude must have the form NAME=PATTERN")
    name, pattern = value.split("=", 1)
    validate_stage_name(name)
    if not pattern:
        raise ValueError(f"stage exclude for {name!r} has an empty pattern")
    if "\0" in pattern:
        raise ValueError("stage exclude patterns must not contain NUL bytes")
    return name, pattern


def stage_input(source: Path, destination: Path, *, excludes: list[str] | tuple[str, ...] = ()) -> None:
    """Copy a file or directory into a unique submission-owned staging path."""
    source = source.expanduser().absolute()
    if not source.exists():
        raise FileNotFoundError(f"stage source does not exist: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if shutil.which("rsync"):
            destination.mkdir(parents=True, exist_ok=True)
            cmd = ["rsync", "-a", "--delete"]
            for pattern in excludes:
                cmd.extend(["--exclude", pattern])
            cmd.extend([f"{source}/", f"{destination}/"])
            subprocess.run(cmd, check=True)
        else:
            if destination.exists():
                shutil.rmtree(destination)
            ignore = shutil.ignore_patterns(*excludes) if excludes else None
            shutil.copytree(source, destination, symlinks=True, ignore=ignore)
        return

    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    shutil.copy2(source, destination, follow_symlinks=False)


def stage_inputs(
    sources: dict[str, Path],
    *,
    cache_dir: Path,
    submission_id: str,
    excludes: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    root = cache_dir / "stages" / submission_id
    excludes = excludes or {}
    unknown = sorted(set(excludes) - set(sources))
    if unknown:
        raise ValueError("stage excludes reference unknown stage(s): " + ", ".join(unknown))
    staged: dict[str, str] = {}
    for name, source in sources.items():
        validate_stage_name(name)
        destination = root / name
        stage_input(source, destination, excludes=excludes.get(name, ()))
        staged[name] = str(destination)
    return staged
