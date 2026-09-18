from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Iterable


class PathCheckError(ValueError):
    pass


@dataclass(frozen=True)
class PathCheck:
    label: str
    path: Path
    resolved: Path
    storage: str
    status: str
    detail: str


_URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_OUTPUT_OPTIONS = {
    "-o",
    "--out",
    "--output",
    "--output-dir",
    "--output-path",
    "--output-file",
    "--out-dir",
    "--out-path",
    "--out-file",
    "--save-dir",
    "--save-path",
    "--save-file",
    "--checkpoint-dir",
    "--checkpoint-path",
    "--checkpoint-file",
    "--ckpt-dir",
    "--ckpt-path",
    "--ckpt-file",
    "--log-dir",
    "--log-path",
    "--log-file",
    "--cache-dir",
    "--cache-path",
    "--tmp-dir",
    "--temp-dir",
    "--work-dir",
    "--workdir",
    "--result-dir",
    "--results-dir",
    "--destination",
    "--dest",
    "--export-dir",
    "--export-path",
}


def _resolved(path: Path) -> Path:
    """Resolve symlinks in every existing prefix without requiring the leaf to exist."""
    try:
        return path.resolve(strict=False)
    except OSError:
        return path.absolute()


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _storage_kind(path: Path) -> str:
    resolved = _resolved(path)
    text = str(resolved)

    # Site-managed temporary/node-specific locations need special handling even
    # when they are nested below a normal storage prefix.
    if text.startswith("/beegfs/tmp/") or text == "/beegfs/tmp" or text.startswith("/beegfs/scratch/") or text == "/beegfs/scratch":
        return "beegfs_tmp"
    if text == "/tmp" or text.startswith("/tmp/") or text == "/dev/shm" or text.startswith("/dev/shm/"):
        return "ramtmp"
    if text.startswith("/nvmeof/ssd"):
        return "nvmeof"

    # Prefer Hummel's own environment variables. This also makes classification
    # work through site-specific symlinks and for group storage variables.
    for name, value in os.environ.items():
        if not value:
            continue
        if name == "BEEGFS" or name.startswith("BEEGFS_"):
            kind = "beegfs"
        elif name == "SSD" or name.startswith("SSD_"):
            kind = "ssd"
        elif name == "USW" or name.startswith("USW_"):
            kind = "usw"
        elif name == "HOME":
            kind = "home"
        else:
            continue
        root = _resolved(Path(value))
        if _is_below(resolved, root):
            return kind

    # Fallbacks cover group paths and configurations where the variables were
    # not exported into the launcher environment.
    if text == "/home" or text.startswith("/home/"):
        return "home"
    if text == "/usw" or text.startswith("/usw/"):
        return "usw"
    if text == "/beegfs" or text.startswith("/beegfs/"):
        return "beegfs"
    if text.startswith("/nfs/ssd") or text.startswith("/ssd"):
        return "ssd"
    return "unknown"


def _nearest_existing(path: Path) -> Path | None:
    current = path
    while True:
        if current.exists():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _frontend_writable(path: Path) -> tuple[bool, str]:
    """Check whether path itself, or the nearest parent needed to create it, is writable now."""
    resolved = _resolved(path)
    if resolved.exists():
        if resolved.is_dir():
            ok = os.access(resolved, os.W_OK | os.X_OK)
            return ok, f"directory {'is' if ok else 'is not'} writable on the submission node"
        ok = os.access(resolved, os.W_OK)
        return ok, f"file {'is' if ok else 'is not'} writable on the submission node"

    ancestor = _nearest_existing(resolved.parent)
    if ancestor is None:
        return False, "no existing parent directory could be found"
    ok = os.access(ancestor, os.W_OK | os.X_OK)
    return ok, f"nearest existing parent {ancestor} {'is' if ok else 'is not'} writable on the submission node"


def check_compute_writable(path: Path, label: str, *, must_be_shared: bool = False) -> PathCheck:
    resolved = _resolved(path)
    storage = _storage_kind(resolved)

    if storage in {"home", "usw"}:
        raise PathCheckError(
            f"{label} {path} resolves to {resolved}, which is on /{storage}; "
            "Hummel-2 mounts /home and /usw read-only in batch jobs"
        )
    if must_be_shared and storage in {"ramtmp", "beegfs_tmp", "nvmeof"}:
        explanations = {
            "ramtmp": "/tmp and /dev/shm are job-private RAM filesystems",
            "beegfs_tmp": "Hummel temporary BeeGFS directories are automatically managed/deleted scratch space",
            "nvmeof": "NVMe-oF storage is not guaranteed to be available on every compute node",
        }
        raise PathCheckError(
            f"{label} {path} cannot hold state shared between resubmission hops: {explanations[storage]}"
        )

    writable, detail = _frontend_writable(resolved)
    if not writable:
        raise PathCheckError(f"{label} {path}: {detail}")

    if must_be_shared and storage == "unknown":
        return PathCheck(
            label, path, resolved, storage, "warning",
            f"{detail}; filesystem is not recognized as Hummel-2 BeeGFS/SSD, so compute-node visibility cannot be guaranteed",
        )
    if storage == "ramtmp":
        return PathCheck(
            label, path, resolved, storage, "warning",
            f"{detail}; /tmp and /dev/shm are RAM-backed, job-private, and deleted at job end",
        )
    if storage == "beegfs_tmp":
        return PathCheck(
            label, path, resolved, storage, "warning",
            f"{detail}; this is Hummel-managed temporary BeeGFS scratch space and is automatically deleted",
        )
    if storage == "nvmeof":
        return PathCheck(
            label, path, resolved, storage, "warning",
            f"{detail}; NVMe-oF storage is node-specific and may not be available to a scheduled job",
        )
    if storage == "beegfs":
        return PathCheck(label, path, resolved, storage, "ok", f"{detail}; BeeGFS is writable in batch jobs")
    if storage == "ssd":
        return PathCheck(label, path, resolved, storage, "ok", f"{detail}; SSD storage is writable in batch jobs")
    return PathCheck(
        label, path, resolved, storage, "warning",
        f"{detail}; filesystem is not recognized, so compute-node writability cannot be guaranteed",
    )


def _looks_like_path(value: str) -> bool:
    if not value or _URI_RE.match(value) or value in {"-", "null", "none", "None"}:
        return False
    return value.startswith(("/", "./", "../", "~")) or "/" in value


def _arg_candidates(args: list[str], extra_writable: Iterable[str]) -> list[tuple[str, str, bool]]:
    extra = set(extra_writable)
    candidates: list[tuple[str, str, bool]] = []
    i = 0
    while i < len(args):
        token = args[i]
        if token.startswith("-") and "=" in token:
            option, value = token.split("=", 1)
            normalized = option.replace("_", "-")
            writable = normalized in _OUTPUT_OPTIONS or option in extra or normalized in extra
            if writable or _looks_like_path(value):
                candidates.append((option, value, writable))
        elif token.startswith("-"):
            option = token
            normalized = option.replace("_", "-")
            writable = normalized in _OUTPUT_OPTIONS or option in extra or normalized in extra
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                value = args[i + 1]
                if writable or _looks_like_path(value):
                    candidates.append((option, value, writable))
                    i += 1
        elif _looks_like_path(token):
            candidates.append((f"argument {i + 1}", token, False))
        i += 1
    return candidates


def _path_from_arg(value: str, project_dir: Path) -> Path | None:
    if _URI_RE.match(value) or "{" in value or "}" in value:
        return None
    expanded = os.path.expanduser(os.path.expandvars(value))
    path = Path(expanded)
    if not path.is_absolute():
        path = project_dir / path
    return path


def check_payload_args(
    args: list[str],
    project_dir: Path,
    *,
    extra_writable: Iterable[str] = (),
) -> list[PathCheck]:
    """Conservatively inspect payload arguments that look like paths.

    Existing paths are allowed as inputs, including paths under /home and /usw.
    A path is required to be compute-writable when its option clearly denotes an
    output, or when it does not exist and is therefore likely to be created by the job.
    Unknown filesystems produce warnings instead of hard failures.
    """
    checks: list[PathCheck] = []
    for option, value, explicitly_writable in _arg_candidates(args, extra_writable):
        path = _path_from_arg(value, project_dir)
        if path is None:
            continue
        resolved = _resolved(path)
        exists = resolved.exists()
        needs_write = explicitly_writable or not exists

        if not needs_write:
            if not os.access(resolved, os.R_OK):
                raise PathCheckError(f"payload {option} path {value!r} exists but is not readable on the submission node")
            checks.append(PathCheck(
                f"payload {option}", path, resolved, _storage_kind(resolved), "ok",
                "existing path treated as an input; readable on the submission node",
            ))
            continue

        check = check_compute_writable(path, f"payload {option}")
        checks.append(check)
    return checks
