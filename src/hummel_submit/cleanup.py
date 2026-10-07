from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
from typing import Iterable

from .chain import query_chain
from .state import chain_root
from .submission import load_submission_spec, submission_root


@dataclass(frozen=True)
class CachePath:
    label: str
    path: Path
    bytes: int


@dataclass(frozen=True)
class SubmissionCacheStatus:
    submission_id: str
    state: str
    terminal_age_hours: float | None
    cache_paths: tuple[CachePath, ...]

    @property
    def active(self) -> bool:
        return self.state == "active"

    @property
    def terminal(self) -> bool:
        return self.state in {"successful", "failed"}


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        try:
            return path.lstat().st_size
        except OSError:
            return 0
    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in files:
            try:
                total += (root_path / name).lstat().st_size
            except OSError:
                pass
        for name in dirs:
            child = root_path / name
            if child.is_symlink():
                try:
                    total += child.lstat().st_size
                except OSError:
                    pass
    return total


def _cache_paths(spec: dict) -> tuple[CachePath, ...]:
    cache_dir = Path(spec["config"]["execution"]["cache_dir"])
    submission_id = str(spec["submission_id"])
    candidates = (
        ("stages", cache_dir / "stages" / submission_id),
        ("payload-work", cache_dir / "payload-work" / submission_id),
    )
    return tuple(CachePath(label, path, _dir_size(path)) for label, path in candidates if path.exists())


def _chain_state_paths(spec: dict) -> list[Path]:
    output_dir = Path(spec["output_dir"])
    return [chain_root(output_dir) / chain_id / "state.json" for chain_id in spec.get("chain_ids", [])]


def inspect_submission_cache(spec_path: Path, *, now: datetime | None = None) -> SubmissionCacheStatus:
    spec = load_submission_spec(spec_path)
    output_dir = Path(spec["output_dir"])
    chain_ids = list(spec.get("chain_ids", []))
    if not chain_ids:
        state = "prepared"
    else:
        effective = [query_chain(output_dir, chain_id) for chain_id in chain_ids]
        if any(not status.terminal for status in effective):
            state = "active"
        elif all(status.state == "finished" for status in effective):
            state = "successful"
        else:
            state = "failed"

    terminal_age_hours: float | None = None
    if state in {"successful", "failed"}:
        timestamps: list[float] = []
        for path in _chain_state_paths(spec):
            try:
                timestamps.append(path.stat().st_mtime)
            except OSError:
                pass
        try:
            timestamps.append(spec_path.stat().st_mtime)
        except OSError:
            pass
        if timestamps:
            current = (now or datetime.now(timezone.utc)).timestamp()
            terminal_age_hours = max(0.0, (current - max(timestamps)) / 3600.0)

    return SubmissionCacheStatus(
        submission_id=str(spec["submission_id"]),
        state=state,
        terminal_age_hours=terminal_age_hours,
        cache_paths=_cache_paths(spec),
    )


def cleanup_submission_cache(spec_path: Path, *, force: bool = False, dry_run: bool = False) -> SubmissionCacheStatus:
    status = inspect_submission_cache(spec_path)
    if status.active and not force:
        raise RuntimeError(
            f"submission {status.submission_id} is still active; refusing to remove runtime caches (use --force to override)"
        )
    if not dry_run:
        remove_cache_paths(status.cache_paths)
    return status


def iter_submission_specs(output_dir: Path) -> Iterable[Path]:
    root = submission_root(output_dir)
    if not root.exists():
        return ()
    return sorted(root.glob("*/submission.json"))


def stale_orphan_cache_dirs(
    cache_dir: Path,
    known_submission_ids: set[str],
    *,
    older_than_hours: float,
    now: datetime | None = None,
) -> list[CachePath]:
    current = (now or datetime.now(timezone.utc)).timestamp()
    found: list[CachePath] = []
    for kind in ("stages", "payload-work"):
        root = cache_dir / kind
        if not root.exists():
            continue
        for path in sorted(root.iterdir()):
            if path.name in known_submission_ids:
                continue
            try:
                age_hours = (current - path.stat().st_mtime) / 3600.0
            except OSError:
                continue
            if age_hours >= older_than_hours:
                found.append(CachePath(f"orphan-{kind}", path, _dir_size(path)))
    return found


def remove_cache_paths(entries: Iterable[CachePath]) -> None:
    for entry in entries:
        if entry.path.is_dir() and not entry.path.is_symlink():
            shutil.rmtree(entry.path, ignore_errors=False)
        else:
            entry.path.unlink(missing_ok=True)
