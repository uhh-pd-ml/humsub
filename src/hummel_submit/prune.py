"""``humsub prune``: delete the provenance (state, chains, run directory, Slurm logs, caches) of finished submissions.

humsub never deletes provenance on its own (``gc`` only removes SSD caches).  After a production has been verified and its results
moved, the provenance is dead weight — but deleting it by date or by name is risky because an ``output_dir`` accumulates the runs of
many projects.  ``prune`` therefore selects *submissions* and deletes only what they own: their submission directory, their chain
directories, their run directory, the Slurm logs of exactly their job ids (and of their supervisor jobs), and their SSD/bulk caches.
Everything else in the ``output_dir`` — older runs, other projects' logs, declared outputs of the manifest — is left alone.
Dry run unless ``--apply``; active submissions are refused unless ``--force``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
from typing import Any

from .cleanup import CachePath, _dir_size, inspect_submission_cache, iter_submission_specs
from .state import chain_root, load_state
from .submission import load_submission_spec, submission_root


@dataclass
class PrunePlan:
    submission_id: str
    run_name: str
    state: str
    paths: list[tuple[str, Path]] = field(default_factory=list)       # (label, path)
    skipped: list[str] = field(default_factory=list)

    def size(self) -> int:
        return sum(_dir_size(p) for _, p in self.paths)


def select_submissions(output_dir: Path, runs: list[str], before: str | None, ids: list[str]) -> list[Path]:
    """submission.json paths matching run names, ids (prefix) or created before ``YYYY-MM-DD``."""
    chosen: list[Path] = []
    for path in iter_submission_specs(output_dir):
        spec = load_submission_spec(path)
        created = str(spec.get("created_at", ""))[:10]
        if spec.get("run_name") in runs or any(spec["submission_id"].startswith(i) for i in ids) or (before and created and created < before):
            chosen.append(path)
    return chosen


def _declared_outputs(spec: dict[str, Any]) -> list[Path]:
    try:
        manifest = load_submission_spec(Path(spec["manifest_workflow"]["manifest"]))
    except (OSError, KeyError, ValueError):
        return []
    return [Path(o) for b in manifest.get("branches", []) for o in b.get("outputs", []) + b.get("extra_outputs", [])]


def plan_submission(output_dir: Path, spec_path: Path) -> PrunePlan:
    spec = load_submission_spec(spec_path)
    status = inspect_submission_cache(spec_path)
    plan = PrunePlan(spec["submission_id"], spec["run_name"], status.state)
    job_name = str(spec["config"]["slurm"]["job_name"])
    logs = output_dir / "logs"
    candidates: list[tuple[str, Path]] = [("submission", spec_path.parent)]
    for chain_id in spec.get("chain_ids", []):
        cdir = chain_root(output_dir) / chain_id
        candidates.append(("chain", cdir))
        try:
            state = load_state(cdir / "state.json")
        except OSError:
            continue
        candidates += [("log", logs / f"{job_name}_{job}.log") for job in state.get("jobs", [])]
    candidates += [("supervisor log", logs / f"{job_name}-supervisor_{job}.log") for job in spec.get("supervisor_job_ids", [])]
    if spec.get("run_dir"):
        candidates.append(("run dir", Path(spec["run_dir"])))
    candidates += [(c.label, c.path) for c in status.cache_paths]

    outputs = _declared_outputs(spec)
    for label, path in candidates:
        if not (path.exists() or path.is_symlink()):
            continue
        resolved = path.resolve()
        hit = next((o for o in outputs if o.resolve() == resolved or resolved in o.resolve().parents), None)
        if hit is not None:      # never delete anything that contains a declared output (e.g. a run_dir that was reused as output dir)
            plan.skipped.append(f"{label} {path} contains the declared output {hit}")
            continue
        plan.paths.append((label, path))
    return plan


def apply_plan(plan: PrunePlan) -> None:
    for _, path in plan.paths:
        if path.is_dir() and not path.is_symlink():
            for root, dirs, _ in os.walk(path):          # trees with read-only directories (e.g. unpacked toolchains) need u+w first
                for d in dirs:
                    try:
                        os.chmod(os.path.join(root, d), 0o755)
                    except OSError:
                        pass
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def fmt(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return str(n)
