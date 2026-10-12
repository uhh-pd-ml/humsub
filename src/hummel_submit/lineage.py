"""Lineage of submissions of one logical run: overview, resume, stale-scratch sweep.

A *lineage* is the original manifest submission plus every ``humsub resume`` (or supervisor)
submission derived from it.  All of them share the same frozen manifest (the full set of branches
and their output paths), so "what is still missing" is always answered from the outputs on disk.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any

from .chain import query_chain
from .cleanup import iter_submission_specs
from .config import ConfigError, validate_config
from .manifest import load_manifest
from .scratch import stale_scratch_dirs
from .state import load_state, chain_root
from .submission import load_submission_spec, submission_root


@dataclass
class Overview:
    lineage_id: str
    submissions: list[str]
    total: int
    done: int
    active: int
    missing: list[int]
    chain_states: dict[str, int] = field(default_factory=dict)
    failed_chains: list[tuple[str, str, list[int]]] = field(default_factory=list)
    hops: int = 0

    @property
    def idle_missing(self) -> int:
        return len(self.missing)


def lineage_id(spec: dict[str, Any]) -> str:
    return str(spec.get("lineage", {}).get("id", spec["submission_id"]))


def lineage_specs(output_dir: Path, lid: str) -> list[tuple[Path, dict[str, Any]]]:
    found = []
    for path in iter_submission_specs(output_dir):
        try:
            spec = load_submission_spec(path)
        except (OSError, json.JSONDecodeError):
            continue
        if lineage_id(spec) == lid:
            found.append((path, spec))
    return found


def find_submission(output_dir: Path, ident: str) -> Path:
    """Resolve a submission id (or unique run name) to its submission.json."""
    direct = submission_root(output_dir) / ident / "submission.json"
    if direct.is_file():
        return direct
    matches = []
    for path in iter_submission_specs(output_dir):
        try:
            spec = load_submission_spec(path)
        except (OSError, json.JSONDecodeError):
            continue
        if spec.get("run_name") == ident or spec["submission_id"].startswith(ident):
            matches.append(path)
    if not matches:
        raise ConfigError(f"no submission or run named {ident!r} under {submission_root(output_dir)}")
    if len(matches) > 1:
        raise ConfigError(f"{ident!r} is ambiguous: " + ", ".join(p.parent.name for p in matches))
    return matches[0]


def chain_branch_map(spec_path: Path) -> dict[str, list[int]]:
    """chain id -> branch ids, from law's control files of one submission."""
    mapping: dict[str, list[int]] = {}
    for control in sorted((spec_path.parent / "law" / "control").glob("slurm_jobs_*.json")):
        try:
            data = json.loads(control.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for job in data.get("jobs", {}).values():
            if isinstance(job, dict) and job.get("job_id"):
                mapping[str(job["job_id"])] = [int(b) for b in job.get("branches", [])]
    return mapping


def root_manifest(spec: dict[str, Any]) -> dict[str, Any]:
    path = Path(spec.get("lineage", {}).get("manifest") or spec["manifest_workflow"]["manifest"])
    return load_manifest(path)


def compute_overview(output_dir: Path, spec: dict[str, Any]) -> Overview:
    lid = lineage_id(spec)
    manifest = root_manifest(spec)
    outputs = {int(b["id"]): b["outputs"] for b in manifest["branches"]}
    done = {bid for bid, outs in outputs.items() if all(Path(o).exists() for o in outs)}

    active_branches: set[int] = set()
    states: dict[str, int] = {}
    failed: list[tuple[str, str, list[int]]] = []
    hops = 0
    submissions = []
    for path, sub in lineage_specs(output_dir, lid):
        submissions.append(sub["submission_id"])
        branch_map = chain_branch_map(path)
        for chain_id in sub.get("chain_ids", []):
            status = query_chain(output_dir, chain_id)
            states[status.state] = states.get(status.state, 0) + 1
            try:
                hops += len(load_state(chain_root(output_dir) / chain_id / "state.json").get("jobs", []))
            except OSError:
                pass
            branches = branch_map.get(chain_id, [])
            if status.state in {"pending", "running"}:
                active_branches.update(branches)
            elif status.state == "failed":
                failed.append((chain_id, status.error or "", [b for b in branches if b not in done]))
    missing = sorted(set(outputs) - done - active_branches)
    return Overview(
        lineage_id=lid,
        submissions=submissions,
        total=len(outputs),
        done=len(done),
        active=len(active_branches - done),
        missing=missing,
        chain_states=states,
        failed_chains=failed,
        hops=hops,
    )


def format_overview(ov: Overview, *, hint: bool = True) -> list[str]:
    lines = [
        f"lineage:    {ov.lineage_id} ({len(ov.submissions)} submission(s): {', '.join(ov.submissions)})",
        f"branches:   {ov.total} total | {ov.done} done | {ov.active} in flight | {len(ov.missing)} missing and idle",
        "chains:     " + (", ".join(f"{n} {s}" for s, n in sorted(ov.chain_states.items())) or "none")
        + f" ({ov.hops} Slurm job(s) used)",
    ]
    for chain_id, error, branches in ov.failed_chains[:20]:
        lines.append(f"  failed chain {chain_id}: {error}; unfinished branches {branches}")
    if len(ov.failed_chains) > 20:
        lines.append(f"  ... and {len(ov.failed_chains) - 20} more failed chain(s)")
    if ov.missing and hint:
        lines.append("next step:  humsub resume <submission>   (re-submits the missing branches)")
    return lines


def sweep_stale_scratch(config: dict[str, Any], submission_ids: set[str] | None, *, apply: bool) -> tuple[int, int]:
    """Remove scratch of Slurm jobs that are gone (killed, crashed, cancelled).  Returns (dirs, bytes)."""
    from .cleanup import _dir_size
    from .slurm import live_job_ids

    exe = config["execution"]
    roots = [Path(exe["cache_dir"]) / "payload-work", Path(exe["bulk_scratch_dir"])]
    stale = stale_scratch_dirs(roots, live_job_ids(), submission_ids)
    total = sum(_dir_size(p) for p in stale)
    if apply:
        for path in stale:
            shutil.rmtree(path, ignore_errors=True)
    return len(stale), total


def _apply_overrides(config: dict[str, Any], overrides: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    import copy
    merged = copy.deepcopy(config)
    for section, values in (overrides or {}).items():
        merged.setdefault(section, {}).update(values)
    validate_config(merged, require_command=False)
    return merged


def resume(
    output_dir: Path,
    spec_path: Path,
    *,
    config_overrides: dict[str, dict[str, Any]] | None = None,
    option_overrides: dict[str, Any] | None = None,
    stage_overrides: dict[str, Path] | None = None,
    dry_run: bool = False,
    include_active: bool = False,
    from_supervisor: bool = False,
) -> dict[str, Any] | None:
    """Submit only the branches of the lineage whose outputs are missing and that no live chain owns."""
    from .manifest_submit import SubmitOptions, submit_manifest_run

    base = load_submission_spec(spec_path)
    ov = compute_overview(output_dir, base)
    for line in format_overview(ov, hint=False):
        print(f"[resume] {line}")
    config = _apply_overrides(base["config"], config_overrides)
    swept, swept_bytes = sweep_stale_scratch(config, set(ov.submissions), apply=not dry_run)
    if swept:
        print(f"[resume] {'would remove' if dry_run else 'removed'} scratch of {swept} dead job(s) ({swept_bytes / 1e6:.0f} MB)")

    manifest = root_manifest(base)
    wanted: set[int] = set(ov.missing)
    if include_active:
        wanted |= {int(b["id"]) for b in manifest["branches"]} - {
            int(b["id"]) for b in manifest["branches"] if all(Path(o).exists() for o in b["outputs"])
        }
    if not wanted:
        print("[resume] nothing to resume" + (" (all missing branches are still in flight)" if ov.active else ""))
        return None

    opts_dict = dict(base.get("submit_options", {}))
    opts_dict.update(option_overrides or {})
    opts_dict.setdefault("wait", False)
    if "wait" not in (option_overrides or {}):
        opts_dict["wait"] = False
        opts_dict["retries"] = 0
        opts_dict["parallel_jobs"] = 0
    if "supervisor_job" not in (option_overrides or {}):
        opts_dict["supervisor_job"] = False
    opts = SubmitOptions.from_dict(opts_dict)

    sources = {k: Path(v) for k, v in base["manifest_workflow"].get("stage_sources", {}).items()}
    sources.update(stage_overrides or {})
    excludes = {k: list(v) for k, v in base["manifest_workflow"].get("stage_excludes", {}).items()}
    for name in list(excludes):
        if name not in sources:
            del excludes[name]

    sdir = spec_path.parent
    resume_dir = sdir / "resume"
    resume_dir.mkdir(exist_ok=True)
    subset = dict(manifest)
    subset["branches"] = [b for b in manifest["branches"] if int(b["id"]) in wanted]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    manifest_path = resume_dir / f"manifest-{stamp}.json"
    manifest_path.write_text(json.dumps(subset, indent=1), encoding="utf-8")

    root_spec_run = base["run_name"]
    suffix_base = root_spec_run.split("-resume")[0]
    run_name = f"{suffix_base}-resume{len(ov.submissions)}"
    payload = Path(base.get("lineage", {}).get("payload") or base["manifest_workflow"]["payload"])
    print(f"[resume] resubmitting {len(subset['branches'])} of {ov.total} branches as run {run_name}")
    return submit_manifest_run(
        project_dir=Path(base["project_dir"]),
        config=config,
        sources=[],
        run_name=run_name,
        manifest_source=manifest_path,
        payload_source=payload,
        stages=sources,
        stage_excludes=excludes,
        opts=opts,
        dry_run=dry_run,
        lineage=base.get("lineage") or {"id": base["submission_id"], "manifest": base["manifest_workflow"]["manifest"], "payload": str(payload)},
        parent_submission=base["submission_id"],
        allow_supervisor=not from_supervisor,
    )
