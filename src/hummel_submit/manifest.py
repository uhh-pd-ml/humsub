from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from .state import atomic_write_json
from .submission import load_submission_spec


class ManifestError(ValueError):
    pass


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ManifestError("manifest root must be a JSON object")
    if data.get("schema") != 1:
        raise ManifestError("manifest schema must be 1")

    common = data.get("common", {})
    if not isinstance(common, dict):
        raise ManifestError("manifest common field must be an object")

    hint = data.get("humsub", {})
    if not isinstance(hint, dict):
        raise ManifestError("manifest humsub field must be an object")
    unknown = set(hint) - {"scratch", "scratch_bytes_per_branch"}
    if unknown:
        raise ManifestError(f"unknown manifest humsub hint(s): {', '.join(sorted(unknown))}")
    if "scratch" in hint and hint["scratch"] not in ("beegfs", "ssd"):
        raise ManifestError("manifest humsub.scratch must be 'beegfs' or 'ssd'")
    if "scratch_bytes_per_branch" in hint and (
        not isinstance(hint["scratch_bytes_per_branch"], int) or hint["scratch_bytes_per_branch"] < 0
    ):
        raise ManifestError("manifest humsub.scratch_bytes_per_branch must be a non-negative integer")

    branches = data.get("branches")
    if not isinstance(branches, list) or not branches:
        raise ManifestError("manifest branches must be a non-empty array")

    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for i, branch in enumerate(branches):
        if not isinstance(branch, dict):
            raise ManifestError(f"branch {i} must be an object")
        branch_id = branch.get("id")
        if not isinstance(branch_id, int) or branch_id < 0:
            raise ManifestError(f"branch {i} has invalid id {branch_id!r}; expected a non-negative integer")
        if branch_id in seen:
            raise ManifestError(f"duplicate branch id {branch_id}")
        seen.add(branch_id)

        payload_data = branch.get("data", {})
        if not isinstance(payload_data, dict):
            raise ManifestError(f"branch {branch_id} data must be an object")

        outputs = branch.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            raise ManifestError(f"branch {branch_id} outputs must be a non-empty array")
        normalized_outputs: list[str] = []
        for output in outputs:
            if not isinstance(output, str) or not output:
                raise ManifestError(f"branch {branch_id} has an invalid output path {output!r}")
            path_obj = Path(output).expanduser()
            if not path_obj.is_absolute():
                raise ManifestError(f"branch {branch_id} output must be absolute: {output}")
            normalized_outputs.append(str(path_obj))

        extra = branch.get("extra_outputs", [])
        if not isinstance(extra, list):
            raise ManifestError(f"branch {branch_id} extra_outputs must be an array")
        normalized_extra: list[str] = []
        for output in extra:
            if not isinstance(output, str) or not output:
                raise ManifestError(f"branch {branch_id} has an invalid extra output path {output!r}")
            path_obj = Path(output).expanduser()
            if not path_obj.is_absolute():
                raise ManifestError(f"branch {branch_id} extra output must be absolute: {output}")
            if str(path_obj) in normalized_outputs:
                raise ManifestError(f"branch {branch_id}: {path_obj} is listed as both output and extra output")
            normalized_extra.append(str(path_obj))

        entry = {
            "id": branch_id,
            "data": payload_data,
            "outputs": normalized_outputs,
        }
        if normalized_extra:
            entry["extra_outputs"] = normalized_extra
        if "output_groups" in branch:
            entry["output_groups"] = _load_output_groups(branch["output_groups"], normalized_outputs, branch_id)
        normalized.append(entry)

    result = {
        "schema": 1,
        "common": common,
        "branches": sorted(normalized, key=lambda b: b["id"]),
    }
    if hint:
        result["humsub"] = hint
    if "finalize" in data:
        result["finalize"] = _load_finalize(data["finalize"], normalized)
    return result


def _load_output_groups(groups: Any, outputs: list[str], branch_id: int) -> list[list[str]]:
    """Optional ``output_groups``: the branch's outputs split into groups that are complete together (e.g. the outputs of
    one input file when several inputs share a branch).  Every output is in exactly one group."""
    if not isinstance(groups, list) or not groups:
        raise ManifestError(f"branch {branch_id} output_groups must be a non-empty array of arrays")
    normalized: list[list[str]] = []
    for group in groups:
        if not isinstance(group, list) or not group or not all(isinstance(o, str) and o for o in group):
            raise ManifestError(f"branch {branch_id} output_groups must be non-empty arrays of output paths")
        normalized.append([str(Path(o).expanduser()) for o in group])
    flat = [o for group in normalized for o in group]
    if sorted(flat) != sorted(outputs):
        raise ManifestError(f"branch {branch_id}: output_groups must contain every output exactly once")
    return normalized


def output_groups(branch: dict[str, Any]) -> list[list[str]]:
    """The branch's output groups; without ``output_groups`` all outputs form one group."""
    return branch.get("output_groups") or [list(branch["outputs"])]


def _load_finalize(fin: Any, branches: list[dict[str, Any]]) -> dict[str, Any]:
    """The optional ``finalize`` step: one extra payload run once every branch is complete (e.g. a merge)."""
    if not isinstance(fin, dict):
        raise ManifestError("manifest finalize field must be an object")
    unknown = set(fin) - {"data", "outputs"}
    if unknown:
        raise ManifestError(f"unknown manifest finalize field(s): {', '.join(sorted(unknown))}")
    payload_data = fin.get("data", {})
    if not isinstance(payload_data, dict):
        raise ManifestError("manifest finalize data must be an object")
    outputs = fin.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ManifestError("manifest finalize outputs must be a non-empty array (they mark the step as done)")
    branch_outputs = {o for b in branches for o in b["outputs"]}
    normalized: list[str] = []
    for output in outputs:
        if not isinstance(output, str) or not output or not Path(output).expanduser().is_absolute():
            raise ManifestError(f"manifest finalize output must be an absolute path: {output!r}")
        path = str(Path(output).expanduser())
        if path in branch_outputs:
            raise ManifestError(f"manifest finalize output {path} is also a branch output")
        normalized.append(path)
    return {"data": payload_data, "outputs": normalized}


def freeze_manifest_workflow(
    spec_path: Path,
    *,
    manifest_source: Path,
    payload_source: Path,
    stages: dict[str, str],
    workflow_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze a generic branch manifest and payload into a submission."""
    spec = load_submission_spec(spec_path)
    manifest = load_manifest(manifest_source)

    root = spec_path.parent / "manifest-workflow"
    payload_dir = root / "payload"
    branch_dir = root / "branches"
    payload_dir.mkdir(parents=True, exist_ok=True)
    branch_dir.mkdir(parents=True, exist_ok=True)

    frozen_manifest = root / "manifest.json"
    atomic_write_json(frozen_manifest, manifest)

    payload_source = payload_source.expanduser().absolute()
    if not payload_source.is_file():
        raise FileNotFoundError(f"payload does not exist: {payload_source}")
    frozen_payload = payload_dir / payload_source.name
    shutil.copy2(payload_source, frozen_payload)
    frozen_payload.chmod(frozen_payload.stat().st_mode | 0o110)

    branch_files: dict[str, str] = {}
    for branch in manifest["branches"]:
        branch_id = int(branch["id"])
        context = {
            "schema": 1,
            "submission_id": spec["submission_id"],
            "run_name": spec["run_name"],
            "run_dir": spec["run_dir"],
            "branch": branch_id,
            "common": manifest["common"],
            "data": branch["data"],
            "outputs": branch["outputs"],
            "extra_outputs": branch.get("extra_outputs", []),
            "stages": stages,
        }
        branch_path = branch_dir / f"{branch_id:06d}.json"
        atomic_write_json(branch_path, context)
        branch_files[str(branch_id)] = str(branch_path)

    finalize_file = None
    if manifest.get("finalize"):
        fin = manifest["finalize"]
        finalize_file = branch_dir / "finalize.json"
        atomic_write_json(finalize_file, {
            "schema": 1,
            "submission_id": spec["submission_id"],
            "run_name": spec["run_name"],
            "run_dir": spec["run_dir"],
            "branch": "finalize",
            "common": manifest["common"],
            "data": dict(fin["data"], finalize=True),
            "outputs": fin["outputs"],
            "extra_outputs": [],
            "stages": stages,
        })

    spec["manifest_workflow"] = {
        "schema": 1,
        "manifest": str(frozen_manifest),
        "payload": str(frozen_payload),
        "branch_dir": str(branch_dir),
        "branch_files": branch_files,
        "branch_count": len(manifest["branches"]),
        "stages": stages,
    }
    if finalize_file is not None:
        spec["manifest_workflow"]["finalize_file"] = str(finalize_file)
    if workflow_extra:
        spec["manifest_workflow"].update(workflow_extra)
    atomic_write_json(spec_path, spec)
    return spec
