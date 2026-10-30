"""``humsub make-manifest``: build a manifest from a folder search, for "input file(s) -> known output file(s)" work.

Nothing is listed per input file: the inputs are found by searching folders, each input's declared outputs follow
from templates, and inputs are packed into branches of ``--files-per-branch``.  Branch data is
``{"inputs": [absolute paths]}``; ``common`` is ``--common``/``--common-json`` plus every ``--set KEY=VALUE``.

Template fields: ``{stem}``, ``{name}``, ``{parent}`` of the input file and every ``--set`` key (e.g. ``{out}``).
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from .config import ConfigError


def find_inputs(
    search: list[Path], pattern: str = "*", *, require: str | None = None, select: str | None = None
) -> tuple[list[Path], int]:
    """(selected files, number found): every regular file below *search* matching the glob *pattern*.

    ``require``: every file name found must fullmatch it, else stop (an unexpected file is a problem to look at,
    not something to skip).  ``select``: only the file names that fullmatch it are returned."""
    found: set[Path] = set()
    for root in search:
        if not root.is_dir():
            raise ConfigError(f"--search {root}: not a folder")
        found.update(p.resolve() for p in root.rglob(pattern) if p.is_file())
    ordered = sorted(found)
    if require:
        bad = [p for p in ordered if not re.fullmatch(require, p.name)]
        if bad:
            listing = "\n  ".join(str(p) for p in bad[:20]) + ("\n  ..." if len(bad) > 20 else "")
            raise ConfigError(f"{len(bad)} file(s) do not match --require {require!r}:\n  {listing}")
    selected = [p for p in ordered if not select or re.fullmatch(select, p.name)]
    return selected, len(ordered)


def _expand(template: str, fields: dict[str, str], what: str) -> str:
    try:
        value = template.format(**fields)
    except (KeyError, IndexError) as exc:
        raise ConfigError(f"{what} template {template!r} uses an unknown field {exc}; known: {', '.join(sorted(fields))}") from exc
    if not Path(value).is_absolute():
        raise ConfigError(f"{what} {value!r} is not an absolute path")
    return value


def build_manifest(
    files: list[Path],
    *,
    outputs: list[str],
    files_per_branch: int = 1,
    common: dict[str, Any] | None = None,
    values: dict[str, str] | None = None,
    finalize_outputs: list[str] | None = None,
    finalize_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not files:
        raise ConfigError("no input files found")
    if not outputs:
        raise ConfigError("give at least one --output template")
    if files_per_branch < 1:
        raise ConfigError("--files-per-branch must be >= 1")
    values = dict(values or {})
    owner: dict[str, Path] = {}
    outputs_of: dict[Path, list[str]] = {}
    for path in files:
        fields = dict(values, stem=path.stem, name=path.name, parent=str(path.parent))
        outs = [_expand(t, fields, "output") for t in outputs]
        for out in outs:
            if out in owner:
                raise ConfigError(f"{path} and {owner[out]} would both write {out}")
            owner[out] = path
        outputs_of[path] = outs

    chunks = [files[i : i + files_per_branch] for i in range(0, len(files), files_per_branch)]
    manifest: dict[str, Any] = {
        "schema": 1,
        "common": dict(common or {}, **values),
        "branches": [
            {"id": i, "data": {"inputs": [str(p) for p in chunk]}, "outputs": [o for p in chunk for o in outputs_of[p]]}
            for i, chunk in enumerate(chunks)
        ],
    }
    if finalize_outputs:
        fin_outs = [_expand(t, values, "finalize output") for t in finalize_outputs]
        clash = [o for o in fin_outs if o in owner]
        if clash:
            raise ConfigError(f"finalize output(s) {clash} are also branch outputs")
        manifest["finalize"] = {"data": dict(finalize_data or {}), "outputs": fin_outs}
    elif finalize_data:
        raise ConfigError("--finalize-data needs at least one --finalize-output (the outputs mark the step as done)")
    return manifest


def parse_assignments(items: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.isidentifier():
            raise ConfigError(f"--set expects KEY=VALUE with an identifier as KEY, got {item!r}")
        values[key] = value
    return values


def json_object(text: str, what: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{what} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"{what} must be a JSON object")
    return value
