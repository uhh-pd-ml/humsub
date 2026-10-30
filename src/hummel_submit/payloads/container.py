#!/usr/bin/env python3
"""Built-in payload ``builtin:container``: run the manifest's command for a branch inside an Apptainer image.

Use it with ``humsub submit-manifest --payload builtin:container``; it is copied into the submission like any payload.
Stdlib only.  Reads from the branch context:

``common.container`` (object)
    ``image``      the .sif (required)
    ``binds``      bind paths (default ``["/beegfs"]``); the stages and the branch scratch are always bound
    ``env``        variables set inside the container, e.g. ``{"PYTHONPATH": "{stage_code}"}``
    ``workdir``    working directory inside the container, e.g. ``"{stage_code}"``
    ``apptainer``  the apptainer binary (default: on ``PATH``, else the newest under /sw/env/system-gcc/apptainer)
    ``nv``         pass ``--nv`` (GPUs), default false
``data.command`` or ``common.command`` (argv list; the branch's own wins, so a finalize step can run another command)
    An element that is exactly ``"{inputs}"`` / ``"{outputs}"`` expands to ``data.inputs`` / the declared outputs.
    ``{field}`` anywhere is replaced from the string values of ``common`` and ``data`` (``data`` wins), plus
    ``stage_<name>`` (staged paths), ``run_dir``, ``branch``, ``scratch`` and ``scratch_fast``.

The host's PYTHONPATH (humsub's own worker code) is not forwarded into the container.  ``execv`` keeps the process
id, so humsub's signal forwarding reaches apptainer directly.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any

APPTAINER_ROOT = Path("/sw/env/system-gcc/apptainer")


class _Fields(dict):
    def __missing__(self, key: str) -> str:
        raise KeyError(f"unknown field {{{key}}} in the container payload command; known: {', '.join(sorted(self))}")


def find_apptainer(configured: str | None = None) -> str:
    if configured:
        return configured
    found = shutil.which("apptainer")
    if found:
        return found

    def version(path: Path) -> tuple[int, ...]:
        return tuple(int(x) for x in re.findall(r"\d+", path.parent.parent.name))

    candidates = sorted(APPTAINER_ROOT.glob("*/bin/apptainer"), key=version)
    if not candidates:
        raise SystemExit("container payload: no apptainer on PATH or under /sw/env/system-gcc/apptainer; "
                         "set common.container.apptainer")
    return str(candidates[-1])


def build_command(ctx: dict[str, Any], environ: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """(argv starting with the apptainer binary, environment for it)."""
    common = ctx.get("common", {})
    data = ctx.get("data", {})
    box = common.get("container")
    if not isinstance(box, dict) or not box.get("image"):
        raise SystemExit("container payload: common.container.image is required")
    command = data.get("command", common.get("command"))
    if not isinstance(command, list) or not command:
        raise SystemExit("container payload: no command (data.command or common.command, a non-empty list)")

    stages = {name: str(path) for name, path in ctx.get("stages", {}).items()}
    fields = _Fields({k: v for k, v in common.items() if isinstance(v, str)})
    fields.update({k: v for k, v in data.items() if isinstance(v, str)})
    fields.update({f"stage_{re.sub(r'[^A-Za-z0-9]+', '_', n)}": p for n, p in stages.items()})
    fields.update(run_dir=str(ctx.get("run_dir", "")), branch=str(ctx.get("branch", "")))
    scratch = [environ[k] for k in ("HUMSUB_SCRATCH", "HUMSUB_SCRATCH_FAST") if environ.get(k)]
    fields.update(scratch=environ.get("HUMSUB_SCRATCH", ""), scratch_fast=environ.get("HUMSUB_SCRATCH_FAST", ""))

    def fmt(value: str) -> str:
        return value.format_map(fields)

    argv: list[str] = []
    for item in command:
        if item == "{inputs}":
            argv += [str(p) for p in data.get("inputs", [])]
        elif item == "{outputs}":
            argv += [str(p) for p in ctx.get("outputs", [])]
        else:
            argv.append(fmt(str(item)))

    binds: list[str] = []
    for path in [fmt(b) for b in box.get("binds", ["/beegfs"])] + list(stages.values()) + scratch:
        if path not in binds:
            binds.append(path)
    cmd = [find_apptainer(box.get("apptainer")), "exec"]
    if box.get("nv"):
        cmd.append("--nv")
    cmd += ["--bind", ",".join(binds)]
    if box.get("workdir"):
        cmd += ["--pwd", fmt(box["workdir"])]
    for key, value in (box.get("env") or {}).items():
        cmd += ["--env", f"{key}={fmt(str(value))}"]
    cmd += [fmt(box["image"])] + argv

    env = {k: v for k, v in environ.items() if k != "PYTHONPATH"}
    return cmd, env


def main() -> None:
    ctx = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    cmd, env = build_command(ctx, dict(os.environ))
    print("[container] " + " ".join(cmd), flush=True)
    os.execve(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
