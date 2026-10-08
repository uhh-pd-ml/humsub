"""End-to-end tests against a real Slurm (hummel-slurm-ci image).

Skipped unless ``HUMSUB_E2E=1`` and ``sbatch`` is on PATH.  Run them through
``tests/e2e/run.sh``, which starts the container and calls pytest inside it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap
import time
import uuid

import pytest

E2E_ENABLED = os.environ.get("HUMSUB_E2E") == "1" and shutil.which("sbatch") is not None


def pytest_collection_modifyitems(config, items):
    if E2E_ENABLED:
        return
    skip = pytest.mark.skip(reason="needs HUMSUB_E2E=1 and a Slurm (see tests/e2e/run.sh)")
    for item in items:
        if "e2e" in str(item.fspath):
            item.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes minutes (real Slurm time limits)")


class Project:
    """A scratch project directory with its own config, output and cache dirs."""

    def __init__(self, root: Path, cache: Path) -> None:
        self.dir = root / "project"
        self.out = root / "out"
        self.cache = cache
        self.dir.mkdir(parents=True)
        self.configure()

    def configure(self, **slurm) -> None:
        values = {
            "job_name": "e2e",
            "account": os.environ.get("HUMSUB_E2E_ACCOUNT", "testgrp_std"),
            "partition": os.environ.get("HUMSUB_E2E_PARTITION", "std"),
            "gpus": 0,
            "time_limit": "00:05:00",
            "max_hops": 3,
            "signal_seconds": 30,
        }
        values.update(slurm)
        lines = ["[execution]", 'image = "none"', f'output_dir = "{self.out}"', f'cache_dir = "{self.cache}"', "", "[slurm]"]
        lines += [f"{k} = {json.dumps(v)}" for k, v in values.items()]
        (self.dir / ".hummel-submit.toml").write_text("\n".join(lines) + "\n")

    def write_payload(self, body: str, name: str = "payload.py") -> Path:
        path = self.dir / name
        path.write_text("#!" + sys.executable + "\n" + textwrap.dedent(body))
        path.chmod(0o755)
        return path

    def write_manifest(self, outputs_per_branch: dict[int, list[Path]], data: dict[int, object] | None = None) -> Path:
        manifest = {
            "schema": 1,
            "common": {},
            "branches": [
                {"id": i, "data": (data or {}).get(i, {}), "outputs": [str(p) for p in outs]}
                for i, outs in outputs_per_branch.items()
            ],
        }
        path = self.dir / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path

    def humsub(self, *args: str, timeout: int = 180, check: bool = False) -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items() if not k.startswith("HUMMEL_")}
        proc = subprocess.run(
            [str(Path(sys.executable).parent / "humsub"), *args],
            cwd=self.dir, env=env, text=True, capture_output=True, timeout=timeout,
        )
        if check and proc.returncode != 0:
            raise AssertionError(f"humsub {' '.join(args)} failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
        return proc

    def submit(self, manifest: Path, payload: Path, run: str, *extra: str, **kw) -> "Submission":
        proc = self.humsub("submit-manifest", "--manifest", str(manifest), "--payload", str(payload),
                           "--run-name", run, *extra, **kw)
        return Submission(self, proc)


class Submission:
    def __init__(self, project: Project, proc: subprocess.CompletedProcess) -> None:
        self.project, self.proc = project, proc
        text = proc.stdout + proc.stderr
        m = re.search(r"\[submit\] submission\s+(\S+)", text)
        self.id = m.group(1) if m else None
        self.chains = re.findall(r"\[submit\] chain\s+(\S+)", text)

    @property
    def text(self) -> str:
        return self.proc.stdout + self.proc.stderr

    def chain_states(self) -> dict[str, str]:
        out = self.project.humsub("submission-status", self.id, check=True).stdout
        return {m.group(1): m.group(2) for m in re.finditer(r"^\s+(\S+): (\w+)", out, re.M)}

    def wait(self, timeout: int = 150) -> dict[str, str]:
        """Wait until no chain is pending/running; returns chain -> state."""
        deadline = time.time() + timeout
        while True:
            states = self.chain_states()
            if states and not {"pending", "running"} & set(states.values()):
                return states
            if time.time() > deadline:
                raise AssertionError(f"chains still active after {timeout}s: {states}\n{squeue()}")
            time.sleep(2)


def squeue() -> str:
    return subprocess.run(["squeue", "-h", "-o", "%i %T %j %E"], text=True, capture_output=True).stdout


def sacct_states(job_ids: list[str]) -> dict[str, str]:
    out = subprocess.run(["sacct", "-X", "-n", "-P", "-o", "JobID,State", "-j", ",".join(job_ids)],
                         text=True, capture_output=True).stdout
    return dict(line.split("|", 1) for line in out.splitlines() if "|" in line)


@pytest.fixture
def project(request):
    """Fresh project with outputs on $BEEGFS and staging on $SSD, removed afterwards."""
    beegfs = Path(os.environ["BEEGFS"])
    ssd = Path(os.environ["SSD"])
    tag = f"e2e-{uuid.uuid4().hex[:8]}"
    root = beegfs / tag
    cache = ssd / ".hummel-submit" / tag
    proj = Project(root, cache)
    yield proj
    subprocess.run(["scancel", "-u", os.environ.get("USER", "testuser")], check=False)
    if not request.config.getoption("--keep-e2e", default=False):
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(cache, ignore_errors=True)


def pytest_addoption(parser):
    parser.addoption("--keep-e2e", action="store_true", default=False, help="keep e2e work directories")
