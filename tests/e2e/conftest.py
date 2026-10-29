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

# slurmctld checks time limits on a ~30 s tick and sends --signal=B:USR1@N once
# (limit - now) <= N + 30 s.  The signal therefore arrives between (limit - N - 30 s)
# and (limit - N) after the job's start -- *early*, not late.  With a 1 min limit and
# N = 30 it can arrive seconds after the job started, before humsub's handler is
# installed on a slow runner; the default action then kills the hop without a
# continuation marker.  A 2 min limit with N = 60 gives 30-60 s after start.
SLOW_TIME_LIMIT = "00:02:00"
SLOW_SIGNAL_SECONDS = 60

E2E_ENABLED = os.environ.get("HUMSUB_E2E") == "1" and shutil.which("sbatch") is not None


def pytest_collection_modifyitems(config, items):
    if E2E_ENABLED:
        # Slow tests mostly wait for Slurm time limits: start them first so that, with
        # pytest-xdist, they overlap each other and the fast tests fill the gaps.
        items.sort(key=lambda item: item.get_closest_marker("slow") is None)
        return
    skip = pytest.mark.skip(reason="needs HUMSUB_E2E=1 and a Slurm (see tests/e2e/run.sh)")
    for item in items:
        if "e2e" in str(item.fspath):
            item.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes minutes (real Slurm time limits)")


class Project:
    """A scratch project directory with its own config, output and cache dirs.

    Every project submits under its own Slurm job name (the directory tag), so tests can run in
    parallel (pytest-xdist): queue checks and the teardown only see this project's jobs.
    """

    def __init__(self, root: Path, cache: Path, job_name: str) -> None:
        self.job_name = job_name
        self.dir = root / "project"
        self.out = root / "out"
        self.cache = cache
        self.dir.mkdir(parents=True)
        # law polls its jobs every minute by default; luigi reads ./luigi.cfg
        (self.dir / "luigi.cfg").write_text("[ManifestPayloadWorkflow]\npoll_interval = 3s\n\n[PayloadWorkflow]\npoll_interval = 3s\n")
        self.configure()

    def configure(self, execution: dict | None = None, **slurm) -> None:
        values = {
            "job_name": self.job_name,
            "account": os.environ.get("HUMSUB_E2E_ACCOUNT", "testgrp_std"),
            "partition": os.environ.get("HUMSUB_E2E_PARTITION", "std"),
            "gpus": 0,
            "time_limit": "00:05:00",
            "max_hops": 3,
            "signal_seconds": 30,
        }
        values.update(slurm)
        ex = {"image": "none", "output_dir": str(self.out), "cache_dir": str(self.cache)}
        ex.update(execution or {})
        lines = ["[execution]"] + [f"{k} = {json.dumps(v)}" for k, v in ex.items()] + ["", "[slurm]"]
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

    def squeue(self) -> str:
        """This project's queued/running jobs (chains, followers, supervisor), one per line."""
        return squeue(self.job_name)

    def cancel_jobs(self) -> None:
        ids = [line.split()[0] for line in self.squeue().splitlines() if line.strip()]
        if ids:
            subprocess.run(["scancel", *ids], check=False)

    def submit_single(self, run: str, *args: str, **kw) -> "Submission":
        proc = self.humsub("submit", "--run-name", run, "--", *args, **kw)
        return Submission(self, proc)

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
        # manifest mode prints "chain  ID", single-command mode "chain id  ID"
        self.chains = re.findall(r"\[submit\] chain(?: id)?\s+(\S+)", text)

    @property
    def text(self) -> str:
        return self.proc.stdout + self.proc.stderr

    def chain_states(self) -> dict[str, str]:
        if self.id is None:        # single-command mode has no submission line: ask per chain
            states = {}
            for chain in self.chains:
                out = self.project.humsub("status", chain, check=True).stdout
                m = re.search(r"^status:\s+(\S+) \(law: (\w+)\)", out, re.M)
                states[chain] = m.group(2)
            return states
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


def squeue(job_name: str | None = None) -> str:
    """The test user's queue; with *job_name*, only jobs named *job_name* or *job_name*-<suffix>."""
    out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", "testuser"), "-o", "%i %T %j %E"],
                         text=True, capture_output=True).stdout
    if job_name is None:
        return out
    return "".join(line + "\n" for line in out.splitlines()
                   if len(line.split()) > 2 and (line.split()[2] == job_name or line.split()[2].startswith(job_name + "-")))


def sacct_states(job_ids: list[str]) -> dict[str, str]:
    out = subprocess.run(["sacct", "-X", "-n", "-P", "-o", "JobID,State", "-j", ",".join(job_ids)],
                         text=True, capture_output=True).stdout
    return dict(line.split("|", 1) for line in out.splitlines() if "|" in line)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)


def _tail(path: Path, lines: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError as exc:
        return f"<unreadable: {exc}>"


def diagnostics(proj: "Project") -> str:
    """Everything needed to understand a failed e2e test from the CI log alone."""
    parts = ["queue (this test):\n" + proj.squeue(),
             "sacct:\n" + subprocess.run(["sacct", "-X", "-n", f"--name={proj.job_name},{proj.job_name}-supervisor",
                                           "-o", "JobID,JobName,State,ExitCode,Elapsed,Timelimit,Start,End"],
                                          text=True, capture_output=True).stdout]
    for state in sorted((proj.out / ".hummel-submit" / "chains").glob("*/state.json")):
        try:
            d = json.loads(state.read_text())
            keep = {k: d.get(k) for k in ("chain_id", "status", "jobs", "exit_code", "error", "current_hop", "current_job_id", "next_job_id")}
        except (OSError, ValueError):
            keep = "<unreadable>"
        parts.append(f"{state}:\n{keep}")
    for log in sorted((proj.out / "logs").glob("*.log")):
        parts.append(f"--- {log.name} (tail) ---\n{_tail(log)}")
    return "\n".join(parts)


@pytest.fixture
def project(request):
    """Fresh project with outputs on $BEEGFS and staging on $SSD, removed afterwards."""
    beegfs = Path(os.environ["BEEGFS"])
    ssd = Path(os.environ["SSD"])
    tag = f"e2e-{uuid.uuid4().hex[:8]}"
    root = beegfs / tag
    cache = ssd / ".hummel-submit" / tag
    proj = Project(root, cache, job_name=tag)
    yield proj
    rep = getattr(request.node, "rep_call", None)
    if rep is not None and rep.failed:
        sys.stderr.write(f"\n===== e2e diagnostics for {request.node.name} =====\n{diagnostics(proj)}\n=====\n")
    proj.cancel_jobs()       # only this project's jobs: other tests may run in parallel
    if not request.config.getoption("--keep-e2e", default=False):
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(cache, ignore_errors=True)


def pytest_addoption(parser):
    parser.addoption("--keep-e2e", action="store_true", default=False, help="keep e2e work directories")
