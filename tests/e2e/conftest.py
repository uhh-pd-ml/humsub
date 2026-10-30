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

# How tests that need a hop to run out of time get there (tests/e2e/run.sh sets it from E2E_PROFILE):
#   "signal" (fast profile)      the test sends the pre-timeout signal itself, exactly what slurmctld sends at the
#                                deadline (scancel --batch --signal=USR1 to the hop's batch shell, which is the worker);
#                                the time limits are long, so nothing else interferes.  Seconds instead of minutes.
#   "slurm"  (realistic profile) real time limits: slurmctld's own clock delivers --signal=B:USR1@N.  Minutes.
DEADLINES = os.environ.get("HUMSUB_E2E_DEADLINES", "signal")
#: --supervisor-interval of the tests (run.sh lowers humsub's 60 s floor to match in the fast profile)
SUPERVISOR_INTERVAL = os.environ.get("HUMSUB_E2E_SUPERVISOR_INTERVAL", "10")


def pytest_collection_modifyitems(config, items):
    if not E2E_ENABLED:
        skip = pytest.mark.skip(reason="needs HUMSUB_E2E=1 and a Slurm (see tests/e2e/run.sh)")
        for item in items:
            if "e2e" in str(item.fspath):
                item.add_marker(skip)
        return
    for item in items:
        if DEADLINES == "slurm" and item.get_closest_marker("deadline"):
            item.add_marker(pytest.mark.slow)
        if int(SUPERVISOR_INTERVAL) >= 60 and item.get_closest_marker("supervisor"):
            item.add_marker(pytest.mark.slow)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes minutes in the current profile")
    config.addinivalue_line("markers", "deadline: a hop runs out of time (see DEADLINES): slow with real time limits")
    config.addinivalue_line("markers", "supervisor: waits for supervisor passes: slow with Hummel's 60 s interval")


class Project:
    """A scratch project directory with its own config, output and cache dirs."""

    def __init__(self, root: Path, cache: Path) -> None:
        self.dir = root / "project"
        self.out = root / "out"
        self.bulk = root / "bulk-scratch"       # per test: tests run in parallel and gc sweeps scratch roots
        self.cache = cache
        self.dir.mkdir(parents=True)
        # law polls its jobs every minute by default; luigi reads ./luigi.cfg
        (self.dir / "luigi.cfg").write_text("[ManifestPayloadWorkflow]\npoll_interval = 3s\n\n[ManifestFinalizeWorkflow]\npoll_interval = 3s\n\n[PayloadWorkflow]\npoll_interval = 3s\n")
        self.configure()

    def configure(self, execution: dict | None = None, **slurm) -> None:
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
        ex = {"image": "none", "output_dir": str(self.out), "cache_dir": str(self.cache),
              "bulk_scratch_dir": str(self.bulk)}
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

    def job_ids(self) -> set[str]:
        """Every Slurm job this test's submissions created: chain hops (incl. followers) and supervisor jobs."""
        ids: set[str] = set()
        hs = self.out / ".hummel-submit"
        for path in list(hs.glob("chains/*/state.json")) + list(hs.glob("submissions/*/submission.json")):
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            ids |= {str(j) for j in data.get("jobs", []) + data.get("supervisor_job_ids", [])}
        return ids

    def queue(self) -> str:
        """squeue lines of this test's jobs only (tests run in parallel on one cluster)."""
        mine = self.job_ids()
        return "\n".join(line for line in squeue().splitlines() if line.split() and line.split()[0] in mine)

    def logs(self) -> str:
        return "".join(p.read_text(errors="replace") for p in sorted((self.out / "logs").glob("*.log")))

    def configure_deadline(self, *, time_limit: str, signal_seconds: int, **slurm) -> None:
        """Slurm settings of a test whose hops run out of time: *time_limit* is the real limit (DEADLINES="slurm");
        with DEADLINES="signal" the limit is long and the test calls reach_deadline() instead.  signal_seconds (and
        with it the grace period) are the same in both modes, so humsub behaves the same."""
        self.configure(time_limit=time_limit if DEADLINES == "slurm" else "00:15:00", signal_seconds=signal_seconds,
                       **slurm)

    def reach_deadline(self, chain: str, hop: int, *, after: str, timeout: int = 240) -> None:
        """Let hop *hop* of *chain* reach its pre-timeout signal.  With DEADLINES="slurm" Slurm sends it at the time
        limit, so this returns at once.  With "signal" it waits until the hop runs and *after* (a line the payload
        prints) appears in the logs, then sends what slurmctld would: SIGUSR1 to the hop's batch shell.  *after* must
        come from the payload: the worker installs its handler only right before starting it, and an earlier SIGUSR1
        would kill the worker (its default action)."""
        if DEADLINES != "signal":
            return
        state_file = self.out / ".hummel-submit" / "chains" / chain / "state.json"
        deadline = time.time() + timeout
        while True:
            try:
                state = json.loads(state_file.read_text())
            except (OSError, ValueError):
                state = {}
            job = state.get("current_job_id")
            if state.get("current_hop") == hop and job and after in self.logs():
                break
            if time.time() > deadline:
                raise AssertionError(f"hop {hop} of {chain} did not get to {after!r} in {timeout}s: {state}\n"
                                     f"{self.queue()}\n{self.logs()[-3000:]}")
            time.sleep(1)
        subprocess.run(["scancel", "--batch", "--signal=USR1", str(job)], check=True, capture_output=True)

    def wait_for_log(self, needle: str, timeout: int) -> str:
        deadline = time.time() + timeout
        while True:
            logs = self.logs()
            if needle in logs:
                return logs
            if time.time() > deadline:
                raise AssertionError(f"{needle!r} not in the logs after {timeout}s\n{self.queue()}\n{logs[-3000:]}")
            time.sleep(2)

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
                raise AssertionError(f"chains still active after {timeout}s: {states}\n{self.project.queue()}")
            time.sleep(2)


def squeue() -> str:
    return subprocess.run(["squeue", "-h", "-o", "%i %T %j %E"], text=True, capture_output=True).stdout


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
    parts = ["queue (this test's jobs):\n" + proj.queue(),
             "sacct:\n" + subprocess.run(["sacct", "-X", "-n", "-o", "JobID,JobName,State,ExitCode,Elapsed,Timelimit,Start,End"],
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
    proj = Project(root, cache)
    yield proj
    rep = getattr(request.node, "rep_call", None)
    if rep is not None and rep.failed:
        sys.stderr.write(f"\n===== e2e diagnostics for {request.node.name} =====\n{diagnostics(proj)}\n=====\n")
    # only this test's jobs: the others run in parallel on the same cluster
    mine = sorted(proj.job_ids())
    if mine:
        subprocess.run(["scancel", *mine], check=False, capture_output=True)
    if not request.config.getoption("--keep-e2e", default=False):
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(cache, ignore_errors=True)


def pytest_addoption(parser):
    parser.addoption("--keep-e2e", action="store_true", default=False, help="keep e2e work directories")
