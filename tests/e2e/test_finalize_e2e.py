"""Fan-out / fan-in against a real Slurm: branches, then the manifest's finalize workflow, submitted by law in the
supervisor's passes once every branch is done (and retried / resumed like any other law job)."""
from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

from .conftest import SUPERVISOR_INTERVAL


PAYLOAD = """
    import json, sys, time
    from pathlib import Path
    ctx = json.loads(Path(sys.argv[1]).read_text())
    common, data = ctx["common"], ctx["data"]
    out = Path(ctx["outputs"][0])
    if data.get("finalize"):
        if common["broken"] and Path(common["broken"]).exists():
            sys.exit(7)                                   # a finalize step that fails until the cause is fixed
        parts = [Path(p) for p in common["parts"]]
        missing = [str(p) for p in parts if not p.exists()]
        if missing:                                       # must never happen: law starts it after all branches
            Path(common["early"]).write_text(" ".join(missing))
            sys.exit(9)
        out.write_text(str(sum(int(p.read_text()) for p in parts)))
    else:
        flag = out.with_suffix(".failed-once")
        if ctx["branch"] in common["fail_once"] and not flag.exists():
            flag.write_text("x")
            sys.exit(4)                                   # fails on its first attempt only
        time.sleep(3)
        out.write_text(str(10 ** ctx["branch"]))
"""


def _logs(project) -> str:
    return "".join(p.read_text(errors="replace") for p in sorted((project.out / "logs").glob("*.log")))


def _setup(project, n: int, *, fail_once: tuple[int, ...] = (), broken: Path | None = None):
    res = project.dir / "results"
    res.mkdir()
    parts = [res / f"part{i}.txt" for i in range(n)]
    manifest = {
        "schema": 1,
        "common": {"parts": [str(p) for p in parts], "early": str(res / "EARLY"), "fail_once": list(fail_once),
                   "broken": str(broken) if broken else ""},
        "branches": [{"id": i, "outputs": [str(p)]} for i, p in enumerate(parts)],
        "finalize": {"data": {}, "outputs": [str(res / "sum.txt")]},
    }
    path = project.dir / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, project.write_payload(PAYLOAD), res


def _wait_for(path: Path, timeout: int, project, what: str) -> None:
    deadline = time.time() + timeout
    while not path.exists() and time.time() < deadline:
        time.sleep(5)
    assert path.exists(), f"{what}\n{project.queue()}\n{_logs(project)[-3000:]}"


def _wait_for_log(project, needle: str, timeout: int) -> None:
    deadline = time.time() + timeout
    while needle not in _logs(project) and time.time() < deadline:
        time.sleep(5)
    assert needle in _logs(project), f"{needle!r} not in the logs\n{project.queue()}\n{_logs(project)[-3000:]}"


@pytest.mark.supervisor
def test_finalize_runs_once_after_all_branches(project):
    manifest, payload, res = _setup(project, 4)
    sub = project.submit(manifest, payload, "fanin", "--supervisor-interval", SUPERVISOR_INTERVAL)
    assert sub.proc.returncode == 0, sub.text
    assert "supervisor job (law passes) is switched on" in sub.text and "[submit] finalize" in sub.text
    _wait_for(res / "sum.txt", 300, project, "the supervisor's pass did not submit the finalize workflow")
    assert (res / "sum.txt").read_text() == "1111"
    assert not (res / "EARLY").exists(), (res / "EARLY").read_text()
    _wait_for_log(project, "pass complete -> done", 150)
    status = project.humsub("submission-status", sub.id, check=True).stdout
    assert "4 done" in status and "finalize:   done" in status, status


@pytest.mark.supervisor
def test_finalize_waits_for_a_retried_branch(project):
    manifest, payload, res = _setup(project, 3, fail_once=(2,))
    sub = project.submit(manifest, payload, "fanin-retry", "--supervisor-interval", SUPERVISOR_INTERVAL, "--retries", "1")
    assert sub.proc.returncode == 0, sub.text
    _wait_for(res / "sum.txt", 420, project, "finalize did not run after the retried branch")
    assert (res / "part2.failed-once").exists(), "branch 2 was expected to fail once"
    assert (res / "sum.txt").read_text() == "111"
    assert not (res / "EARLY").exists(), (res / "EARLY").read_text()
    _wait_for_log(project, "pass complete -> done", 150)


@pytest.mark.supervisor
def test_failed_finalize_gives_up_and_resume_retries_it(project):
    broken = project.dir / "BROKEN"
    broken.write_text("x")
    manifest, payload, res = _setup(project, 2, broken=broken)
    sub = project.submit(manifest, payload, "fanin-fix", "--supervisor-interval", SUPERVISOR_INTERVAL, "--retries", "0")
    assert sub.proc.returncode == 0, sub.text
    _wait_for_log(project, "giving up", 300)
    status = project.humsub("submission-status", sub.id, check=True).stdout
    assert "2 done" in status and "finalize:   chain" in status and "failed" in status, status
    assert not (res / "sum.txt").exists()

    broken.unlink()                                       # "fix the cause"
    resumed = project.humsub("resume", sub.id, check=True)
    assert "1 failed law job(s) get a fresh budget of 1 retry" in resumed.stdout, resumed.stdout
    assert "supervisor queued again" in resumed.stdout, resumed.stdout
    _wait_for(res / "sum.txt", 150, project, "resume did not resubmit the finalize workflow")
    assert (res / "sum.txt").read_text() == "11"


@pytest.mark.supervisor
def test_chain_ends_bring_the_next_pass_forward(project):
    """With a half-hour interval nothing would happen for 30 minutes: the ends of the branches bring the pass that
    submits finalize forward, the end of the finalize chain the pass that ends supervision."""
    manifest, payload, res = _setup(project, 2)
    sub = project.submit(manifest, payload, "trigger", "--supervisor-interval", "1800",
                         "--supervisor-trigger-delay", SUPERVISOR_INTERVAL)
    assert sub.proc.returncode == 0, sub.text
    _wait_for(res / "sum.txt", 300, project, "no pass was triggered by the ends of the branches")
    assert (res / "sum.txt").read_text() == "11"
    logs = project.wait_for_log("pass complete -> done", 300)
    assert "chain ended; pass requested: supervisor job" in logs and "brought forward" in logs, logs[-3000:]
