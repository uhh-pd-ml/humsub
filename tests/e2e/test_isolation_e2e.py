"""Two projects of one user, side by side in the same output_dir and cache_dir, each with its own supervisor: they must
not interfere, and gc must never touch the scratch of jobs that are still running."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import pytest

from .conftest import SUPERVISOR_INTERVAL


PAYLOAD = """
    import json, os, sys, time
    from pathlib import Path
    ctx = json.loads(Path(sys.argv[1]).read_text())
    marker = Path(os.environ["HUMSUB_SCRATCH"]) / "still-here"
    marker.write_text("x")
    flag = Path(ctx["common"]["gc_done"])            # the test runs gc --force meanwhile, then creates this
    deadline = time.time() + 180
    while not flag.exists() and time.time() < deadline:
        time.sleep(1)
    if not marker.exists():
        sys.exit(9)                                  # our scratch was deleted under our feet
    Path(ctx["outputs"][0]).write_text(os.environ["HUMSUB_SUBMISSION_ID"])
"""


def _logs(out: Path) -> dict[str, str]:
    return {p.name: p.read_text(errors="replace") for p in sorted((out / "logs").glob("*.log"))}


@pytest.mark.supervisor
def test_two_projects_with_supervisors_and_a_forced_gc_do_not_interfere(project):
    # the second project shares output_dir and cache_dir (the default for all projects of a user)
    other = project.dir.parent / "project2"
    other.mkdir()
    for name in (".hummel-submit.toml", "luigi.cfg"):
        shutil.copy(project.dir / name, other / name)

    def humsub(cwd: Path, *args: str) -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items() if not k.startswith("HUMMEL_")}
        return subprocess.run([str(Path(sys.executable).parent / "humsub"), *args], cwd=cwd, env=env,
                              text=True, capture_output=True, timeout=240)

    gc_done = project.dir.parent / "gc-done"
    subs = {}
    outs = {}
    for name, cwd in (("alpha", project.dir), ("beta", other)):
        res = cwd / "results"
        res.mkdir()
        outs[name] = [res / f"o{i}.txt" for i in range(2)]
        manifest = cwd / "manifest.json"
        manifest.write_text(json.dumps({"schema": 1, "common": {"gc_done": str(gc_done)},
                                        "branches": [{"id": i, "outputs": [str(o)]}
                                                                  for i, o in enumerate(outs[name])]}))
        payload = cwd / "payload.py"
        payload.write_text("#!" + sys.executable + "\n" + "\n".join(l[4:] for l in PAYLOAD.splitlines()))
        payload.chmod(0o755)
        proc = humsub(cwd, "submit-manifest", "--manifest", str(manifest), "--payload", str(payload),
                      "--run-name", name, "--supervisor-job", "--supervisor-interval", SUPERVISOR_INTERVAL)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        subs[name] = re.search(r"\[submit\] submission\s+(\S+)", proc.stdout).group(1)

    # while all four branches run: a forced gc with stale-scratch cleanup, from one of the projects
    bulk = project.bulk                              # both projects share it (same config)
    deadline = time.time() + 120
    while len([p for p in bulk.glob("*/*") if p.is_dir()]) < 4 and time.time() < deadline:
        time.sleep(2)
    gc = humsub(project.dir, "gc", "--stale-scratch", "--force")
    assert gc.returncode == 0, gc.stdout + gc.stderr
    assert not re.search(r"^gc:\s+scratch\s", gc.stdout, re.M), f"gc listed scratch of running jobs:\n{gc.stdout}"
    gc_done.write_text("x")                          # the payloads now check that their scratch survived

    deadline = time.time() + 300
    while not all(o.exists() for os_ in outs.values() for o in os_) and time.time() < deadline:
        time.sleep(5)
    for name, files in outs.items():
        for f in files:
            assert f.exists(), f"{name}: {f} missing (scratch deleted while running?)\n{project.squeue()}"
            assert f.read_text() == subs[name]

    # each supervisor ends on its own, and only ever dealt with its own submission
    deadline = time.time() + 240
    while time.time() < deadline:
        sup = {n: t for n, t in _logs(project.out).items() if "supervisor" in n}
        if sum("pass complete -> done" in t for t in sup.values()) >= 2:
            break
        time.sleep(5)
    sup = {n: t for n, t in _logs(project.out).items() if "supervisor" in n}
    assert sum("pass complete -> done" in t for t in sup.values()) == 2, sup
    for name, text in sup.items():
        mentioned = {s for s in subs.values() if f"submission {s} " in text}
        assert len(mentioned) == 1, f"{name} mentions {mentioned}"
    for sub_id in subs.values():
        lock = project.out / ".hummel-submit" / "submissions" / sub_id / "control.lock"
        assert not lock.exists(), f"{lock} left behind"
