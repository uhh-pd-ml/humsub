"""Manifest workflow against a real Slurm: submission, chains, failures, resubmission."""
from __future__ import annotations

import json
import time

import pytest

from .conftest import sacct_states, squeue

SUM_PAYLOAD = """
    import json, os, sys
    from pathlib import Path
    ctx = json.loads(Path(sys.argv[1]).read_text())
    n = ctx["data"]["n"]
    out = Path(ctx["outputs"][0])
    tmp = out.with_name("." + out.name + ".tmp")
    tmp.write_text(json.dumps({"n": n, "sum": n * (n + 1) // 2, "hop": os.environ["HUMSUB_HOP"],
                               "attempt": os.environ["HUMSUB_ATTEMPT"], "stage": os.environ.get("HUMSUB_STAGE_LOOKUP", "")}))
    os.replace(tmp, out)
"""


def _sum_project(project, numbers):
    outs = {i: [project.dir / "results" / f"sum-{i}.json"] for i in range(len(numbers))}
    (project.dir / "results").mkdir()
    manifest = project.write_manifest(outs, {i: {"n": n} for i, n in enumerate(numbers)})
    return manifest, project.write_payload(SUM_PAYLOAD), outs


def test_roundtrip_resubmit_and_cleanup(project):
    (project.dir / "lookup").mkdir()
    (project.dir / "lookup" / "factor.txt").write_text("3")
    manifest, payload, outs = _sum_project(project, [10, 100, 1000])

    sub = project.submit(manifest, payload, "first", "--stage", f"lookup={project.dir / 'lookup'}")
    assert sub.proc.returncode == 0, sub.text
    assert len(sub.chains) == 3
    states = sub.wait()
    assert set(states.values()) == {"finished"}, states

    for i, n in enumerate([10, 100, 1000]):
        data = json.loads(outs[i][0].read_text())
        assert data["sum"] == n * (n + 1) // 2
        assert data["hop"] == "0" and data["attempt"] == "1"
        assert data["stage"].startswith(str(project.cache)), "stage must come from the frozen cache copy"

    # a completed chain must not leave its pre-queued follower behind
    time.sleep(3)
    assert squeue().strip() == "", squeue()

    # resubmitting a complete manifest is a successful no-op
    again = project.submit(manifest, payload, "second")
    assert again.proc.returncode == 0
    assert "nothing to submit" in again.text and not again.chains

    # cleanup removes the staged inputs of the finished submission
    assert (project.cache / "stages" / sub.id / "lookup").exists()
    project.humsub("cleanup", sub.id, check=True)
    assert not (project.cache / "stages" / sub.id).exists()


def test_resubmit_runs_only_missing_branches(project):
    manifest, payload, outs = _sum_project(project, [5, 6, 7])
    first = project.submit(manifest, payload, "a")
    assert set(first.wait().values()) == {"finished"}

    outs[1][0].unlink()
    second = project.submit(manifest, payload, "b")
    assert second.proc.returncode == 0, second.text
    assert len(second.chains) == 1, second.text
    assert set(second.wait().values()) == {"finished"}
    assert json.loads(outs[1][0].read_text())["sum"] == 21


def test_failing_payload_fails_chain_and_leaves_no_partial_output(project):
    out = project.dir / "results" / "o.json"
    (project.dir / "results").mkdir()
    manifest = project.write_manifest({0: [out]})
    payload = project.write_payload(f"""
        import sys
        open({str(out)!r}, "w").write("partial")      # a partial output must be removed
        print("about to fail", flush=True)
        sys.exit(3)
    """)
    sub = project.submit(manifest, payload, "fail")
    states = sub.wait()
    assert list(states.values()) == ["failed"], states
    assert not out.exists(), "partial output of a failed branch must be deleted"
    status = project.humsub("status", sub.chains[0], check=True).stdout
    assert "failed" in status
    # the pre-queued follower of a failed chain is not left pending
    time.sleep(3)
    assert squeue().strip() == "", squeue()


def test_exit_zero_without_output_is_a_failure(project):
    out = project.dir / "results" / "o.json"
    (project.dir / "results").mkdir()
    manifest = project.write_manifest({0: [out]})
    payload = project.write_payload("print('I did nothing')\n")
    sub = project.submit(manifest, payload, "noout")
    assert list(sub.wait().values()) == ["failed"]
    logs = "".join(p.read_text() for p in (project.out / "logs").glob("*.log"))
    assert "did not materialize" in logs, logs[-1500:]


def test_cancel_stops_chain_and_its_jobs(project):
    out = project.dir / "results" / "o.json"
    (project.dir / "results").mkdir()
    manifest = project.write_manifest({0: [out]})
    payload = project.write_payload("import time\ntime.sleep(120)\n")
    sub = project.submit(manifest, payload, "cancel")
    chain = sub.chains[0]
    for _ in range(30):                       # wait until it is actually running
        if "running" in sub.chain_states().values():
            break
        time.sleep(1)
    r = project.humsub("cancel", chain)
    assert r.returncode == 0, r.stdout + r.stderr
    time.sleep(4)
    assert squeue().strip() == "", squeue()
    assert list(sub.chain_states().values()) == ["failed"]
    assert not out.exists()


def test_dry_run_submits_nothing(project):
    manifest, payload, _ = _sum_project(project, [1, 2])
    r = project.humsub("submit-manifest", "--dry-run", "--manifest", str(manifest), "--payload", str(payload),
                       "--run-name", "dry")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "branches    2" in r.stdout
    assert squeue().strip() == ""
    assert not (project.out / ".hummel-submit" / "chains").exists()


def test_site_rules_are_enforced_by_the_cluster(project):
    manifest, payload, _ = _sum_project(project, [1])
    # --mem is forbidden on Hummel; the emulated submit filter (or humsub) must refuse it
    r = project.submit(manifest, payload, "mem", "--sbatch-arg=--mem=1G")
    assert r.proc.returncode != 0
    assert "mem" in r.text.lower()
    assert squeue().strip() == ""
    # a partition the account has no association for is rejected by accounting
    r = project.submit(manifest, payload, "wrongpart", "--partition", "gpu")
    assert r.proc.returncode != 0
    assert "Invalid account or account/partition" in r.text or "invalid" in r.text.lower(), r.text[-800:]


@pytest.mark.slow
def test_continuation_across_hops(project):
    """Hop 0 is stopped by the pre-timeout signal; hop 1 re-runs the branch and completes."""
    out = project.dir / "results" / "o.json"
    (project.dir / "results").mkdir()
    manifest = project.write_manifest({0: [out]})
    payload = project.write_payload(f"""
        import os, time
        from pathlib import Path
        hop = int(os.environ["HUMSUB_HOP"])
        print("payload running in hop", hop, flush=True)
        if hop == 0:
            time.sleep(600)                      # never finishes; the chain stops it before the limit
        Path({str(out)!r}).write_text("done in hop %d" % hop)
    """)
    project.configure(time_limit="00:01:00", signal_seconds=30, max_hops=3)
    sub = project.submit(manifest, payload, "hops")
    states = sub.wait(timeout=300)
    assert list(states.values()) == ["finished"], states
    assert out.read_text() == "done in hop 1"
    status = project.humsub("status", sub.chains[0], check=True).stdout
    jobs_line = next(l for l in status.splitlines() if l.startswith("jobs:"))
    job_ids = [j.strip() for j in jobs_line.split(":", 1)[1].split(",")]
    assert len(job_ids) >= 2, status
    # hop 0 stops itself on the pre-timeout signal (so it ends cleanly, not by TIMEOUT)
    # and the follower finishes the chain
    assert set(sacct_states(job_ids[:2]).values()) == {"COMPLETED"}, sacct_states(job_ids[:2])
    logs = "".join(p.read_text() for p in sorted((project.out / "logs").glob("*.log")))
    assert "payload running in hop 0" in logs and "payload running in hop 1" in logs
    assert "time limit approaching: marked chain for continuation" in logs
    assert "pre-timeout signal received: True" in logs
    assert "law payload completed normally; stopping chain" in logs
