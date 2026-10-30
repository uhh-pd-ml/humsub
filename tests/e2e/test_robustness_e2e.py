"""Robust-production features against a real Slurm: in-job retries, retry hops, lanes, resume, scratch cleanup,
extra outputs, soft stop, supervisor passes."""
from __future__ import annotations

import json
import re
import time

import pytest

from .conftest import DEADLINES, SUPERVISOR_INTERVAL, sacct_states


def _job_ids(project, chain: str) -> list[str]:
    status = project.humsub("status", chain, check=True).stdout
    line = next(l for l in status.splitlines() if l.startswith("jobs:"))
    return [j.strip() for j in line.split(":", 1)[1].split(",") if j.strip()]


def _ran_jobs(project, chain: str) -> list[str]:
    """Slurm jobs of a chain that actually ran (the queued follower of a finished chain is cancelled, not run)."""
    jobs = _job_ids(project, chain)
    return [j for j, state in sacct_states(jobs).items() if not state.startswith("CANCELLED")]


def _logs(project) -> str:
    return "".join(p.read_text(errors="replace") for p in sorted((project.out / "logs").glob("*.log")))


def _results(project) -> "Path":
    d = project.dir / "results"
    d.mkdir(exist_ok=True)
    return d


def test_retry_payload_reruns_inside_the_job_with_a_fresh_scratch(project):
    out = _results(project) / "o.json"
    manifest = project.write_manifest({0: [out]})
    payload = project.write_payload(f"""
        import os, sys
        from pathlib import Path
        scratch = Path(os.environ["HUMSUB_SCRATCH"])
        attempt = int(os.environ["HUMSUB_PAYLOAD_ATTEMPT"])
        dirty = (scratch / "left-by-previous-attempt").exists()
        (scratch / "left-by-previous-attempt").write_text("x")
        if attempt < 3:
            sys.exit(7)
        Path({str(out)!r}).write_text(f"attempt={{attempt}} dirty={{dirty}}")
    """)
    sub = project.submit(manifest, payload, "retry", "--retry-payload", "2")
    assert sub.proc.returncode == 0, sub.text
    assert list(sub.wait().values()) == ["finished"]
    assert out.read_text() == "attempt=3 dirty=False"
    assert len(_ran_jobs(project, sub.chains[0])) == 1, "retries happen inside one Slurm job"
    assert "attempt 1 failed" in _logs(project) and "attempt 2 failed" in _logs(project)


def test_retry_budget_is_respected(project):
    out = _results(project) / "o.json"
    manifest = project.write_manifest({0: [out]})
    payload = project.write_payload("import sys\nsys.exit(7)\n")
    sub = project.submit(manifest, payload, "budget", "--retry-payload", "1")
    assert list(sub.wait().values()) == ["failed"]
    assert "retry budget of 1 exhausted" in _logs(project)


def test_safety_margin_retries_a_failed_hop_through_the_follower(project):
    out = _results(project) / "o.json"
    manifest = project.write_manifest({0: [out]})
    payload = project.write_payload(f"""
        import os, sys
        from pathlib import Path
        if os.environ["HUMSUB_HOP"] == "0":
            sys.exit(8)
        Path({str(out)!r}).write_text("hop " + os.environ["HUMSUB_HOP"])
    """)
    project.configure(max_hops=1)  # only the failure budget provides a follower
    sub = project.submit(manifest, payload, "margin", "--safety-margin", "0.5")
    assert sub.proc.returncode == 0, sub.text
    assert list(sub.wait().values()) == ["finished"]
    assert out.read_text() == "hop 1"
    jobs = _job_ids(project, sub.chains[0])
    assert len(jobs) == 2
    states = sacct_states(jobs)
    assert states[jobs[0]] == "FAILED" and states[jobs[1]] == "COMPLETED", states  # history stays readable
    assert "failure budget 1/1 used" in _logs(project)


def test_one_broken_chain_cannot_consume_other_chains_retry_hops(project):
    good, bad = _results(project) / "good.json", _results(project) / "bad.json"
    manifest = project.write_manifest({0: [good], 1: [bad]}, {0: {"ok": True}, 1: {"ok": False}})
    payload = project.write_payload(f"""
        import json, sys
        from pathlib import Path
        ctx = json.loads(Path(sys.argv[1]).read_text())
        if not ctx["data"]["ok"]:
            sys.exit(9)
        Path(ctx["outputs"][0]).write_text("fine")
    """)
    project.configure(max_hops=1)
    sub = project.submit(manifest, payload, "blackhole", "--safety-margin", "0.5")
    states = sub.wait(timeout=200)
    assert sorted(states.values()) == ["failed", "finished"], states
    assert good.read_text() == "fine" and not bad.exists()
    failed_chain = next(c for c, s in states.items() if s == "failed")
    assert len(_job_ids(project, failed_chain)) == 2, "the broken chain used exactly its own budget (1 extra hop)"


def test_max_concurrent_caps_running_chains(project):
    res = _results(project)
    outs = {i: [res / f"o{i}.json"] for i in range(6)}
    manifest = project.write_manifest(outs)
    payload = project.write_payload(f"""
        import json, sys, time
        from pathlib import Path
        ctx = json.loads(Path(sys.argv[1]).read_text())
        start = time.time()
        time.sleep(8)
        Path(ctx["outputs"][0]).write_text(json.dumps([start, time.time()]))
    """)
    sub = project.submit(manifest, payload, "lanes", "--max-concurrent", "2")
    assert sub.proc.returncode == 0, sub.text
    assert set(sub.wait(timeout=240).values()) == {"finished"}
    spans = [json.loads(o[0].read_text()) for o in outs.values()]
    peak = max(sum(1 for s, e in spans if s <= t < e) for t, _ in spans)
    assert peak <= 2, f"{peak} chains ran at the same time: {spans}"


def test_resume_grants_failed_jobs_new_retries_in_the_same_submission(project):
    res = _results(project)
    outs = {i: [res / f"o{i}.json"] for i in range(3)}
    manifest = project.write_manifest(outs)
    payload = project.write_payload("""
        import json, os, sys
        from pathlib import Path
        ctx = json.loads(Path(sys.argv[1]).read_text())
        out = Path(ctx["outputs"][0])
        flag = out.with_suffix(".failed-once")
        if ctx["branch"] == 1 and not flag.exists():
            flag.write_text("x")
            sys.exit(5)                     # fails on its first attempt only
        out.write_text(os.environ["SLURM_JOB_ID"])
    """)
    first = project.submit(manifest, payload, "first")
    states = first.wait()
    assert sorted(states.values()) == ["failed", "finished", "finished"], states
    status = project.humsub("submission-status", first.id, check=True).stdout
    assert "2 done" in status and "1 failed" in status and "humsub resume" in status
    done_before = outs[0][0].read_text()

    dry = project.humsub("resume", first.id, "--dry-run", check=True)
    assert "1 failed law job(s) get a fresh budget of 1 retry" in dry.stdout, dry.stdout
    assert not outs[1][0].exists(), "--dry-run submits nothing"

    again = project.humsub("resume", first.id, check=True)
    assert "[resume] pass: progress" in again.stdout, again.stdout
    assert "[submit] submission" not in again.stdout, "resume stays in the same submission"
    deadline = time.time() + 150
    while not outs[1][0].exists() and time.time() < deadline:
        time.sleep(3)
    assert outs[1][0].exists(), f"law did not resubmit the failed job\n{project.squeue()}\n{_logs(project)[-2000:]}"
    assert outs[0][0].read_text() == done_before, "branches that were done are not redone"
    time.sleep(5)
    summary = project.humsub("submission-status", first.id, check=True).stdout
    assert "3 done" in summary and "0 failed" in summary
    nothing = project.humsub("resume", first.id, check=True)
    assert "0 failed law job(s)" in nothing.stdout and "[resume] pass: complete" in nothing.stdout


def test_failed_branch_leaves_no_big_scratch_but_keeps_small_diagnostics(project):
    out = _results(project) / "o.json"
    bulk = project.out / "bulk-scratch"
    project.configure(execution={"bulk_scratch_dir": str(bulk)})
    manifest = project.write_manifest({0: [out]})
    payload = project.write_payload("""
        import os, sys
        from pathlib import Path
        Path(os.environ["HUMSUB_SCRATCH"], "big.bin").write_bytes(b"0" * 5_000_000)
        Path(os.environ["HUMSUB_SCRATCH"], "params.json").write_text("{}")
        Path(os.environ["HUMSUB_SCRATCH_FAST"], "note.txt").write_text("hi")
        assert os.environ["HUMSUB_SCRATCH"] != os.environ["HUMSUB_SCRATCH_FAST"]
        sys.exit(3)
    """)
    sub = project.submit(manifest, payload, "scratch")
    assert list(sub.wait().values()) == ["failed"]
    leftovers = [p for p in (bulk).rglob("*") if p.is_file()] + [p for p in (project.cache / "payload-work").rglob("*") if p.is_file()]
    assert leftovers == [], f"scratch must be removed, found {leftovers}"
    kept = sorted(p.name for p in (project.out / "runs" / "scratch" / "leftovers").rglob("*") if p.is_file())
    assert kept == ["note.txt", "params.json"], kept            # the 5 MB file is gone
    assert str(bulk) in _logs(project), "bulk scratch must live under bulk_scratch_dir"


def test_extra_outputs_are_never_deleted_and_never_count_for_completion(project):
    out = _results(project) / "o.json"
    extra = _results(project) / "o.ckpt"
    extra.write_text("checkpoint from an earlier study")
    manifest_path = project.dir / "manifest.json"
    manifest_path.write_text(json.dumps({"schema": 1, "branches": [
        {"id": 0, "outputs": [str(out)], "extra_outputs": [str(extra)]}]}))
    payload = project.write_payload(f"""
        import json, os, sys
        from pathlib import Path
        ctx = json.loads(Path(sys.argv[1]).read_text())
        ck = Path(ctx["extra_outputs"][0])
        seen = ck.read_text()                              # the extra output must still be there at every attempt
        ck.write_text("updated by attempt " + os.environ["HUMSUB_PAYLOAD_ATTEMPT"])
        if os.environ["HUMSUB_PAYLOAD_ATTEMPT"] == "1":
            sys.exit(7)                                    # a failed attempt must not wipe it either
        Path(ctx["outputs"][0]).write_text(seen)
    """)
    sub = project.submit(manifest_path, payload, "extra", "--retry-payload", "1")
    # the extra output existed before submission, yet the branch is not complete: it must have been run
    assert sub.proc.returncode == 0 and sub.chains, sub.text
    assert list(sub.wait().values()) == ["finished"]
    assert out.read_text() == "updated by attempt 1"
    assert extra.read_text() == "updated by attempt 2", "kept after success"


def test_extra_output_paths_are_validated(project):
    manifest = project.dir / "manifest.json"
    manifest.write_text(json.dumps({"schema": 1, "branches": [
        {"id": 0, "outputs": ["/tmp/x.json"], "extra_outputs": ["relative/ckpt"]}]}))
    payload = project.write_payload("pass\n")
    r = project.submit(manifest, payload, "badextra")
    assert r.proc.returncode != 0 and "extra output must be absolute" in r.text


@pytest.mark.deadline
def test_soft_stop_lets_a_nearly_done_payload_finish_and_holds_back_the_next_branch(project):
    res = _results(project)
    outs = {0: [res / "a.json"], 1: [res / "b.json"]}
    manifest = project.write_manifest(outs, {0: {"until_notice": True}, 1: {"work": 5}})
    payload = project.write_payload("""
        import json, os, sys, time
        from pathlib import Path
        ctx = json.loads(Path(sys.argv[1]).read_text())
        print(f"branch {ctx['branch']} working in hop {os.environ['HUMSUB_HOP']}", flush=True)
        if ctx["data"].get("until_notice"):         # straddles the notice, then is nearly done
            end = time.time() + 280
            while not os.path.exists(os.environ["HUMSUB_SOFT_STOP_FILE"]) and time.time() < end:
                time.sleep(1)
            time.sleep(2)
        else:
            time.sleep(ctx["data"]["work"])
        flag = os.path.exists(os.environ["HUMSUB_SOFT_STOP_FILE"])
        Path(ctx["outputs"][0]).write_text(json.dumps({"hop": os.environ["HUMSUB_HOP"], "soft": flag}))
    """)
    # real limits: the notice comes 90-120 s into hop 0 (0-30 s early delivery); the hard stop is 60 s after it
    project.configure_deadline(time_limit="00:04:00", signal_seconds=150, grace_seconds=60, max_hops=3)
    sub = project.submit(manifest, payload, "soft", "--tasks-per-job", "2")
    assert sub.proc.returncode == 0, sub.text
    project.reach_deadline(sub.chains[0], 0, after="branch 0 working in hop 0")
    states = sub.wait(timeout=400)
    assert list(states.values()) == ["finished"], states
    a, b = (json.loads(outs[i][0].read_text()) for i in (0, 1))
    assert a == {"hop": "0", "soft": True}, "the nearly-done branch finished inside the grace window of hop 0"
    assert b["hop"] == "1", "the next branch of the chain must not be started after the notice"
    logs = _logs(project)
    assert "soft-stop notice (soft-stop-0)" in logs
    assert "soft-stop notice already given: branch not started" in logs
    assert len(_ran_jobs(project, sub.chains[0])) == 2, "hop 0 (branch a) and hop 1 (branch b)"


@pytest.mark.deadline
def test_payload_that_stops_on_the_notice_is_a_hop_boundary_not_a_failure(project):
    res = _results(project)
    out, progress = res / "o.json", res / "o.progress"
    # with real limits the work must outlast hop 0's usable time (notice after 90-120 s); with a sent signal a few
    # seconds of work around the notice are enough
    total, step = (150, 5) if DEADLINES == "slurm" else (8, 1)
    manifest_path = project.dir / "manifest.json"
    manifest_path.write_text(json.dumps({"schema": 1, "common": {"total": total, "step": step}, "branches": [
        {"id": 0, "outputs": [str(out)], "extra_outputs": [str(progress)]}]}))
    payload = project.write_payload(f"""
        import json, os, sys, time
        from pathlib import Path
        ctx = json.loads(Path(sys.argv[1]).read_text())
        total, step = ctx["common"]["total"], ctx["common"]["step"]
        prog = Path(ctx["extra_outputs"][0])
        done = int(prog.read_text()) if prog.exists() else 0
        while done < total:
            time.sleep(step)
            done += step
            prog.write_text(str(done))
            print(f"progress {{done}} in hop {{os.environ['HUMSUB_HOP']}}", flush=True)
            if os.path.exists(os.environ["HUMSUB_SOFT_STOP_FILE"]):
                sys.exit(3)                                # out of time: checkpointed, stop
        Path(ctx["outputs"][0]).write_text("done after hop " + os.environ["HUMSUB_HOP"])
    """)
    project.configure_deadline(time_limit="00:04:00", signal_seconds=150, grace_seconds=60, max_hops=5)
    sub = project.submit(manifest_path, payload, "boundary", "--retry-payload", "2")
    project.reach_deadline(sub.chains[0], 0, after="progress 3 in hop 0")
    assert list(sub.wait(timeout=500).values()) == ["finished"]
    assert out.read_text().startswith("done after hop"), out.read_text()
    assert out.read_text() != "done after hop 0", "the notice must have ended hop 0"
    logs = _logs(project)
    assert "payload stopped after the soft-stop notice" in logs
    assert "attempt 1 failed" not in logs, "a payload that yields at the deadline must not be retried or counted as failing"


@pytest.mark.supervisor
def test_supervisor_passes_retry_a_failed_job_without_a_controller(project):
    out = _results(project) / "o.json"
    manifest = project.write_manifest({0: [out]})
    payload = project.write_payload(f"""
        import os, sys
        from pathlib import Path
        flag = Path({str(out)!r} + ".failed-once")
        if not flag.exists():
            flag.write_text("x")
            sys.exit(4)                       # the first attempt fails, law's retry (a supervisor pass) succeeds
        Path({str(out)!r}).write_text(os.environ["SLURM_JOB_ID"])
    """)
    sub = project.submit(manifest, payload, "sup", "--supervisor-job", "--supervisor-interval", SUPERVISOR_INTERVAL, "--retries", "2")
    assert sub.proc.returncode == 0 and "[submit] supervisor  job" in sub.text, sub.text
    deadline = time.time() + 420
    while not out.exists() and time.time() < deadline:
        time.sleep(5)
    assert out.exists(), f"the supervisor's pass did not retry the job\n{project.squeue()}\n{_logs(project)[-2000:]}"
    project.wait_for_log("pass complete -> done", 120)  # the next pass finds the top task complete
    summary = project.humsub("submission-status", sub.id, check=True).stdout
    assert "1 done" in summary and "0 failed" in summary
    assert "pass complete -> done" in _logs(project)


@pytest.mark.supervisor
def test_lanes_and_law_retries_work_together(project):
    """--max-concurrent with law retries: a retry chain is created long after the initial submission; it must not wait on
    a lane predecessor that has ended, and the cap must still hold."""
    res = _results(project)
    outs = {i: [res / f"o{i}.json"] for i in range(3)}
    manifest = project.write_manifest(outs)
    payload = project.write_payload("""
        import json, sys, time
        from pathlib import Path
        ctx = json.loads(Path(sys.argv[1]).read_text())
        out = Path(ctx["outputs"][0])
        flag = out.with_suffix(".failed-once")
        if ctx["branch"] == 0 and not flag.exists():
            flag.write_text("x")
            sys.exit(4)
        start = time.time()
        time.sleep(4)
        out.write_text(json.dumps([start, time.time()]))
    """)
    sub = project.submit(manifest, payload, "lanes-retry", "--max-concurrent", "1", "--supervisor-job",
                         "--supervisor-interval", SUPERVISOR_INTERVAL, "--retries", "1")
    assert sub.proc.returncode == 0, sub.text
    deadline = time.time() + 420
    while not all(o[0].exists() for o in outs.values()) and time.time() < deadline:
        time.sleep(5)
    assert all(o[0].exists() for o in outs.values()), f"{project.squeue()}\n{_logs(project)[-3000:]}"
    spans = sorted(json.loads(o[0].read_text()) for o in outs.values())
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), f"lanes overlapped: {spans}"
    status = project.humsub("submission-status", sub.id, check=True).stdout
    assert "3 done" in status and "0 failed" in status, status
