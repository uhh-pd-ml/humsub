"""Single-command mode, --wait/--retries, follow/status, gc/cleanup, hop exhaustion."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import time

import pytest

from .conftest import squeue

TRAIN = """
    import json, os, sys
    from pathlib import Path
    run_dir = Path(os.environ["HUMMEL_RUN_DIR"])
    (run_dir / "result.json").write_text(json.dumps({
        "argv": sys.argv[1:], "foo": os.environ.get("FOO"), "quoted": os.environ.get("QUOTED"),
        "job_name": os.environ.get("SLURM_JOB_NAME"), "hop": os.environ.get("HUMSUB_HOP"),
        "cwd": os.getcwd(),
    }))
    print("training done", flush=True)
"""


def _single_config(project, **extra):
    (project.dir / ".env").write_text("# comment\nFOO=bar\nexport QUOTED='a b'\n")
    execution = {
        "command": [sys.executable, "train.py"],
        "auto_args": ["--run={RUN}", "--out={RUN_DIR}/result.json", "--ngpu={NGPU}"],
        "env_file": str(project.dir / ".env"),
    }
    execution.update(extra)
    return execution


def test_single_command_mode_and_follow(project):
    project.write_payload(TRAIN, "train.py")
    project.configure(execution=_single_config(project))
    sub = project.submit_single("single", "--epochs=3", "--ngpu=9")
    assert sub.proc.returncode == 0, sub.text
    assert len(sub.chains) == 1
    chain = sub.chains[0]

    follow = project.humsub("follow", chain, "-n", "5", timeout=150)
    assert follow.returncode == 0, follow.stdout + follow.stderr
    assert "training done" in follow.stdout

    result = json.loads((project.out / "runs" / "single" / "result.json").read_text())
    assert result["argv"][:2] == ["--run=single", f"--out={project.out}/runs/single/result.json"]
    assert "--epochs=3" in result["argv"]
    # a user-supplied option replaces the auto-generated one instead of duplicating it
    assert [a for a in result["argv"] if a.startswith("--ngpu")] == ["--ngpu=9"]
    assert result["foo"] == "bar" and result["quoted"] == "a b"      # env file, no shell evaluation
    assert result["cwd"] == str(project.dir)
    assert result["hop"] == "0"

    status = project.humsub("status", chain, check=True).stdout
    assert "completed" in status
    # a Slurm job id is accepted wherever a chain id is
    job_id = next(l for l in status.splitlines() if l.startswith("jobs:")).split(":", 1)[1].split(",")[0].strip()
    assert chain in project.humsub("status", job_id, check=True).stdout


def test_single_command_failure_is_reported_by_follow(project):
    project.write_payload("import sys\nprint('boom', flush=True)\nsys.exit(5)\n", "train.py")
    project.configure(execution=_single_config(project, auto_args=[]))
    sub = project.submit_single("failing")
    follow = project.humsub("follow", sub.chains[0], timeout=150)
    assert follow.returncode != 0
    assert "boom" in follow.stdout
    status = project.humsub("status", sub.chains[0]).stdout
    # the chain code is law's job exit code; the payload's own code is in the log
    assert re.search(r"status:\s+failed-\d+", status), status
    logs = "".join(p.read_text() for p in sorted((project.out / "logs").glob("*.log")))
    assert "payload finished with exit code 5" in logs


def test_single_command_dry_run_prints_the_sbatch_command(project):
    project.write_payload(TRAIN, "train.py")
    project.configure(execution=_single_config(project, auto_args=[]))
    r = project.humsub("submit", "--run-name", "ro", "--skip-path-checks", "--dry-run", "--", "--x=1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "sbatch" in r.stdout


def test_wait_mode_retries_a_failed_branch(project):
    out = project.dir / "results" / "o.json"
    (project.dir / "results").mkdir()
    marker = project.dir / "failed-once"
    manifest = project.write_manifest({0: [out]})
    # HUMSUB_ATTEMPT is law's in-job counter and stays 1 across controller-side retries
    # (see README), so the payload keeps its own state: fail once, then succeed.
    payload = project.write_payload(f"""
        import sys
        from pathlib import Path
        marker = Path({str(marker)!r})
        if not marker.exists():
            marker.write_text("x")
            print("failing the first time", flush=True)
            sys.exit(1)
        Path({str(out)!r}).write_text("ok after retry")
    """)
    sub = project.submit(manifest, payload, "retry", "--wait", "--retries", "1", timeout=400)
    assert sub.proc.returncode == 0, sub.text[-2500:]
    assert out.read_text() == "ok after retry"
    assert sorted(sub.chain_states().values()) == ["failed", "finished"]
    time.sleep(3)
    assert squeue().strip() == ""


def test_wait_mode_without_retries_reports_failure(project):
    out = project.dir / "results" / "o.json"
    (project.dir / "results").mkdir()
    manifest = project.write_manifest({0: [out]})
    payload = project.write_payload("import sys\nsys.exit(2)\n")
    sub = project.submit(manifest, payload, "nowait-retry", "--wait", timeout=300)
    assert sub.proc.returncode != 0
    assert not out.exists()


def test_wait_mode_parallel_jobs(project):
    (project.dir / "results").mkdir()
    outs = {i: [project.dir / "results" / f"o{i}"] for i in range(3)}
    manifest = project.write_manifest(outs)
    payload = project.write_payload("""
        import json, sys, time
        from pathlib import Path
        out = Path(json.loads(Path(sys.argv[1]).read_text())["outputs"][0])
        time.sleep(2)
        out.write_text("x")
    """)
    sub = project.submit(manifest, payload, "par", "--wait", "--parallel-jobs", "1", timeout=400)
    assert sub.proc.returncode == 0, sub.text[-2500:]
    assert all(o[0].exists() for o in outs.values())


def test_follow_manifest_chain(project):
    out = project.dir / "results" / "o.json"
    (project.dir / "results").mkdir()
    manifest = project.write_manifest({0: [out]})
    payload = project.write_payload(f"""
        import time
        from pathlib import Path
        print("working", flush=True)
        time.sleep(3)
        Path({str(out)!r}).write_text("x")
    """)
    sub = project.submit(manifest, payload, "follow")
    follow = project.humsub("follow", sub.chains[0], timeout=150)
    assert follow.returncode == 0, follow.stdout + follow.stderr
    assert "working" in follow.stdout and out.exists()


def test_gc_and_cleanup(project):
    (project.dir / "lookup").mkdir()
    (project.dir / "lookup" / "f.txt").write_text("1")
    (project.dir / "results").mkdir()
    out = project.dir / "results" / "o"
    manifest = project.write_manifest({0: [out]})
    payload = project.write_payload(f"from pathlib import Path\nPath({str(out)!r}).write_text('x')\n")
    sub = project.submit(manifest, payload, "gc", "--stage", f"lookup={project.dir / 'lookup'}")
    assert set(sub.wait().values()) == {"finished"}
    stages = project.cache / "stages"

    # an unreferenced old staging dir, and one owned by another (still existing) submission
    orphan = stages / "orphan-1"
    orphan.mkdir(parents=True)
    (orphan / "x").write_text("x")
    foreign_spec = project.dir / "other-submission.json"
    foreign_spec.write_text("{}")
    foreign = stages / "foreign-1"
    foreign.mkdir()
    (foreign / ".humsub-owner").write_text(str(foreign_spec) + "\n")
    old = time.time() - 48 * 3600
    for d in (orphan, foreign):
        os.utime(d, (old, old))

    dry = project.humsub("gc", check=True).stdout
    assert "DRY RUN" in dry and (stages / sub.id).exists() and orphan.exists()

    applied = project.humsub("gc", "--apply", "--successful-after-hours", "0",
                             "--orphans-after-hours", "24", check=True).stdout
    assert "APPLY" in applied
    assert not (stages / sub.id).exists(), "finished submission's staging must be removed"
    assert not orphan.exists(), "old unreferenced staging must be removed"
    assert foreign.exists(), "staging owned by another live submission must be spared"
    # provenance and outputs are preserved
    assert out.exists() and (project.out / ".hummel-submit" / "submissions" / sub.id / "submission.json").exists()

    # cleanup of an already-clean submission is harmless, and --dry-run removes nothing
    again = project.humsub("cleanup", sub.id, "--dry-run")
    assert again.returncode == 0, again.stdout + again.stderr


def test_cleanup_refuses_active_submission_without_force(project):
    (project.dir / "results").mkdir()
    out = project.dir / "results" / "o"
    manifest = project.write_manifest({0: [out]})
    payload = project.write_payload("import time\ntime.sleep(60)\n")
    sub = project.submit(manifest, payload, "active")
    for _ in range(30):
        if "running" in sub.chain_states().values():
            break
        time.sleep(1)
    refused = project.humsub("cleanup", sub.id)
    assert refused.returncode != 0 and "still active" in (refused.stdout + refused.stderr)
    forced = project.humsub("cleanup", sub.id, "--force")
    assert forced.returncode == 0, forced.stdout + forced.stderr
    project.humsub("cancel", sub.chains[0])


@pytest.mark.slow
def test_single_command_resumes_from_checkpoint(project):
    project.write_payload("""
        import os, sys, time
        from pathlib import Path
        run_dir = Path(os.environ["HUMMEL_RUN_DIR"])
        resume = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--resume=")][0]
        print("resume argument:", resume, flush=True)
        if resume == "null":
            (run_dir / "ckpt-0.txt").write_text("state after hop 0")
            time.sleep(600)                      # stopped by the pre-timeout signal
        (run_dir / "final.txt").write_text(Path(resume).read_text())
    """, "train.py")
    project.configure(
        execution=_single_config(project, auto_args=["--resume={CKPT}"], checkpoint_glob="ckpt-*.txt"),
        time_limit="00:01:00", signal_seconds=30, max_hops=3,
    )
    sub = project.submit_single("resume")
    follow = project.humsub("follow", sub.chains[0], timeout=300)
    assert follow.returncode == 0, follow.stdout + follow.stderr
    assert (project.out / "runs" / "resume" / "final.txt").read_text() == "state after hop 0"
    logs = "".join(p.read_text() for p in sorted((project.out / "logs").glob("*.log")))
    assert "resume argument: null" in logs and "resuming from" in logs


@pytest.mark.slow
def test_hop_limit_exhausted_marks_chain_failed(project):
    out = project.dir / "results" / "o"
    (project.dir / "results").mkdir()
    manifest = project.write_manifest({0: [out]})
    payload = project.write_payload("import time\ntime.sleep(600)\n")
    project.configure(time_limit="00:01:00", signal_seconds=30, max_hops=2)
    sub = project.submit(manifest, payload, "limit")
    states = sub.wait(timeout=400)
    assert list(states.values()) == ["failed"], states
    status = project.humsub("status", sub.chains[0]).stdout
    assert "stopped-no-follower" in status, status
    assert not out.exists()
