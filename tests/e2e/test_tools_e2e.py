"""humsub doctor / stats / prune against a real Slurm."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .conftest import sacct_states


def _results(project) -> Path:
    d = project.dir / "results"
    d.mkdir(exist_ok=True)
    return d


def test_doctor_probe_checks_a_real_batch_node(project):
    r = project.humsub("doctor", "--probe", "--timeout", "170", "--check-path", str(project.dir / "results"), timeout=240)
    text = r.stdout + r.stderr
    assert r.returncode == 0, text
    assert "[  ok] output_dir" in text and "[  ok] scheduler" in text
    assert "probe: writable from the batch node" in text
    assert "probe: nested sbatch" in text and "[  ok] probe: nested sbatch" in text, "a batch job must be able to submit jobs (--supervisor-job)"
    assert "[  ok] probe: humsub importable in batch" in text
    assert "doctor: 0 failed" in text
    leftovers = [p for p in (project.out / "logs").glob("*") if "doctor" in p.name]
    assert leftovers == [], f"the probe must clean up after itself: {leftovers}"
    assert not [l for l in os.popen("squeue -h -o %j").read().split() if "doctor" in l], "no probe job left in the queue"


def test_doctor_reports_an_unusable_location_and_exits_nonzero(project):
    bad = Path(os.environ["USW"]) / "nope"            # read-only in batch jobs: must be refused
    r = project.humsub("doctor", "--check-path", str(bad))
    assert r.returncode == 1, r.stdout + r.stderr
    assert "[fail] path" in r.stdout and "1 failed" in r.stdout


def test_stats_and_prune_after_two_runs(project):
    res = _results(project)
    payload = project.write_payload("""
        import json, sys
        from pathlib import Path
        ctx = json.loads(Path(sys.argv[1]).read_text())
        Path(ctx["outputs"][0]).write_text(ctx["run_name"])
    """)
    runs = {}
    for name in ("keep", "gone"):
        outs = {i: [res / f"{name}-{i}.json"] for i in range(2)}
        manifest = project.write_manifest(outs)
        sub = project.submit(manifest, payload, name)
        assert set(sub.wait().values()) == {"finished"}, sub.text
        runs[name] = (sub, outs)

    sub, outs = runs["gone"]
    stats = json.loads(project.humsub("stats", sub.id, "--json", check=True).stdout)
    assert stats["states"].get("COMPLETED") == 2, stats
    assert stats["ran_jobs"] >= 2 and stats["peak_concurrency"] >= 1 and stats["core_hours"] > 0
    assert stats["failed_jobs"] == [] and stats["chains"] == 2
    text = project.humsub("stats", sub.id, check=True).stdout
    assert "core-hours" in text and "peak running" in text

    chains = json.loads((project.out / ".hummel-submit" / "chains" / sub.chains[0] / "state.json").read_text())["jobs"]
    logs = lambda ids: [project.out / "logs" / f"{project.job_name}_{j}.log" for j in ids]
    keep_chains = json.loads((project.out / ".hummel-submit" / "chains" / runs["keep"][0].chains[0] / "state.json").read_text())["jobs"]
    existing = lambda ids: [p for p in logs(ids) if p.exists()]       # a cancelled follower never wrote a log
    assert existing(chains), "the job ran and wrote a log"

    dry = project.humsub("prune", "--run", "gone", check=True)
    assert "dry run" in dry.stdout and "1 other submission(s)" in dry.stdout
    assert existing(chains), "a dry run deletes nothing"

    done = project.humsub("prune", "--run", "gone", "--apply", check=True)
    assert "declared outputs of the manifests were never touched" in done.stdout
    assert not existing(chains), "the logs of exactly its jobs are gone"
    assert not (project.out / "runs" / "gone").exists()
    assert not (project.out / ".hummel-submit" / "submissions" / sub.id).exists()
    assert not (project.cache / "stages" / sub.id).exists()
    assert all(o[0].exists() for o in outs.values()), "declared outputs are never deleted"
    # the other run is untouched and still fully usable
    keep = runs["keep"][0]
    assert existing(keep_chains) and (project.out / "runs" / "keep").exists()
    assert "finished" in project.humsub("submission-status", keep.id, check=True).stdout
