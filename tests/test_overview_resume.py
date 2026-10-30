from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from hummel_submit import overview, resume as resume_mod
from hummel_submit.chain import ChainStatus
from hummel_submit.config import DEFAULTS, ConfigError
from hummel_submit.scratch import sweep_stale_scratch
from hummel_submit.supervisor import decide
from hummel_submit.tick import TickResult


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _config(tmp_path: Path) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    cfg["execution"].update(cache_dir=str(tmp_path / "ssd"), bulk_scratch_dir=str(tmp_path / "bulk"))
    cfg["slurm"].update(nice=5, partition="std", gpus=0)
    return cfg


def _make_submission(tmp_path: Path):
    """6 branches in 3 law jobs (2 branches each) on chains cA (finished), cB (failed, 1 retry used, its first chain
    cB0 failed too), cC (running); outputs of branches 0, 1 exist."""
    out = tmp_path / "out"
    results = tmp_path / "results"
    results.mkdir()
    branches = [{"id": i, "data": {}, "outputs": [str(results / f"o{i}.txt")]} for i in range(6)]
    for i in (0, 1):
        (results / f"o{i}.txt").write_text("x")
    sdir = out / ".hummel-submit" / "submissions" / "sub1"
    manifest = sdir / "manifest-workflow" / "manifest.json"
    _write(manifest, {"schema": 1, "common": {}, "branches": branches})
    stage = tmp_path / "ssd" / "stages" / "sub1" / "code"
    stage.mkdir(parents=True)
    source = tmp_path / "src"
    source.mkdir()
    (source / "new.txt").write_text("renewed")
    params = {"humsub_spec": str(sdir / "submission.json"), "workflow": "hummel", "retries": 1, "tasks_per_job": 2,
              "parallel_jobs": 0, "job_workers": 1}
    spec = {
        "schema": 2, "submission_id": "sub1", "run_name": "run", "output_dir": str(out), "project_dir": str(tmp_path),
        "law_dir": str(sdir / "law"), "chain_ids": ["cA", "cB0", "cB", "cC"], "supervisor_job_ids": ["77"],
        "manifest_workflow": {"manifest": str(manifest), "stages": {"code": str(stage)},
                              "stage_sources": {"code": str(source)}, "stage_excludes": {}},
        "law_task": {"module": "hummel_submit.manifest_workflow", "class": "ManifestPayloadWorkflow", "params": params},
        "submit_options": {"tasks_per_job": 2, "supervisor_job": True},
        "config": _config(tmp_path),
    }
    _write(sdir / "submission.json", spec)
    _write(sdir / "law" / "control" / "hummel_jobs_0To6.json", {
        "jobs": {"1": {"job_id": "cA", "branches": [0, 1]}, "2": {"job_id": "cB", "branches": [2, 3]},
                 "3": {"job_id": "cC", "branches": [4, 5]}},
        "unsubmitted_jobs": {}, "attempts": {"2": 1},
    })
    return out, sdir / "submission.json", spec


STATES = {"cA": ChainStatus("cA", "finished", 0), "cB0": ChainStatus("cB0", "failed", 60, "failed-60"),
          "cB": ChainStatus("cB", "failed", 60, "failed-60"), "cC": ChainStatus("cC", "running")}


def _states():
    return patch("hummel_submit.overview.query_chain", side_effect=lambda o, c: STATES[c])


def test_overview_follows_law_jobs_not_old_chains(tmp_path: Path) -> None:
    out, _, spec = _make_submission(tmp_path)
    with _states():
        ov = overview.compute_overview(out, spec)
    assert (ov.total, ov.done, ov.active, ov.waiting, ov.missing) == (6, 2, 2, 0, [2, 3])
    assert ov.chain_states == {"finished": 1, "failed": 2, "running": 1}
    assert [(j.num, j.chain_id, j.attempts, e) for j, e in ov.failed_jobs] == [(2, "cB", 1, "failed-60")]
    text = "\n".join(overview.format_overview(ov))
    assert "2 in flight | 2 failed" in text and "humsub resume sub1" in text and "cB0" not in text


def test_overview_counts_branches_law_has_not_submitted(tmp_path: Path) -> None:
    out, _, spec = _make_submission(tmp_path)
    jobs = Path(spec["law_dir"]) / "control" / "hummel_jobs_0To6.json"
    data = json.loads(jobs.read_text())
    del data["jobs"]["3"]
    data["unsubmitted_jobs"] = {"3": [4, 5]}
    jobs.write_text(json.dumps(data))
    with _states():
        ov = overview.compute_overview(out, spec)
    assert (ov.active, ov.waiting, ov.missing) == (0, 2, [2, 3])


def test_overview_reads_jobs_files_of_submissions_before_the_hummel_workflow_type(tmp_path: Path) -> None:
    out, _, spec = _make_submission(tmp_path)
    control = Path(spec["law_dir"]) / "control"
    (control / "hummel_jobs_0To6.json").rename(control / "slurm_jobs_0To6.json")
    with _states():
        assert overview.compute_overview(out, spec).missing == [2, 3]


def test_find_submission_by_id_prefix_and_run_name(tmp_path: Path) -> None:
    out, spec_path, _ = _make_submission(tmp_path)
    assert overview.find_submission(out, "sub1") == spec_path
    assert overview.find_submission(out, "run") == spec_path
    assert overview.find_submission(out, "su") == spec_path
    with pytest.raises(ConfigError):
        overview.find_submission(out, "nothing")


def _resume(out, spec_path, **kwargs):
    calls = {}
    with _states(), patch.object(resume_mod, "query_chain", side_effect=lambda o, c: STATES[c]), \
         patch.object(resume_mod, "sweep_stale_scratch", return_value=(0, 0)), \
         patch.object(resume_mod, "run_tick", side_effect=lambda p, purpose="": calls.setdefault("tick", TickResult("progress"))), \
         patch("hummel_submit.slurm.live_job_ids", return_value=set()), \
         patch("hummel_submit.supervisor.submit_supervisor", side_effect=lambda p, round_no: calls.setdefault("sup", "99")):
        result = resume_mod.resume(out, spec_path, **kwargs)
    return result, calls


def test_resume_resets_failed_jobs_applies_settings_and_requeues_the_supervisor(tmp_path: Path) -> None:
    out, spec_path, spec = _make_submission(tmp_path)
    result, calls = _resume(out, spec_path, config_overrides={"slurm": {"time_limit": "04:00:00"}}, retries=2,
                            stage_overrides={"code": tmp_path / "src"})
    assert result.state == "progress" and "tick" in calls and calls["sup"] == "99"
    jobs = json.loads((Path(spec["law_dir"]) / "control" / "hummel_jobs_0To6.json").read_text())
    assert jobs["attempts"] == {"2": 0}                         # only the failed job; the running one is untouched
    new = json.loads(spec_path.read_text())
    assert new["config"]["slurm"]["time_limit"] == "04:00:00"   # used by every chain created from now on
    assert new["law_task"]["params"]["retries"] == 2
    assert (Path(spec["manifest_workflow"]["stages"]["code"]) / "new.txt").read_text() == "renewed"


def test_resume_dry_run_changes_nothing(tmp_path: Path) -> None:
    out, spec_path, spec = _make_submission(tmp_path)
    before = spec_path.read_text()
    result, calls = _resume(out, spec_path, retries=3, dry_run=True)
    assert result is None and not calls and spec_path.read_text() == before


def test_resume_grants_at_least_one_retry(tmp_path: Path) -> None:
    out, spec_path, spec = _make_submission(tmp_path)
    spec["law_task"]["params"]["retries"] = 0
    spec_path.write_text(json.dumps(spec))
    _resume(out, spec_path)
    assert json.loads(spec_path.read_text())["law_task"]["params"]["retries"] == 1


def test_resume_refuses_unknown_stages_and_old_submissions(tmp_path: Path) -> None:
    out, spec_path, spec = _make_submission(tmp_path)
    with pytest.raises(ConfigError, match="unknown stage"):
        _resume(out, spec_path, stage_overrides={"proxy": tmp_path})
    del spec["law_task"]
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(ConfigError, match="predates law passes"):
        _resume(out, spec_path)


def test_sweep_removes_only_dead_jobs(tmp_path: Path) -> None:
    cfg = {"execution": {"cache_dir": str(tmp_path / "ssd"), "bulk_scratch_dir": str(tmp_path / "bulk")}}
    dead = tmp_path / "bulk" / "sub1" / "10-1"
    dead_finalize = tmp_path / "bulk" / "sub1" / "12-finalize"
    live = tmp_path / "ssd" / "payload-work" / "sub1" / "11-2"
    for d in (dead, dead_finalize, live):
        d.mkdir(parents=True)
        (d / "ntuple.root").write_bytes(b"x" * 1000)
    with patch("hummel_submit.slurm.live_job_ids", return_value={"11"}):
        assert sweep_stale_scratch(cfg, None, apply=False)[0] == 2
        assert dead.exists()
        count, nbytes = sweep_stale_scratch(cfg, None, apply=True)
    assert count == 2 and nbytes >= 2000 and not dead.exists() and not dead_finalize.exists() and live.exists()


@pytest.mark.parametrize(
    "tick_state,checks,expected",
    [
        ("complete", 1, "done"),
        ("progress", 1, "wait"),
        ("busy", 5, "wait"),        # another pass (e.g. a manual `humsub tick`) holds the lock
        ("failed", 1, "give-up"),   # law reports jobs failed with no retries left
        ("progress", 2000, "give-up"),
    ],
)
def test_supervisor_decisions(tick_state, checks, expected) -> None:
    assert decide(tick_state=tick_state, checks=checks) == expected


def test_supervisor_can_use_its_own_account_and_partition(tmp_path: Path) -> None:
    """GPU chains run under a GPU account that cannot submit a 1-CPU job to the CPU partition."""
    from hummel_submit.supervisor import supervisor_command

    out, spec_path, spec = _make_submission(tmp_path)
    spec["python_executable"] = "/py"
    spec["config"]["slurm"].update(account="grp_gpu", partition="gpu", supervisor_account="grp_std", supervisor_partition="std")
    spec_path.write_text(json.dumps(spec))
    args = supervisor_command(spec_path, 1)
    assert "--account=grp_std" in args and "--partition=std" in args
    spec["config"]["slurm"].update(supervisor_account="", supervisor_partition="")
    spec_path.write_text(json.dumps(spec))
    args = supervisor_command(spec_path, 1)
    assert "--account=grp_gpu" in args and "--partition=gpu" in args
    assert not any(a.startswith("--nice") for a in args), "the watchdog must not be starved by the chains' nice penalty"
