from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from hummel_submit import lineage
from hummel_submit.chain import ChainStatus
from hummel_submit.config import DEFAULTS, ConfigError
from hummel_submit.supervisor import decide


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _config(tmp_path: Path) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    cfg["execution"].update(cache_dir=str(tmp_path / "ssd"), bulk_scratch_dir=str(tmp_path / "bulk"))
    cfg["slurm"].update(nice=5, partition="std", gpus=0)
    return cfg


def _make_lineage(tmp_path: Path):
    """Root submission with 3 chains of 2 branches (ids 0-5); outputs of branches 0,1 exist."""
    out = tmp_path / "out"
    results = tmp_path / "results"
    results.mkdir()
    branches = [{"id": i, "data": {}, "outputs": [str(results / f"o{i}.txt")]} for i in range(6)]
    for i in (0, 1):
        (results / f"o{i}.txt").write_text("x")
    sdir = out / ".hummel-submit" / "submissions" / "sub1"
    manifest = sdir / "manifest-workflow" / "manifest.json"
    _write(manifest, {"schema": 1, "common": {}, "branches": branches, "humsub": {"scratch": "beegfs"}})
    payload = sdir / "manifest-workflow" / "payload" / "p.py"
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_text("#!/bin/sh\n")
    spec = {
        "schema": 2, "submission_id": "sub1", "run_name": "run", "output_dir": str(out), "project_dir": str(tmp_path),
        "chain_ids": ["cA", "cB", "cC"],
        "manifest_workflow": {"manifest": str(manifest), "payload": str(payload), "stage_sources": {}, "stage_excludes": {}},
        "submit_options": {"tasks_per_job": 2},
        "config": _config(tmp_path),
    }
    _write(sdir / "submission.json", spec)
    _write(sdir / "law" / "control" / "slurm_jobs_0To6.json", {"jobs": {
        "1": {"job_id": "cA", "branches": [0, 1]},
        "2": {"job_id": "cB", "branches": [2, 3]},
        "3": {"job_id": "cC", "branches": [4, 5]},
    }, "unsubmitted_jobs": []})
    return out, sdir / "submission.json", spec


STATES = {"cA": ChainStatus("cA", "finished", 0), "cB": ChainStatus("cB", "failed", 60, "failed-60"),
          "cC": ChainStatus("cC", "running")}


def test_chain_branch_map(tmp_path: Path) -> None:
    _, spec_path, _ = _make_lineage(tmp_path)
    assert lineage.chain_branch_map(spec_path) == {"cA": [0, 1], "cB": [2, 3], "cC": [4, 5]}


def test_overview_separates_done_active_and_missing(tmp_path: Path) -> None:
    out, _, spec = _make_lineage(tmp_path)
    with patch.object(lineage, "query_chain", side_effect=lambda o, c: STATES[c]):
        ov = lineage.compute_overview(out, spec)
    assert (ov.total, ov.done, ov.active, ov.missing) == (6, 2, 2, [2, 3])
    assert ov.chain_states == {"finished": 1, "failed": 1, "running": 1}
    assert ov.failed_chains == [("cB", "failed-60", [2, 3])]
    text = "\n".join(lineage.format_overview(ov))
    assert "2 in flight" in text and "humsub resume" in text


def test_find_submission_by_id_prefix_and_run_name(tmp_path: Path) -> None:
    out, spec_path, _ = _make_lineage(tmp_path)
    assert lineage.find_submission(out, "sub1") == spec_path
    assert lineage.find_submission(out, "run") == spec_path
    assert lineage.find_submission(out, "su") == spec_path
    with pytest.raises(ConfigError):
        lineage.find_submission(out, "nothing")


def test_resume_submits_only_missing_and_idle_branches(tmp_path: Path) -> None:
    out, spec_path, spec = _make_lineage(tmp_path)
    captured = {}

    def fake_submit(**kwargs):
        captured.update(kwargs)
        captured["subset"] = json.loads(Path(kwargs["manifest_source"]).read_text())
        return {"submission_id": "new"}

    with patch.object(lineage, "query_chain", side_effect=lambda o, c: STATES[c]), \
         patch("hummel_submit.manifest_submit.submit_manifest_run", fake_submit), \
         patch.object(lineage, "sweep_stale_scratch", return_value=(0, 0)):
        lineage.resume(out, spec_path, option_overrides={"max_concurrent": 10})
    assert [b["id"] for b in captured["subset"]["branches"]] == [2, 3]
    assert captured["subset"]["humsub"] == {"scratch": "beegfs"}  # hints survive
    assert captured["run_name"] == "run-resume1"
    assert captured["opts"].max_concurrent == 10 and captured["opts"].tasks_per_job == 2
    assert captured["opts"].wait is False and captured["opts"].supervisor_job is False
    assert captured["lineage"]["id"] == "sub1" and captured["parent_submission"] == "sub1"
    assert captured["allow_supervisor"] is True


def test_resume_include_active_adds_running_branches(tmp_path: Path) -> None:
    out, spec_path, _ = _make_lineage(tmp_path)
    seen = {}
    with patch.object(lineage, "query_chain", side_effect=lambda o, c: STATES[c]), \
         patch("hummel_submit.manifest_submit.submit_manifest_run",
               lambda **kw: seen.update(ids=[b["id"] for b in json.loads(Path(kw["manifest_source"]).read_text())["branches"]])), \
         patch.object(lineage, "sweep_stale_scratch", return_value=(0, 0)):
        lineage.resume(out, spec_path, include_active=True)
    assert seen["ids"] == [2, 3, 4, 5]


def test_resume_with_nothing_missing_is_a_noop(tmp_path: Path, capsys) -> None:
    out, spec_path, spec = _make_lineage(tmp_path)
    for i in range(6):
        (tmp_path / "results" / f"o{i}.txt").write_text("x")
    with patch.object(lineage, "query_chain", side_effect=lambda o, c: STATES[c]), \
         patch.object(lineage, "sweep_stale_scratch", return_value=(0, 0)):
        assert lineage.resume(out, spec_path) is None
    assert "nothing to resume" in capsys.readouterr().out


def test_sweep_removes_only_dead_jobs(tmp_path: Path) -> None:
    cfg = {"execution": {"cache_dir": str(tmp_path / "ssd"), "bulk_scratch_dir": str(tmp_path / "bulk")}}
    dead = tmp_path / "bulk" / "sub1" / "10-1"
    live = tmp_path / "ssd" / "payload-work" / "sub1" / "11-2"
    for d in (dead, live):
        d.mkdir(parents=True)
        (d / "ntuple.root").write_bytes(b"x" * 1000)
    with patch("hummel_submit.slurm.live_job_ids", return_value={"11"}):
        assert lineage.sweep_stale_scratch(cfg, None, apply=False)[0] == 1
        assert dead.exists()
        count, nbytes = lineage.sweep_stale_scratch(cfg, None, apply=True)
    assert count == 1 and nbytes >= 1000 and not dead.exists() and live.exists()


@pytest.mark.parametrize(
    "missing,active,round_no,checks,expected",
    [
        (0, 0, 1, 1, "done"),
        (3, 5, 1, 1, "wait"),
        (3, 0, 1, 1, "resume"),
        (3, 0, 3, 7, "resume"),
        (3, 0, 4, 9, "give-up"),   # more resume rounds than allowed: a bad input cannot loop forever
        (0, 5, 1, 1, "wait"),
    ],
)
def test_supervisor_decisions(missing, active, round_no, checks, expected) -> None:
    assert decide(missing=missing, active_branches=active, round_no=round_no, max_rounds=3, checks=checks) == expected


def test_supervisor_can_use_its_own_account_and_partition(tmp_path: Path) -> None:
    """GPU chains run under a GPU account that cannot submit a 1-CPU job to the CPU partition."""
    from hummel_submit.supervisor import supervisor_command

    out, spec_path, spec = _make_lineage(tmp_path)
    spec["python_executable"] = "/py"
    spec["config"]["slurm"].update(account="grp_gpu", partition="gpu", supervisor_account="grp_std", supervisor_partition="std")
    spec_path.write_text(json.dumps(spec))
    args = supervisor_command(spec_path, 1)
    assert "--account=grp_std" in args and "--partition=std" in args
    spec["config"]["slurm"].update(supervisor_account="", supervisor_partition="")
    spec_path.write_text(json.dumps(spec))
    args = supervisor_command(spec_path, 1)
    assert "--account=grp_gpu" in args and "--partition=gpu" in args
