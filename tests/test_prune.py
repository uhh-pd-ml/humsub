from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from hummel_submit.cli import main
from hummel_submit.config import DEFAULTS
from hummel_submit.prune import plan_submission, select_submissions


class Env:
    def __init__(self, tmp: Path) -> None:
        self.out = tmp / "out"
        self.cache = tmp / "ssd"
        self.tmp = tmp
        self.out.mkdir()
        (self.out / "logs").mkdir()
        (self.tmp / ".hummel-submit.toml").write_text(
            f'[execution]\noutput_dir = "{self.out}"\ncache_dir = "{self.cache}"\nbulk_scratch_dir = "{self.tmp / 'bulk'}"\n[slurm]\njob_name = "job"\naccount = "a"\npartition = "p"\ngpus = 0\n'
        )

    def submission(self, sub_id: str, run: str, jobs: dict[str, list[str]], created: str, status: str = "completed",
                   supervisor: list[str] | None = None, lineage: str | None = None, outputs: list[Path] | None = None) -> None:
        cfg = copy.deepcopy(DEFAULTS)
        cfg["execution"].update(output_dir=str(self.out), cache_dir=str(self.cache), bulk_scratch_dir=str(self.tmp / "bulk"))
        cfg["slurm"]["job_name"] = "job"
        sdir = self.out / ".hummel-submit" / "submissions" / sub_id
        (sdir / "manifest-workflow").mkdir(parents=True)
        branches = [{"id": i, "outputs": [str(o)]} for i, o in enumerate(outputs or [])]
        (sdir / "manifest-workflow" / "manifest.json").write_text(json.dumps({"schema": 1, "branches": branches}))
        run_dir = self.out / "runs" / run
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "leftovers.txt").write_text("x")
        (sdir / "submission.json").write_text(json.dumps({
            "submission_id": sub_id, "run_name": run, "run_dir": str(run_dir), "output_dir": str(self.out), "created_at": created + "T10:00:00+00:00",
            "chain_ids": list(jobs), "config": cfg, "supervisor_job_ids": supervisor or [], "lineage": {"id": lineage or sub_id},
            "manifest_workflow": {"manifest": str(sdir / "manifest-workflow" / "manifest.json")},
        }))
        for chain, ids in jobs.items():
            cdir = self.out / ".hummel-submit" / "chains" / chain
            cdir.mkdir(parents=True)
            (cdir / "state.json").write_text(json.dumps({"jobs": ids, "status": status, "exit_code": 0}))
            for j in ids:
                (self.out / "logs" / f"job_{j}.log").write_text("log")
        for j in supervisor or []:
            (self.out / "logs" / f"job-supervisor_{j}.log").write_text("log")
        stage = self.cache / "stages" / sub_id
        stage.mkdir(parents=True)
        (stage / "proxy").write_text("secret")

    def run(self, *args: str) -> int:
        old = os.getcwd()
        os.chdir(self.tmp)
        try:
            with patch.dict(os.environ, {"BEEGFS": str(self.tmp), "USW": str(self.tmp), "SSD": str(self.tmp)}):
                return main(["prune", *args])
        finally:
            os.chdir(old)


@pytest.fixture
def env(tmp_path: Path) -> Env:
    e = Env(tmp_path)
    e.submission("20260921-100000-aaaaaaaa", "old-run", {"cOld": ["1", "2"]}, "2026-09-21")
    e.submission("20261008-120000-bbbbbbbb", "prod", {"c1": ["10", "11"], "c2": ["12"]}, "2026-10-08", supervisor=["90"])
    e.submission("20261009-120000-cccccccc", "prod-resume1", {"c3": ["13"]}, "2026-10-09", lineage="20261008-120000-bbbbbbbb")
    (e.out / "logs" / "job_99999.log").write_text("someone else's log")
    return e


def test_dry_run_deletes_nothing_and_says_what_is_untouched(env: Env, capsys) -> None:
    assert env.run("--run", "prod") == 0
    text = capsys.readouterr().out
    assert "dry run" in text and "2 other submission(s) of this output_dir are untouched" in text
    assert (env.out / "logs" / "job_10.log").exists() and (env.cache / "stages" / "20261008-120000-bbbbbbbb").exists()


def test_apply_removes_exactly_what_the_submission_owns(env: Env) -> None:
    assert env.run("--run", "prod", "--apply") == 0
    out = env.out
    assert not (out / ".hummel-submit" / "submissions" / "20261008-120000-bbbbbbbb").exists()
    assert not (out / ".hummel-submit" / "chains" / "c1").exists() and not (out / ".hummel-submit" / "chains" / "c2").exists()
    assert not (out / "runs" / "prod").exists()
    assert [p.name for p in sorted((out / "logs").iterdir())] == ["job-supervisor_90.log"][0:0] + sorted(
        ["job_1.log", "job_13.log", "job_2.log", "job_99999.log"])
    assert not (env.cache / "stages" / "20261008-120000-bbbbbbbb").exists(), "staged proxy copy is removed"
    # the older run, the resume (not selected without --lineage) and the foreign log are untouched
    assert (out / ".hummel-submit" / "submissions" / "20260921-100000-aaaaaaaa").exists()
    assert (out / ".hummel-submit" / "submissions" / "20261009-120000-cccccccc").exists()
    assert (out / ".hummel-submit" / "chains" / "cOld").exists() and (out / "runs" / "old-run").exists()


def test_lineage_flag_includes_resumes(env: Env) -> None:
    assert env.run("--run", "prod", "--lineage", "--apply") == 0
    assert not (env.out / ".hummel-submit" / "submissions" / "20261009-120000-cccccccc").exists()
    assert not (env.out / "logs" / "job_13.log").exists()
    assert (env.out / ".hummel-submit" / "submissions" / "20260921-100000-aaaaaaaa").exists()


def test_before_date_selects_by_creation(env: Env) -> None:
    assert env.run("--before", "2026-10-01", "--apply") == 0
    assert not (env.out / ".hummel-submit" / "submissions" / "20260921-100000-aaaaaaaa").exists()
    assert (env.out / ".hummel-submit" / "submissions" / "20261008-120000-bbbbbbbb").exists()


def test_active_submissions_are_refused_unless_forced(tmp_path: Path, capsys) -> None:
    e = Env(tmp_path)
    e.submission("20261008-120000-bbbbbbbb", "live", {"c1": ["10"]}, "2026-10-08", status="running-hop-1")
    import hummel_submit.chain as chain
    from hummel_submit.slurm import SlurmJobStatus
    orig = chain.query_jobs
    chain.query_jobs = lambda ids: {i: SlurmJobStatus(i, "RUNNING") for i in ids}
    try:
        assert e.run("--run", "live", "--apply") == 1
        assert "REFUSED" in capsys.readouterr().out
        assert (e.out / "runs" / "live").exists()
        assert e.run("--run", "live", "--apply", "--force") == 1 or not (e.out / "runs" / "live").exists()
        assert not (e.out / "runs" / "live").exists()
    finally:
        chain.query_jobs = orig


def test_never_deletes_a_directory_that_contains_a_declared_output(tmp_path: Path, capsys) -> None:
    e = Env(tmp_path)
    precious = e.out / "runs" / "prod" / "results" / "o.root"          # a manifest that wrote its outputs into the run dir
    precious.parent.mkdir(parents=True)
    precious.write_text("data")
    e.submission("20261008-120000-bbbbbbbb", "prod", {"c1": ["10"]}, "2026-10-08", outputs=[precious])
    assert e.run("--run", "prod", "--apply") == 0
    assert precious.read_text() == "data"
    assert "KEPT (safety)" in capsys.readouterr().out
    assert not (e.out / ".hummel-submit" / "submissions" / "20261008-120000-bbbbbbbb").exists()   # the rest is pruned


def test_selection_is_required(env: Env) -> None:
    assert env.run() == 2


def test_select_submissions_by_id_prefix(env: Env) -> None:
    got = select_submissions(env.out, [], None, ["20261009"])
    assert [p.parent.name for p in got] == ["20261009-120000-cccccccc"]
