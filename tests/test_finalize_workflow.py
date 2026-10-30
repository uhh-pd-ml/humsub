from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from hummel_submit import overview
from hummel_submit.chain import ChainStatus
from hummel_submit.manifest import freeze_manifest_workflow
from hummel_submit.manifest_workflow import ManifestFinalizeWorkflow, ManifestPayloadWorkflow
from hummel_submit.tick import manifest_law_task

from test_overview_resume import _write


def _frozen(tmp_path: Path) -> Path:
    """A submission frozen from a manifest with a finalize step, wired like submit_manifest_run does."""
    sdir = tmp_path / "out" / ".hummel-submit" / "submissions" / "s1"
    spec_path = sdir / "submission.json"
    _write(spec_path, {"submission_id": "s1", "run_name": "r", "run_dir": str(tmp_path / "run"),
                       "law_dir": str(sdir / "law"), "output_dir": str(tmp_path / "out")})
    source = tmp_path / "m.json"
    source.write_text(json.dumps({
        "schema": 1, "common": {"c": 1},
        "branches": [{"id": i, "data": {"inputs": [f"/in{i}"]}, "outputs": [str(tmp_path / f"o{i}")]} for i in range(3)],
        "finalize": {"data": {"command": ["merge"]}, "outputs": [str(tmp_path / "merged")]},
    }))
    payload = tmp_path / "p.sh"
    payload.write_text("#!/bin/sh\n")
    spec = freeze_manifest_workflow(spec_path, manifest_source=source, payload_source=payload, stages={"code": "/s"})
    opts = {"retries": 2, "tasks_per_job": 2, "parallel_jobs": 0}
    spec["manifest_workflow"]["branch_task"] = manifest_law_task(spec_path, opts)
    spec["law_task"] = manifest_law_task(spec_path, dict(opts, tasks_per_job=1), "ManifestFinalizeWorkflow")
    spec_path.write_text(json.dumps(spec))
    return spec_path


def test_freeze_writes_the_finalize_context(tmp_path: Path) -> None:
    spec = json.loads(_frozen(tmp_path).read_text())
    ctx = json.loads(Path(spec["manifest_workflow"]["finalize_file"]).read_text())
    assert ctx["branch"] == "finalize" and ctx["common"] == {"c": 1} and ctx["stages"] == {"code": "/s"}
    assert ctx["data"] == {"command": ["merge"], "finalize": True}
    assert ctx["outputs"] == [str(tmp_path / "merged")]


def test_finalize_workflow_requires_the_branch_workflow(tmp_path: Path) -> None:
    spec_path = _frozen(tmp_path)
    fin = ManifestFinalizeWorkflow(humsub_spec=str(spec_path), workflow="slurm", retries=2, tasks_per_job=1)
    assert list(fin.create_branch_map()) == [0]
    assert fin.create_branch_map()[0]["outputs"] == [str(tmp_path / "merged")]
    branches = fin.workflow_requires()["branches"]
    assert type(branches) is ManifestPayloadWorkflow
    assert branches.tasks_per_job == 2 and branches.retries == 2 and branches.humsub_spec == str(spec_path)
    assert fin.slurm_output_directory().path.endswith("/law/control/finalize")
    label, context = fin.req(fin, branch=0)._branch_context(json.loads(spec_path.read_text())["manifest_workflow"])
    assert label == "finalize" and context.name == "finalize.json"


def test_finalize_state_line(tmp_path: Path) -> None:
    spec_path = _frozen(tmp_path)
    spec = json.loads(spec_path.read_text())
    assert overview.finalize_state(spec).startswith("waiting for the branches")
    _write(Path(spec["law_dir"]) / "control" / "finalize" / "slurm_jobs_0To1.json",
           {"jobs": {"1": {"job_id": "cF", "status": None}}, "attempts": {"1": 1}})
    with patch.object(overview, "query_chain", return_value=ChainStatus("cF", "running")):
        assert overview.finalize_state(spec) == "chain cF running (1 retry used)"
    (tmp_path / "merged").write_text("{}")
    assert overview.finalize_state(spec) == "done"
