from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
from unittest.mock import patch

import luigi
import pytest
from law.workflow.remote import BaseRemoteWorkflowProxy

from hummel_submit import tick
from hummel_submit.contrib.hummel.workflow import HummelWorkflowProxy
from hummel_submit.manifest_submit import SubmitOptions


def _spec(tmp_path: Path) -> Path:
    path = tmp_path / "sub" / "submission.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"submission_id": "sub", "submit_options": {"retries": 2, "tasks_per_job": 3}}))
    return path


class Done(luigi.Task):
    def complete(self) -> bool:
        return True


class Waiting(luigi.Task):
    def complete(self) -> bool:
        return False

    def run(self) -> None:  # e.g. a workflow that polled once and returned: incomplete, not failed
        assert os.environ.get(tick.TICK_ENV) == "1"


class Broken(luigi.Task):
    def complete(self) -> bool:
        return False

    def run(self) -> None:
        raise RuntimeError("jobs failed with no retries left")


class Downstream(luigi.Task):
    def requires(self):
        return Waiting()

    def complete(self) -> bool:
        return False

    def run(self) -> None:
        pass


@pytest.mark.parametrize("cls,state", [(Done, "complete"), (Waiting, "progress"), (Broken, "failed"), (Downstream, "progress")])
def test_pass_outcomes(tmp_path: Path, cls, state) -> None:
    with patch.object(tick, "instantiate", return_value=cls()):
        result = tick.run_tick(_spec(tmp_path))
    assert result.state == state
    if state == "failed":
        assert "no retries left" in result.errors[0]
    assert tick.TICK_ENV not in os.environ


def test_a_pass_backs_off_while_someone_else_holds_the_control_lock(tmp_path: Path) -> None:
    spec_path = _spec(tmp_path)
    (spec_path.parent / "control.lock").write_text(json.dumps(
        {"host": "elsewhere", "pid": 1, "purpose": "supervisor", "since": 0, "token": "other"}))
    result = tick.run_tick(spec_path)
    assert result.state == "busy" and "supervisor" in result.errors[0]


def test_old_submissions_get_the_manifest_task(tmp_path: Path) -> None:
    spec_path = _spec(tmp_path)
    desc = tick.law_task_of(spec_path, json.loads(spec_path.read_text()))
    assert desc["class"] == "ManifestPayloadWorkflow"
    assert desc["params"]["retries"] == 2 and desc["params"]["tasks_per_job"] == 3


def test_retry_counter_is_seeded_from_persisted_attempts() -> None:
    proxy = HummelWorkflowProxy.__new__(HummelWorkflowProxy)
    proxy._job_retries = defaultdict(int)
    proxy.job_data = type("JD", (), {"attempts": {1: 2, 3: 1}})()
    with patch.object(BaseRemoteWorkflowProxy, "poll", lambda self: None):
        proxy.poll()
    assert dict(proxy._job_retries) == {1: 2, 3: 1}


def test_law_retries_defaults_with_the_supervisor() -> None:
    assert SubmitOptions().law_retries() == 0
    assert SubmitOptions(supervisor_job=True).law_retries() == 3           # --supervisor-rounds
    assert SubmitOptions(supervisor_job=True, retries=1).law_retries() == 1
    assert SubmitOptions(supervisor_job=True, safety_margin=0.5).law_retries() == 3   # nested, not exclusive
    assert SubmitOptions(supervisor_job=True, retries=0).law_retries() == 0       # an explicit 0 is respected
    SubmitOptions(supervisor_job=True, retries=2).validate()                 # no --wait needed any more
