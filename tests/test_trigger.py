from __future__ import annotations

import json
import os
from pathlib import Path
import time
from unittest.mock import patch

from hummel_submit import trigger, worker
from hummel_submit.locks import LOCK_NAME
from hummel_submit.trigger import MARKER


class _Proc:
    def __init__(self, rc: int = 0, out: str = "", err: str = "") -> None:
        self.returncode, self.stdout, self.stderr = rc, out, err


def _scontrol_show(state: str, start_in: float) -> str:
    start = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + start_in))
    return f"JobId=77 JobName=e2e-supervisor JobState={state} Reason=BeginTime StartTime={start} EndTime=Unknown"


def test_trigger_delay_is_capped_by_the_interval() -> None:
    assert trigger.trigger_delay({"supervisor_interval": 1800}) == 60
    assert trigger.trigger_delay({"supervisor_interval": 1800, "supervisor_trigger_delay": 5}) == 5
    assert trigger.trigger_delay({"supervisor_interval": 30, "supervisor_trigger_delay": 60}) == 30


def test_a_pending_supervisor_is_brought_forward_but_never_postponed() -> None:
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return _Proc(out=_scontrol_show("PENDING", 1500)) if "show" in cmd else _Proc()

    with patch("subprocess.run", side_effect=run):
        assert "brought forward to start in 60 s" in trigger.bring_forward("77", 60)
    assert calls[-1] == ["scontrol", "update", "JobId=77", "StartTime=now+60"]

    calls.clear()
    with patch("subprocess.run", side_effect=lambda cmd, **kw: calls.append(cmd) or _Proc(out=_scontrol_show("PENDING", 20))):
        assert "starts within 60 s anyway" in trigger.bring_forward("77", 60)
    assert all("update" not in c for c in calls)

    with patch("subprocess.run", return_value=_Proc(out=_scontrol_show("RUNNING", -5))):
        assert "is RUNNING" in trigger.bring_forward("77", 60)
    with patch("subprocess.run", return_value=_Proc(1, err="Invalid job id specified")):
        assert "unknown to Slurm" in trigger.bring_forward("77", 60)


def _spec(tmp_path: Path, **opts) -> Path:
    path = tmp_path / "sub" / "submission.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"submission_id": "sub", "supervisor_job_ids": ["70", "77"],
                                "submit_options": {"supervisor_job": True, "supervisor_interval": 1800, **opts}}))
    return path


def test_request_pass_brings_the_last_supervisor_forward_under_the_lock(tmp_path: Path) -> None:
    spec_path = _spec(tmp_path, supervisor_trigger_delay=30)
    seen = {}

    def fake(job, delay):
        seen.update(job=job, delay=delay, locked=(spec_path.parent / LOCK_NAME).exists())
        return "ok"

    with patch.object(trigger, "bring_forward", side_effect=fake):
        assert trigger.request_pass(spec_path) == "ok"
    assert seen == {"job": "77", "delay": 30, "locked": True}
    assert (spec_path.parent / MARKER).exists() and not (spec_path.parent / LOCK_NAME).exists()


def test_request_pass_only_leaves_a_marker_while_a_supervisor_runs(tmp_path: Path) -> None:
    spec_path = _spec(tmp_path)
    (spec_path.parent / LOCK_NAME).write_text(json.dumps(
        {"host": "elsewhere", "pid": 1, "purpose": "supervisor", "since": time.time(), "token": "x"}))
    with patch.object(trigger, "bring_forward", side_effect=AssertionError("schedule changed without the lock")):
        assert "marker left for the running supervisor" in trigger.request_pass(spec_path)
    assert (spec_path.parent / MARKER).exists()


def test_unsupervised_submissions_are_left_alone(tmp_path: Path) -> None:
    spec_path = _spec(tmp_path, supervisor_job=False)
    assert trigger.request_pass(spec_path) == "not supervised"
    assert not (spec_path.parent / MARKER).exists()


def test_a_pass_consumes_older_requests_and_notices_new_ones(tmp_path: Path) -> None:
    marker = tmp_path / MARKER
    marker.touch()
    os.utime(marker, (time.time() - 100, time.time() - 100))
    start = time.time() - 10
    assert not trigger.requested_since(tmp_path, start)
    trigger.consume_marker(tmp_path, start)
    assert not marker.exists()
    marker.touch()                                    # a chain ended during the pass
    assert trigger.requested_since(tmp_path, start)
    trigger.consume_marker(tmp_path, start)
    assert marker.exists(), "a request made during the pass is for the next one"


def test_the_worker_requests_a_pass_for_its_submission(tmp_path: Path) -> None:
    out = tmp_path / "out"
    state_path = out / ".hummel-submit" / "chains" / "c1" / "state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({"output_dir": str(out), "submission_id": "sub"}))
    spec_path = out / ".hummel-submit" / "submissions" / "sub" / "submission.json"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text("{}")
    with patch("hummel_submit.trigger.request_pass", return_value="brought forward") as req:
        worker._request_pass(state_path)
    req.assert_called_once_with(spec_path)
