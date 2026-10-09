from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import signal
import threading
import time
from unittest.mock import patch

import pytest

from hummel_submit.chain_runner import hop_timing_env, run_chain_payload
from hummel_submit.config import DEFAULTS, ConfigError, effective_grace_seconds, validate_config
from hummel_submit.manifest import ManifestError, freeze_manifest_workflow, load_manifest
from hummel_submit.state import atomic_write_json, load_state
from hummel_submit.worker import main as worker_main


def _slurm(**kw) -> dict:
    s = copy.deepcopy(DEFAULTS["slurm"])
    s.update({"time_limit": "01:00:00", "signal_seconds": 600, **kw})
    return s


def test_automatic_grace_leaves_about_two_minutes_before_the_limit() -> None:
    assert effective_grace_seconds(_slurm()) == 450
    assert effective_grace_seconds(_slurm(signal_seconds=120)) == 0   # window too small: hard stop at once
    assert effective_grace_seconds(_slurm(grace_seconds=0)) == 0
    assert effective_grace_seconds(_slurm(grace_seconds=200)) == 200


def test_explicit_grace_must_leave_room_for_the_hard_stop() -> None:
    cfg = copy.deepcopy(DEFAULTS)
    cfg["slurm"]["grace_seconds"] = 560
    with pytest.raises(ConfigError, match="grace_seconds"):
        validate_config(cfg, require_command=False)
    cfg["slurm"]["grace_seconds"] = 500
    validate_config(cfg, require_command=False)
    cfg["slurm"]["grace_seconds"] = -2
    with pytest.raises(ConfigError):
        validate_config(cfg, require_command=False)


def test_hop_env_announces_soft_and_hard_stop_times() -> None:
    state = {"config": {"slurm": _slurm(max_hops=4)}, "resubmit": True}
    env = hop_timing_env(state, now=1000.0)
    assert float(env["HUMSUB_SOFT_STOP_AT"]) == 1000 + 3000
    assert float(env["HUMSUB_HARD_STOP_AT"]) == 1000 + 3000 + 450
    single = {"config": {"slurm": _slurm(max_hops=1)}, "resubmit": True}
    assert "HUMSUB_SOFT_STOP_AT" not in hop_timing_env(single, now=1.0)  # no continuation -> no signal -> no soft stop


def _chain_state(tmp_path: Path, script: str, grace: int) -> tuple[dict, Path]:
    payload = tmp_path / "payload.sh"
    payload.write_text(script)
    state_path = tmp_path / "chain" / "state.json"
    state = {
        "chain_id": "c", "run_name": "r", "payload_script": str(payload), "payload_cwd": str(tmp_path), "resubmit": True,
        "config": {"slurm": _slurm(max_hops=3, signal_seconds=300, grace_seconds=grace)},
    }
    state_path.parent.mkdir()
    atomic_write_json(state_path, state)
    return state, state_path


def _usr1_after(delay: float) -> threading.Timer:
    timer = threading.Timer(delay, lambda: os.kill(os.getpid(), signal.SIGUSR1))
    timer.start()
    return timer


NEARLY_DONE = 'while [ ! -e "$HUMSUB_SOFT_STOP_FILE" ]; do sleep 0.1; done\nsleep 0.5\nexit 0\n'
IGNORES_NOTICE = "sleep 60\n"


def test_payload_that_is_nearly_done_finishes_inside_the_grace_window(tmp_path: Path) -> None:
    state, state_path = _chain_state(tmp_path, NEARLY_DONE, grace=20)
    timer = _usr1_after(0.5)
    start = time.time()
    rc, timed_out = run_chain_payload(state, state_path, 0)
    timer.cancel()
    assert (rc, timed_out) == (0, True)
    assert time.time() - start < 10, "no SIGTERM was needed"
    assert (state_path.parent / "soft-stop-0").exists() and (state_path.parent / "continue-0").exists()


def test_payload_that_ignores_the_notice_gets_sigterm_after_the_grace(tmp_path: Path) -> None:
    state, state_path = _chain_state(tmp_path, IGNORES_NOTICE, grace=2)
    timer = _usr1_after(0.5)
    start = time.time()
    rc, timed_out = run_chain_payload(state, state_path, 0)
    timer.cancel()
    assert timed_out and rc != 0
    assert 2 <= time.time() - start < 15


def test_grace_zero_is_the_old_hard_stop(tmp_path: Path) -> None:
    state, state_path = _chain_state(tmp_path, IGNORES_NOTICE, grace=0)
    timer = _usr1_after(0.5)
    start = time.time()
    rc, timed_out = run_chain_payload(state, state_path, 0)
    timer.cancel()
    assert timed_out and rc != 0 and time.time() - start < 8
    assert not (state_path.parent / "soft-stop-0").exists()


def test_exit_zero_after_the_notice_completes_the_chain(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    atomic_write_json(state_path, {
        "chain_id": "c", "run_name": "r", "jobs": ["1"], "status": "queued", "resubmit": True,
        "config": {"slurm": {"max_hops": 3}},
    })
    with patch.dict(os.environ, {"SLURM_JOB_ID": "2"}), \
         patch("hummel_submit.worker.submit", return_value="3"), \
         patch("hummel_submit.worker.cancel_jobs") as cancel_mock, \
         patch("hummel_submit.worker.run_chain_payload", return_value=(0, True)):
        assert worker_main([str(state_path), "0"]) == 0
    assert load_state(state_path)["status"] == "completed"
    cancel_mock.assert_called_once_with(["3"])  # the follower is not needed any more


def _manifest(tmp_path: Path, branch: dict) -> Path:
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"schema": 1, "branches": [{"id": 0, "outputs": ["/abs/result.json"], **branch}]}))
    return path


def test_extra_outputs_are_parsed_and_separate_from_required_outputs(tmp_path: Path) -> None:
    m = load_manifest(_manifest(tmp_path, {"extra_outputs": ["/abs/ckpt/last.ckpt", "/abs/debug"]}))
    assert m["branches"][0]["outputs"] == ["/abs/result.json"]
    assert m["branches"][0]["extra_outputs"] == ["/abs/ckpt/last.ckpt", "/abs/debug"]
    assert "extra_outputs" not in load_manifest(_manifest(tmp_path, {}))["branches"][0]


@pytest.mark.parametrize("bad", [["relative/path"], [""], "notalist", ["/abs/result.json"]])
def test_bad_extra_outputs_are_rejected(tmp_path: Path, bad) -> None:
    with pytest.raises(ManifestError):
        load_manifest(_manifest(tmp_path, {"extra_outputs": bad}))


def test_extra_outputs_reach_the_branch_context_and_survive_a_resume_subset(tmp_path: Path) -> None:
    spec_path = tmp_path / "sub" / "submission.json"
    atomic_write_json(spec_path, {"submission_id": "s", "run_name": "r", "run_dir": str(tmp_path / "run")})
    payload = tmp_path / "p.sh"
    payload.write_text("#!/bin/sh\n")
    freeze_manifest_workflow(
        spec_path, manifest_source=_manifest(tmp_path, {"extra_outputs": ["/abs/ckpt"]}), payload_source=payload, stages={}
    )
    ctx = json.loads((spec_path.parent / "manifest-workflow" / "branches" / "000000.json").read_text())
    assert ctx["extra_outputs"] == ["/abs/ckpt"] and ctx["outputs"] == ["/abs/result.json"]
