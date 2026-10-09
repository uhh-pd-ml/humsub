from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

from hummel_submit.state import load_state
from hummel_submit.worker import main as worker_main


def _state(tmp_path: Path, *, budget: int, used: int = 0, hop: int = 0) -> Path:
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "chain_id": "chain", "run_name": "run", "jobs": ["100"], "status": "queued", "resubmit": True,
        "failure_budget": budget, "failures_used": used,
        "config": {"slurm": {"max_hops": 1}},
    }))
    (tmp_path / f"continue-{hop - 1}").touch() if hop else None
    return state_path


def _run(state_path: Path, hop: int, rc: int, timed_out: bool = False):
    with patch.dict(os.environ, {"SLURM_JOB_ID": "101"}), \
         patch("hummel_submit.worker.submit", return_value="102") as submit_mock, \
         patch("hummel_submit.worker.cancel_jobs") as cancel_mock, \
         patch("hummel_submit.worker.run_chain_payload", return_value=(rc, timed_out)):
        result = worker_main([str(state_path), str(hop)])
    return result, submit_mock, cancel_mock


def test_failure_with_budget_keeps_the_follower_and_requests_a_retry(tmp_path: Path) -> None:
    state_path = _state(tmp_path, budget=2)
    result, submit_mock, cancel_mock = _run(state_path, 0, rc=3)
    assert result == 3  # Slurm still shows the hop as FAILED for bookkeeping
    submit_mock.assert_called_once()  # budget makes max_hops 1 + 2 = 3 -> follower queued
    cancel_mock.assert_not_called()
    state = load_state(state_path)
    assert state["status"] == "retrying-after-failure-hop-1"
    assert state["failures_used"] == 1 and state["last_failure_exit_code"] == 3
    assert (tmp_path / "continue-0").exists() and not (tmp_path / "done").exists()


def test_failure_after_budget_exhausted_ends_the_chain(tmp_path: Path) -> None:
    state_path = _state(tmp_path, budget=1, used=1, hop=1)
    result, _, cancel_mock = _run(state_path, 1, rc=3)
    assert result == 3
    state = load_state(state_path)
    assert state["status"] == "failed-3"
    assert (tmp_path / "done").exists()


def test_without_budget_failure_cancels_the_follower_as_before(tmp_path: Path) -> None:
    state_path = _state(tmp_path, budget=0)
    state = json.loads(state_path.read_text())
    state["config"]["slurm"]["max_hops"] = 3
    state_path.write_text(json.dumps(state))
    result, _, cancel_mock = _run(state_path, 0, rc=1)
    assert result == 1
    cancel_mock.assert_called_once_with(["102"])
    assert load_state(state_path)["status"] == "failed-1"


def test_success_after_retry_hop_completes_and_cancels_spare_follower(tmp_path: Path) -> None:
    state_path = _state(tmp_path, budget=2, used=1, hop=1)
    result, _, cancel_mock = _run(state_path, 1, rc=0)
    assert result == 0
    cancel_mock.assert_called_once_with(["102"])
    assert load_state(state_path)["status"] == "completed"
