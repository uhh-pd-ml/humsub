from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import time
from unittest.mock import patch

import pytest

from hummel_submit import locks, supervisor
from hummel_submit.locks import LOCK_NAME, ControlLock, LockBusy, held_here


def _foreign(tmp_path: Path, *, age: float = 0, **owner) -> Path:
    """A lock file of another owner (another host and pid unless given), last refreshed *age* seconds ago."""
    path = tmp_path / LOCK_NAME
    data = {"job_id": None, "host": "elsewhere", "pid": 4242, "purpose": "humsub tick", "since": time.time() - age,
            "token": "foreign"}
    data.update(owner)
    path.write_text(json.dumps(data))
    os.utime(path, (time.time() - age, time.time() - age))
    return path


def test_acquire_release_and_reentrance(tmp_path: Path) -> None:
    lock = ControlLock(tmp_path, "test")
    assert lock.try_acquire() and (tmp_path / LOCK_NAME).exists()
    assert held_here(tmp_path / "submission.json")
    nested = ControlLock(tmp_path, "nested")
    assert nested.try_acquire()                     # same process: re-entrant
    nested.release()
    assert (tmp_path / LOCK_NAME).exists(), "the outer holder still holds it"
    lock.release()
    assert not (tmp_path / LOCK_NAME).exists() and not held_here(tmp_path / "submission.json")


def test_a_live_foreign_lock_is_respected(tmp_path: Path) -> None:
    path = _foreign(tmp_path)
    lock = ControlLock(tmp_path, "test")
    assert not lock.try_acquire()
    with pytest.raises(LockBusy, match="humsub tick"):
        lock.acquire()
    assert json.loads(path.read_text())["token"] == "foreign"


def test_a_lock_without_refresh_for_the_lease_is_stale(tmp_path: Path) -> None:
    _foreign(tmp_path, age=locks.LEASE_SECONDS + 5)
    lock = ControlLock(tmp_path, "test")
    assert lock.try_acquire()
    assert json.loads((tmp_path / LOCK_NAME).read_text())["token"] == lock.owner["token"]
    assert [p.name for p in tmp_path.iterdir()] == [LOCK_NAME], "the stale lock is moved aside and removed"
    lock.release()


def test_a_lock_of_a_dead_process_on_this_host_is_stale(tmp_path: Path) -> None:
    _foreign(tmp_path, host=socket.gethostname(), pid=2 ** 22 + 12345)
    assert ControlLock(tmp_path, "test").try_acquire()


def test_a_lock_of_a_live_process_on_this_host_is_kept(tmp_path: Path) -> None:
    _foreign(tmp_path, host=socket.gethostname(), pid=os.getppid())
    assert not ControlLock(tmp_path, "test").try_acquire()


def test_a_slurm_owner_counts_as_long_as_slurm_lists_its_job(tmp_path: Path) -> None:
    _foreign(tmp_path, job_id="77", purpose="supervisor")
    with patch.object(locks, "slurm_job_alive", return_value=True):
        assert not ControlLock(tmp_path, "test").try_acquire()
    with patch.object(locks, "slurm_job_alive", return_value=False):   # crashed / killed / node failure
        assert ControlLock(tmp_path, "test").try_acquire()


def test_squeue_errors_never_make_a_lock_stale() -> None:
    class Proc:
        def __init__(self, rc, out="", err=""):
            self.returncode, self.stdout, self.stderr = rc, out, err

    with patch("subprocess.run", return_value=Proc(1, err="slurm_load_jobs error: Unable to contact slurm controller")):
        assert locks.slurm_job_alive("77") is True
    with patch("subprocess.run", return_value=Proc(1, err="slurm_load_jobs error: Invalid job id specified")):
        assert locks.slurm_job_alive("77") is False
    with patch("subprocess.run", return_value=Proc(0, out="")):
        assert locks.slurm_job_alive("77") is False
    with patch("subprocess.run", return_value=Proc(0, out="RUNNING\n")):
        assert locks.slurm_job_alive("77") is True


def test_breaking_puts_back_a_lock_that_was_taken_over_meanwhile(tmp_path: Path) -> None:
    """Between judging a lock stale and moving it aside, it may have been released and taken by a live owner."""
    _foreign(tmp_path, token="fresh-owner")
    lock = ControlLock(tmp_path, "test")
    lock._break({"token": "the-stale-one"})
    assert json.loads((tmp_path / LOCK_NAME).read_text())["token"] == "fresh-owner"
    assert [p.name for p in tmp_path.iterdir()] == [LOCK_NAME]


def test_release_never_removes_someone_elses_lock(tmp_path: Path) -> None:
    lock = ControlLock(tmp_path, "test")
    assert lock.try_acquire()
    _foreign(tmp_path)                                # e.g. ours was broken as stale and someone else took it
    lock.release()
    assert json.loads((tmp_path / LOCK_NAME).read_text())["token"] == "foreign"


# --- the supervisor and the lock ---------------------------------------------------------------------------------

def test_a_second_supervisor_ends_without_a_successor(tmp_path: Path) -> None:
    _foreign(tmp_path, job_id="77", purpose=supervisor.SUPERVISOR_PURPOSE)
    with patch.object(locks, "slurm_job_alive", return_value=True):
        assert supervisor.acquire_control(tmp_path / "submission.json", wait=0) is None


def test_the_supervisor_waits_for_a_tick_and_then_gives_up(tmp_path: Path) -> None:
    _foreign(tmp_path, purpose="humsub resume")
    with pytest.raises(LockBusy, match="humsub resume"):
        supervisor.acquire_control(tmp_path / "submission.json", wait=0.2, poll=0.05)


def test_supervisors_are_only_queued_under_the_lock(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="control lock"):
        supervisor.submit_supervisor(tmp_path / "submission.json", round_no=1)


def test_supervisor_releases_the_lock_when_its_pass_fails(tmp_path: Path) -> None:
    spec_path = tmp_path / "submission.json"
    spec_path.write_text(json.dumps({"output_dir": str(tmp_path)}))
    with patch.object(supervisor, "_supervise", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            supervisor.main([str(spec_path), "1:1"])
    assert not (tmp_path / LOCK_NAME).exists()


# --- the short-held mutex ------------------------------------------------------------------------------------------

def _bump(state_path: str, times: int) -> None:
    from hummel_submit.state import _locked_update

    for _ in range(times):
        _locked_update(Path(state_path), lambda data: data.update(n=data["n"] + 1))


def test_mutex_loses_no_update_between_processes(tmp_path: Path) -> None:
    import multiprocessing

    state_path = tmp_path / "chain" / "state.json"
    state_path.parent.mkdir()
    state_path.write_text(json.dumps({"n": 0}))
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_bump, args=(str(state_path), 25)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    assert json.loads(state_path.read_text())["n"] == 100
    assert not (state_path.parent / ".state.mutex").exists()


def test_mutex_waits_for_a_live_holder_and_breaks_a_stale_one(tmp_path: Path) -> None:
    path = tmp_path / ".x.mutex"
    path.write_text(json.dumps({"token": "other", "purpose": "mutex", "host": "elsewhere", "pid": 1}))
    with pytest.raises(TimeoutError, match="held for more than"):
        with locks.mutex(path, timeout=0.3):
            pass
    old = time.time() - locks.MUTEX_STALE_SECONDS - 5
    os.utime(path, (old, old))                       # its holder was killed inside the critical section
    with locks.mutex(path, timeout=5):
        assert json.loads(path.read_text())["token"] != "other"
    assert not path.exists()
