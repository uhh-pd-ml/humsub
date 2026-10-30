"""Several projects and users side by side: nothing may act on what belongs to another submission or another user."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from hummel_submit import cli, slurm, tick
from hummel_submit.config import ConfigError
from hummel_submit.scratch import stale_scratch_dirs, sweep_stale_scratch
from hummel_submit.submission import CONTROL_LOCK_GENERATION, require_owner, submission_owner


class _Proc:
    def __init__(self, rc: int, out: str = "", err: str = "") -> None:
        self.returncode, self.stdout, self.stderr = rc, out, err


def test_live_jobs_raise_instead_of_answering_nothing() -> None:
    """An empty answer would mean "every job is dead", and stale scratch is deleted on that basis."""
    with patch("subprocess.run", return_value=_Proc(1, err="slurm_load_jobs error: Socket timed out")):
        with pytest.raises(slurm.SlurmError, match="cannot list live Slurm jobs"):
            slurm.live_job_ids(all_users=True)
    with patch("subprocess.run", return_value=_Proc(0, out="11\n12_3\n")) as run:
        assert slurm.live_job_ids(all_users=True) == {"11", "12"}
    assert "-u" not in run.call_args.args[0], "liveness of directories must be judged against every user's jobs"


def test_sweep_deletes_nothing_when_slurm_cannot_be_asked(tmp_path: Path) -> None:
    cfg = {"execution": {"cache_dir": str(tmp_path / "ssd"), "bulk_scratch_dir": str(tmp_path / "bulk")}}
    running = tmp_path / "bulk" / "sub1" / "10-1"
    running.mkdir(parents=True)
    with patch("hummel_submit.slurm.live_job_ids", side_effect=slurm.SlurmError("down")):
        with pytest.raises(slurm.SlurmError):
            sweep_stale_scratch(cfg, None, apply=True)
    assert running.exists()


def test_scratch_of_other_users_is_never_stale_for_me(tmp_path: Path) -> None:
    dead = tmp_path / "bulk" / "subX" / "10-1"
    dead.mkdir(parents=True)
    assert stale_scratch_dirs([tmp_path / "bulk"], set()) == [dead]
    assert stale_scratch_dirs([tmp_path / "bulk"], set(), owner_uid=os.getuid() + 1) == []


def _spec(tmp_path: Path, **fields) -> Path:
    path = tmp_path / "sub1" / "submission.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"submission_id": "sub1", **fields}))
    return path


def test_only_the_owner_may_change_a_submission(tmp_path: Path) -> None:
    mine = _spec(tmp_path, uid=os.getuid(), user="me")
    require_owner(mine, json.loads(mine.read_text()), "resume")
    theirs = _spec(tmp_path, uid=os.getuid() + 1, user="colleague")
    with pytest.raises(PermissionError, match="belongs to colleague; only colleague can resume it"):
        require_owner(theirs, json.loads(theirs.read_text()), "resume")


def test_older_submissions_belong_to_the_owner_of_their_directory(tmp_path: Path) -> None:
    old = _spec(tmp_path)
    assert submission_owner(old, json.loads(old.read_text()))[0] == os.getuid()


def test_an_old_supervisor_blocks_driving_its_submission_until_it_ends(tmp_path: Path) -> None:
    spec_path = _spec(tmp_path, supervisor_job_ids=["55"])
    snapshot = spec_path.parent / "supervisor" / "hummel-submit-worker.zip"
    snapshot.parent.mkdir()
    snapshot.write_text("old code")
    with patch("hummel_submit.slurm.live_job_ids", return_value={"55"}):
        with pytest.raises(ConfigError, match="older humsub"):
            tick.ensure_lock_generation(spec_path)
    assert snapshot.exists()
    with patch("hummel_submit.slurm.live_job_ids", return_value=set()):
        tick.ensure_lock_generation(spec_path)
    assert json.loads(spec_path.read_text())["control_lock"] == CONTROL_LOCK_GENERATION
    assert not snapshot.exists(), "the next supervisor is built from the current version"
    with patch("hummel_submit.slurm.live_job_ids", side_effect=AssertionError("not asked again")):
        tick.ensure_lock_generation(spec_path)


# --- gc selection ------------------------------------------------------------------------------------------------

def _groups() -> list:
    return [cli._GcGroup("20261030-100000-aaaa", "run=alpha project=/p"),
            cli._GcGroup("20261030-110000-bbbb", "run=beta project=/p"),
            cli._GcGroup("orphans", "cache directories no submission of this output_dir refers to")]


def test_gc_selection_by_name_prefix_and_run() -> None:
    groups = _groups()
    assert cli._gc_select_named(groups, ["20261030-11"]) == [groups[1]]
    assert cli._gc_select_named(groups, ["alpha", "orphans"]) == [groups[0], groups[2]]
    with pytest.raises(ConfigError, match="ambiguous"):
        cli._gc_select_named(groups, ["20261030"])
    with pytest.raises(ConfigError, match="not among"):
        cli._gc_select_named(groups, ["gamma"])


@pytest.mark.parametrize("answer,picked", [("", []), ("all", [0, 1, 2]), ("1,3", [0, 2]), ("2-3", [1, 2]), (" 3 ", [2])])
def test_gc_asks_which_to_clean(answer, picked) -> None:
    groups = _groups()
    assert cli._gc_ask(groups, lambda prompt: answer) == [groups[i] for i in picked]


@pytest.mark.parametrize("answer", ["4", "x", "1,,"])
def test_gc_deletes_nothing_on_an_unclear_answer(answer) -> None:
    with pytest.raises(ConfigError, match="nothing deleted|cannot read"):
        cli._gc_ask(_groups(), lambda prompt: answer)
