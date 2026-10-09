from __future__ import annotations

from pathlib import Path

import pytest

from hummel_submit.scratch import (
    effective_scratch_kind,
    hop_timing,
    keep_small_files,
    remove_scratch,
    reset_scratch,
    retry_allowed,
    scratch_dirs,
    stale_scratch_dirs,
)


def _config(tmp_path: Path, scratch: str = "beegfs") -> dict:
    return {"execution": {"cache_dir": str(tmp_path / "ssd"), "bulk_scratch_dir": str(tmp_path / "bulk"), "scratch": scratch}}


def test_scratch_kind_precedence() -> None:
    cfg = {"execution": {"scratch": "beegfs"}}
    assert effective_scratch_kind(cfg) == "beegfs"
    assert effective_scratch_kind(cfg, {"scratch": "ssd"}) == "ssd"
    assert effective_scratch_kind(cfg, {"scratch": "ssd"}, "beegfs") == "beegfs"


def test_bulk_scratch_is_on_beegfs_by_default_and_fast_on_ssd(tmp_path: Path) -> None:
    dirs = scratch_dirs(_config(tmp_path), "sub1", "123", 7, "beegfs")
    assert dirs.bulk == tmp_path / "bulk" / "sub1" / "123-7"
    assert dirs.fast == tmp_path / "ssd" / "payload-work" / "sub1" / "123-7"
    assert len(dirs.all) == 2


def test_ssd_kind_uses_one_directory(tmp_path: Path) -> None:
    dirs = scratch_dirs(_config(tmp_path, "ssd"), "sub1", "123", 7, "ssd")
    assert dirs.bulk == dirs.fast and dirs.all == [dirs.fast]


def test_reset_and_remove(tmp_path: Path) -> None:
    dirs = scratch_dirs(_config(tmp_path), "s", "1", 0, "beegfs")
    reset_scratch(dirs)
    (dirs.bulk / "big.root").write_bytes(b"x" * 100)
    reset_scratch(dirs)
    assert list(dirs.bulk.iterdir()) == []
    remove_scratch(dirs)
    assert not dirs.bulk.exists() and not dirs.fast.exists()


def test_keep_small_files_skips_big_and_binary(tmp_path: Path) -> None:
    dirs = scratch_dirs(_config(tmp_path), "s", "1", 0, "beegfs")
    reset_scratch(dirs)
    (dirs.bulk / "params.json").write_text("{}")
    (dirs.bulk / "ntuple.root").write_bytes(b"x" * 10)  # wrong suffix
    (dirs.fast / "huge.log").write_bytes(b"x" * (2 << 20))  # too big
    (dirs.fast / "run.log").write_text("hello")
    kept = keep_small_files(dirs, tmp_path / "left")
    names = sorted(p.name for p in kept)
    assert names == ["params.json", "run.log"]
    assert all(str(p).startswith(str(tmp_path / "left")) for p in kept)


@pytest.mark.parametrize(
    "failed,elapsed,usable,durations,expected",
    [
        (1, 100, 1000, [100], True),       # plenty of time
        (1, 600, 1000, [500], False),      # 500*1.2 = 600 > 400 left
        (2, 400, 1000, [200, 200], True),  # 240 < 600
        (2, 700, 1000, [350, 350], False),  # needs 420 s, 300 s left: skipped past ~58 %
        (3, 760, 1000, [230, 230, 230], False),  # needs 276 s, 240 s left: skipped past ~72 %
        (4, 10, 1000, [1, 1, 1, 1], False),  # budget (max 3) exhausted
    ],
)
def test_retry_allowed_is_time_aware(failed, elapsed, usable, durations, expected) -> None:
    allowed, reason = retry_allowed(
        failed_attempts=failed, max_retries=3, elapsed=elapsed, usable=usable, attempt_durations=durations
    )
    assert allowed is expected, reason


def test_retry_without_time_information_is_allowed_within_budget() -> None:
    assert retry_allowed(failed_attempts=1, max_retries=2, elapsed=None, usable=None, attempt_durations=[5])[0]
    assert not retry_allowed(failed_attempts=3, max_retries=2, elapsed=None, usable=None, attempt_durations=[5])[0]


def test_hop_timing_from_environment() -> None:
    env = {"HUMSUB_HOP_START": "1000", "HUMSUB_HOP_USABLE_SECONDS": "600"}
    assert hop_timing(env, now=lambda: 1250.0) == (250.0, 600.0)
    assert hop_timing({}, now=lambda: 1.0) == (None, None)


def test_stale_scratch_only_dead_jobs(tmp_path: Path) -> None:
    root = tmp_path / "bulk"
    for name in ("100-1", "101-2", "garbage", "102-x"):
        (root / "subA" / name).mkdir(parents=True)
    (root / "subB" / "100-9").mkdir(parents=True)
    stale = stale_scratch_dirs([root], live_jobs={"101"})
    assert sorted(p.name for p in stale) == ["100-1", "100-9"]
    only_a = stale_scratch_dirs([root], live_jobs={"101"}, submission_ids={"subA"})
    assert [p.name for p in only_a] == ["100-1"]
