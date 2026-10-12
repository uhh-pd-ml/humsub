from __future__ import annotations

from pathlib import Path

import pytest

from hummel_submit.config import ConfigError
from hummel_submit.quota import check_fast_quota, concurrent_chains, parse_rrz_quota

GIB = 1 << 30
RRZ = """
/beegfs       4 %  (  3757.04 GiB / 100000.00 GiB)
/home        16 %  (     1.59 GiB /     10.00 GiB)
/nfs/ssd2.0   7 %  (     6.85 GiB /    100.00 GiB)
/usw          4 %  (     1.89 GiB /     50.00 GiB)
"""


def test_parse_rrz_quota_picks_the_right_filesystem() -> None:
    used, total = parse_rrz_quota(RRZ, Path("/nfs/ssd2.0/uu/x/y/.hummel-submit/cache"))
    assert abs(used / GIB - 6.85) < 0.01 and total == 100 * GIB
    assert parse_rrz_quota(RRZ, Path("/elsewhere")) is None


def test_concurrency_is_the_smallest_limit() -> None:
    assert concurrent_chains(chains=900, max_concurrent=0, wait=False, parallel_jobs=0) == 900
    assert concurrent_chains(chains=900, max_concurrent=40, wait=False, parallel_jobs=0) == 40
    assert concurrent_chains(chains=900, max_concurrent=0, wait=True, parallel_jobs=25) == 25
    assert concurrent_chains(chains=900, max_concurrent=0, wait=False, parallel_jobs=25) == 900  # needs --wait


def _config() -> dict:
    return {"execution": {"cache_dir": "/nfs/ssd2.0/x", "fast_bytes_per_job": 250_000_000}}


def test_uncapped_submission_of_hundreds_of_chains_is_refused() -> None:
    """The 2026-10-09 incident: ~540 simultaneous chains filled a 100 GiB SSD within minutes."""
    with pytest.raises(ConfigError, match="--max-concurrent"):
        check_fast_quota(
            _config(), branches=2261, tasks_per_job=4, max_concurrent=0, wait=False, parallel_jobs=0,
            scratch_kind="beegfs", usage=(7 * GIB, 100 * GIB),
        )


def test_capped_submission_passes_and_reports() -> None:
    line = check_fast_quota(
        _config(), branches=2261, tasks_per_job=4, max_concurrent=50, wait=False, parallel_jobs=0,
        scratch_kind="beegfs", usage=(7 * GIB, 100 * GIB),
    )
    assert "50 concurrent" in line


def test_ssd_scratch_counts_towards_the_footprint() -> None:
    with pytest.raises(ConfigError):
        check_fast_quota(
            _config(), branches=400, tasks_per_job=1, max_concurrent=100, wait=False, parallel_jobs=0,
            scratch_kind="ssd", scratch_bytes_per_branch=700_000_000, usage=(7 * GIB, 100 * GIB),
        )
    check_fast_quota(
        _config(), branches=400, tasks_per_job=1, max_concurrent=100, wait=False, parallel_jobs=0,
        scratch_kind="beegfs", scratch_bytes_per_branch=700_000_000, usage=(7 * GIB, 100 * GIB),
    )


def test_unknown_quota_is_not_fatal() -> None:
    line = check_fast_quota(
        _config(), branches=10, tasks_per_job=1, max_concurrent=0, wait=False, parallel_jobs=0,
        scratch_kind="beegfs", usage=None,
    )
    assert "not checked" in line
