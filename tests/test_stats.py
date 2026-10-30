from __future__ import annotations

import json
from pathlib import Path

from hummel_submit.stats import format_stats, parse_elapsed, submission_job_ids, parse_sacct, summarize

SACCT = """\
100|das-ntuple|COMPLETED|2026-10-09T10:00:00|2026-10-09T10:05:00|2026-10-09T11:35:00|01:30:00|0:0|8|std
101|das-ntuple|COMPLETED|2026-10-09T10:00:00|2026-10-09T10:10:00|2026-10-09T12:10:00|02:00:00|0:0|8|std
102|das-ntuple|FAILED|2026-10-09T10:00:00|2026-10-09T10:20:00|2026-10-09T10:20:30|00:00:30|60:0|8|std
103|das-ntuple|CANCELLED by 1|2026-10-09T10:00:00|2026-10-09T10:30:00|2026-10-09T10:45:00|00:15:00|0:15|8|std
104|das-ntuple|PENDING|2026-10-09T10:00:00|Unknown|Unknown|00:00:00|0:0|8|std
105|das-ntuple-supervisor|COMPLETED|2026-10-09T10:00:00|2026-10-09T11:00:00|2026-10-09T11:00:10|00:00:10|0:0|1|std
100.batch|batch|COMPLETED|2026-10-09T10:00:00|2026-10-09T10:05:00|2026-10-09T11:35:00|01:30:00|0:0|8|
"""


def test_elapsed_formats() -> None:
    assert parse_elapsed("00:00:30") == 30
    assert parse_elapsed("01:30:00") == 5400
    assert parse_elapsed("1-02:00:00") == 93600


def test_parse_ignores_job_steps() -> None:
    rows = parse_sacct(SACCT)
    assert [r["job"] for r in rows] == ["100", "101", "102", "103", "104", "105"]
    assert rows[3]["state"] == "CANCELLED"


def test_summary_numbers() -> None:
    s = summarize(parse_sacct(SACCT), {"c1": 1, "c2": 2}, supervisor_name="das-ntuple-supervisor")
    assert s["jobs"] == 5 and s["states"] == {"COMPLETED": 2, "FAILED": 1, "CANCELLED": 1, "PENDING": 1}
    assert s["supervisor_jobs"] == 1
    assert s["completed_hours"] == {"min": 1.5, "median": 1.75, "max": 2.0, "total": 3.5}
    assert s["ran_jobs"] == 4
    assert abs(s["core_hours"] - 8 * (5400 + 7200 + 30 + 900) / 3600) < 1e-6
    assert abs(s["wasted_hours"] - (30 + 900) / 3600) < 1e-9          # the failed and the cancelled job
    assert s["peak_concurrency"] == 3                                    # jobs 100, 101 and 103 overlap at 10:30
    assert s["chains"] == 2 and s["chains_with_more_than_one_job"] == 1
    assert s["failed_jobs"] == [{"job": "102", "state": "FAILED", "elapsed_s": 30, "exit": "60:0"}]
    assert s["wall_hours"] == 2.1666666666666665 or abs(s["wall_hours"] - (2 + 5 / 60)) < 1e-6
    assert abs(s["queue_wait_hours"]["max"] - 0.5) < 1e-9


def test_report_lines_mention_the_headline_numbers() -> None:
    text = "\n".join(format_stats(summarize(parse_sacct(SACCT), {"c1": 1, "c2": 2}, "das-ntuple-supervisor")))
    for needle in ("5 (", "2 COMPLETED", "median 1.75 h", "core-hours", "peak running:  3", "chains:        2", "102 FAILED after 30 s"):
        assert needle in text, needle


def test_submission_job_ids_cover_retried_chains_and_supervisors(tmp_path: Path) -> None:
    for chain, jobs in (("cA", ["10", "11"]), ("cB", ["12"]), ("cC", ["13"])):
        d = tmp_path / ".hummel-submit" / "chains" / chain
        d.mkdir(parents=True)
        (d / "state.json").write_text(json.dumps({"jobs": jobs}))
    spec = {"submission_id": "s", "chain_ids": ["cA", "cB", "cC"], "supervisor_job_ids": ["90"]}
    ids, per_chain, chains = submission_job_ids(tmp_path, spec)
    assert ids == ["10", "11", "12", "13", "90"]
    assert per_chain == {"cA": 2, "cB": 1, "cC": 1} and chains == 3
