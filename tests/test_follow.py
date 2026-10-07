from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hummel_submit.follow import follow_chain
from hummel_submit.slurm import SlurmJobStatus


class FollowTests(unittest.TestCase):
    def _write_state(self, path: Path, **updates) -> None:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = {
                "chain_id": "chain-test",
                "output_dir": str(path.parents[3]),
                "run_name": "run-test",
                "status": "running-hop-1",
                "jobs": ["100", "101"],
                "current_job_id": "100",
                "next_job_id": "101",
                "config": {"slurm": {"job_name": "analysis"}},
            }
        data.update(updates)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_follows_current_log_then_successor_from_start(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_path = root / ".hummel-submit" / "chains" / "chain-test" / "state.json"
            self._write_state(state_path)
            logs = root / "logs"
            logs.mkdir()
            first = logs / "analysis_100.log"
            second = logs / "analysis_101.log"
            first.write_text("".join(f"old-{i}\n" for i in range(1, 13)), encoding="utf-8")

            phase = {"value": 0}

            def fake_sleep(_seconds: float) -> None:
                phase["value"] += 1
                if phase["value"] == 1:
                    with first.open("a", encoding="utf-8") as handle:
                        handle.write("old-final\n")
                    second.write_text("new-start\n", encoding="utf-8")
                    self._write_state(
                        state_path,
                        status="running-hop-2",
                        current_job_id="101",
                        next_job_id=None,
                    )
                elif phase["value"] == 2:
                    with second.open("a", encoding="utf-8") as handle:
                        handle.write("new-final\n")
                    self._write_state(state_path, status="completed", exit_code=0)

            def fake_query_jobs(job_ids):
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state["status"] == "completed":
                    return {str(job_id): SlurmJobStatus(str(job_id), "COMPLETED", 0) for job_id in job_ids}
                return {str(job_id): SlurmJobStatus(str(job_id), "RUNNING") for job_id in job_ids}

            out = io.StringIO()
            err = io.StringIO()
            finished = type("ChainStatus", (), {"state": "finished", "code": 0})()
            with (
                patch("hummel_submit.follow.time.sleep", side_effect=fake_sleep),
                patch("hummel_submit.follow.query_jobs", side_effect=fake_query_jobs),
                patch("hummel_submit.follow.query_chain", return_value=finished),
            ):
                rc = follow_chain(state_path, initial_lines=10, poll_interval=0.01, out=out, err=err)

            self.assertEqual(rc, 0)
            text = out.getvalue()
            self.assertNotIn("old-1\n", text)
            self.assertNotIn("old-2\n", text)
            self.assertIn("old-3\n", text)
            self.assertIn("old-final\n", text)
            self.assertIn("new-start\n", text)
            self.assertIn("new-final\n", text)
            self.assertIn("job 100", err.getvalue())
            self.assertIn("job 101", err.getvalue())

    def test_failed_chain_returns_failure_code(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_path = root / ".hummel-submit" / "chains" / "chain-test" / "state.json"
            self._write_state(state_path, status="failed-17", exit_code=17, jobs=["100"], current_job_id="100")
            logs = root / "logs"
            logs.mkdir()
            (logs / "analysis_100.log").write_text("boom\n", encoding="utf-8")

            failed = type("ChainStatus", (), {"state": "failed", "code": 17})()
            with (
                patch("hummel_submit.follow.time.sleep", return_value=None),
                patch("hummel_submit.follow.query_jobs", return_value={"100": SlurmJobStatus("100", "FAILED", 17)}),
                patch("hummel_submit.follow.query_chain", return_value=failed),
            ):
                rc = follow_chain(state_path, poll_interval=0.01, out=io.StringIO(), err=io.StringIO())
            self.assertEqual(rc, 17)


if __name__ == "__main__":
    unittest.main()
