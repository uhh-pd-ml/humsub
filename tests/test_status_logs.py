from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hummel_submit.cli import cmd_status
from hummel_submit.slurm import SlurmJobStatus, base_sbatch_args, slurm_log_path


class _Args:
    def __init__(self, chain: str):
        self.chain = chain


class StatusLogTests(unittest.TestCase):
    def _state(self, root: Path) -> Path:
        chain = "chain-test"
        path = root / ".hummel-submit" / "chains" / chain / "state.json"
        path.parent.mkdir(parents=True)
        data = {
            "chain_id": chain,
            "run_name": "run-test",
            "status": "running-hop-1",
            "jobs": ["100", "101"],
            "current_job_id": "100",
            "next_job_id": "101",
            "output_dir": str(root),
            "config": {"slurm": {"job_name": "analysis"}},
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_log_path_matches_sbatch_pattern(self) -> None:
        state = {
            "output_dir": "/tmp/output",
            "project_dir": "/tmp/project",
            "resubmit": False,
            "config": {
                "slurm": {
                    "job_name": "analysis",
                    "account": "a",
                    "partition": "p",
                    "nodes": 1,
                    "gpus": 0,
                    "time_limit": "1:00:00",
                    "signal_seconds": 60,
                    "mail": "",
                    "reservation": "",
                    "max_hops": 1,
                    "extra_args": [],
                }
            },
        }
        self.assertEqual(
            slurm_log_path(state, "123"),
            Path("/tmp/output/logs/analysis_123.log"),
        )
        self.assertIn("--output=/tmp/output/logs/%x_%j.log", base_sbatch_args(state))

    def test_status_lists_all_known_job_logs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_path = self._state(root)
            current_log = root / "logs" / "analysis_100.log"
            current_log.parent.mkdir()
            current_log.write_text("running\n", encoding="utf-8")

            with (
                patch("hummel_submit.cli._find_chain", return_value=state_path),
                patch("hummel_submit.cli.query_chain") as query_chain,
                patch("hummel_submit.cli.queue_status", return_value="100 RUNNING 0:12 n001\n101 PENDING 0:00 (Dependency)"),
                patch("builtins.print") as printer,
            ):
                query_chain.return_value = type("Status", (), {"state": "running"})()
                rc = cmd_status(_Args("chain-test"))

            self.assertEqual(rc, 0)
            lines = [str(call.args[0]) for call in printer.call_args_list if call.args]
            self.assertIn("logs:", lines)
            self.assertTrue(any("100 current" in line and "analysis_100.log" in line and "(exists)" in line for line in lines))
            self.assertTrue(any("101 next" in line and "analysis_101.log" in line and "(not created yet)" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
