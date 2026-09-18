from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hummel_submit.state import load_state
from hummel_submit.worker import main as worker_main, should_queue_follower


class HopTests(unittest.TestCase):
    def test_max_hops_is_not_off_by_one(self) -> None:
        queued_followers = [hop for hop in range(20) if should_queue_follower(hop, 20)]
        self.assertEqual(queued_followers[-1], 18)
        self.assertEqual(len(queued_followers), 19)
        self.assertFalse(should_queue_follower(19, 20))

    def test_follower_without_continuation_marker_does_not_run_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_path = root / "state.json"
            state = {
                "chain_id": "chain",
                "run_name": "run",
                "jobs": ["100"],
                "status": "queued",
                "config": {"slurm": {"max_hops": 20}},
            }
            state_path.write_text(json.dumps(state), encoding="utf-8")
            with patch.dict(os.environ, {"SLURM_JOB_ID": "101"}, clear=False), \
                 patch("hummel_submit.worker.submit") as submit_mock, \
                 patch("hummel_submit.worker.run_payload") as run_mock:
                rc = worker_main([str(state_path), "1"])
            self.assertEqual(rc, 0)
            submit_mock.assert_not_called()
            run_mock.assert_not_called()
            final = load_state(state_path)
            self.assertEqual(final["status"], "stopped-no-continuation-marker")
            self.assertTrue((root / "done").exists())


if __name__ == "__main__":
    unittest.main()
