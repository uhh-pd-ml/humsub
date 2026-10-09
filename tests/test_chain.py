from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hummel_submit.chain import query_chain
from hummel_submit.slurm import SlurmJobStatus


class ChainStatusTests(unittest.TestCase):
    def _state(self, root: Path, status: str, jobs: list[str] | None = None, **extra) -> Path:
        chain = "chain-test"
        path = root / ".hummel-submit" / "chains" / chain / "state.json"
        path.parent.mkdir(parents=True)
        data = {"chain_id": chain, "status": status, "jobs": jobs or []}
        data.update(extra)
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_completed_chain_is_finished_without_scheduler_query(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._state(root, "completed", ["10"], exit_code=0)
            with patch("hummel_submit.chain.query_jobs") as query:
                status = query_chain(root, "chain-test")
            self.assertEqual(status.state, "finished")
            query.assert_not_called()

    def test_running_inner_job_maps_to_running_chain(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._state(root, "running-hop-1", ["10", "11"])
            with patch(
                "hummel_submit.chain.query_jobs",
                return_value={
                    "10": SlurmJobStatus("10", "RUNNING"),
                    "11": SlurmJobStatus("11", "PENDING"),
                },
            ):
                status = query_chain(root, "chain-test")
            self.assertEqual(status.state, "running")

    def test_failed_chain_maps_to_failed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._state(root, "failed-17", ["10"], exit_code=17)
            status = query_chain(root, "chain-test")
            self.assertEqual(status.state, "failed")
            self.assertEqual(status.code, 17)


if __name__ == "__main__":
    unittest.main()


def test_lane_dependency_follows_the_chain_n_back(tmp_path):
    import json
    from hummel_submit.chain import lane_dependency
    from hummel_submit.state import state_path

    out = tmp_path / "out"
    for chain_id, jobs in (("c0", ["10"]), ("c1", ["11"]), ("c2", ["12", "20"])):
        path = state_path(out, chain_id)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"jobs": jobs}))
    spec = {"output_dir": str(out), "max_concurrent": 2, "chain_ids": ["c0", "c1"]}
    assert lane_dependency(spec) == "10"                      # third chain waits for the first
    spec["chain_ids"] = ["c0", "c1", "c2"]
    assert lane_dependency(spec) == "11"                      # fourth waits for the second
    assert lane_dependency({**spec, "chain_ids": ["c0"]}) is None   # fewer chains than lanes
    assert lane_dependency({**spec, "max_concurrent": 0}) is None   # no cap
    spec["chain_ids"] = ["c1", "c2"]
    assert lane_dependency({**spec, "max_concurrent": 1}) == "20"  # most recent job of the previous chain
