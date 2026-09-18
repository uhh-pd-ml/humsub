from __future__ import annotations

from pathlib import Path
import unittest

from hummel_submit.slurm import base_sbatch_args


class SlurmTests(unittest.TestCase):
    def test_extra_args_are_part_of_frozen_base_for_every_hop(self) -> None:
        state = {
            "project_dir": "/project",
            "output_dir": "/output",
            "config": {
                "slurm": {
                    "job_name": "job",
                    "account": "acct",
                    "partition": "gpu",
                    "nodes": 1,
                    "gpus_per_node": 1,
                    "time_limit": "04:00:00",
                    "signal_seconds": 600,
                    "mail": "",
                    "reservation": "",
                    "extra_args": ["--cpus-per-task=8", "--mem=64G", "--exclude=g002"],
                }
            },
        }
        args = base_sbatch_args(state)
        self.assertIn("--cpus-per-task=8", args)
        self.assertIn("--mem=64G", args)
        self.assertIn("--exclude=g002", args)
        self.assertIn("--export=NONE", args)


if __name__ == "__main__":
    unittest.main()
