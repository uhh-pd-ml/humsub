from __future__ import annotations

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
                    "gpus": 1,
                    "time_limit": "04:00:00",
                    "signal_seconds": 600,
                    "mail": "",
                    "reservation": "",
                    "extra_args": ["--cpus-per-task=8", "--exclude=g002"],
                }
            },
        }
        args = base_sbatch_args(state)
        self.assertIn("--cpus-per-task=8", args)
        self.assertIn("--exclude=g002", args)
        self.assertIn("--gpus=1", args)
        self.assertNotIn("--gpus-per-node=1", args)
        self.assertIn("--export=NONE", args)

    def test_cpu_single_hop_omits_gpu_and_continuation_signal(self) -> None:
        state = {
            "project_dir": "/project",
            "output_dir": "/output",
            "resubmit": False,
            "config": {
                "slurm": {
                    "job_name": "job",
                    "account": "acct_std",
                    "partition": "std",
                    "nodes": 1,
                    "gpus": 0,
                    "time_limit": "02:00:00",
                    "signal_seconds": 600,
                    "max_hops": 1,
                    "mail": "",
                    "reservation": "",
                    "extra_args": ["--ntasks=1", "--cpus-per-task=8"],
                }
            },
        }
        args = base_sbatch_args(state)
        self.assertIn("--partition=std", args)
        self.assertIn("--ntasks=1", args)
        self.assertIn("--cpus-per-task=8", args)
        self.assertFalse(any(arg.startswith("--gpus=") for arg in args))
        self.assertFalse(any(arg.startswith("--signal=") for arg in args))


if __name__ == "__main__":
    unittest.main()


class QueryJobsTests(unittest.TestCase):
    """A scheduler hiccup must not look like "all jobs are gone" to callers that decide on it."""

    @staticmethod
    def _run(squeue_rc: int, sacct_rc: int):
        import subprocess

        def fake_run(cmd, **kwargs):
            rc = squeue_rc if cmd[0].endswith("squeue") else sacct_rc
            return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="slurm_load_jobs error: timeout")
        return fake_run

    def test_failing_scheduler_is_empty_by_default(self) -> None:
        from unittest.mock import patch
        from hummel_submit.slurm import query_jobs

        with patch("hummel_submit.slurm.subprocess.run", side_effect=self._run(1, 1)):
            self.assertEqual(query_jobs(["10"]), {})

    def test_failing_sacct_raises_when_strict(self) -> None:
        from unittest.mock import patch
        from hummel_submit.slurm import SlurmError, query_jobs

        with patch("hummel_submit.slurm.subprocess.run", side_effect=self._run(1, 1)):
            with self.assertRaises(SlurmError):
                query_jobs(["10"], strict=True)

    def test_expired_job_unknown_to_squeue_is_not_an_error_when_strict(self) -> None:
        from unittest.mock import patch
        from hummel_submit.slurm import query_jobs

        # squeue rejects an expired single job id; sacct answers (here: nothing known).
        with patch("hummel_submit.slurm.subprocess.run", side_effect=self._run(1, 0)):
            self.assertEqual(query_jobs(["10"], strict=True), {})
