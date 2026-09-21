from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from hummel_submit.runner import detect_ngpu, newest_checkpoint, render_auto_args


class RunnerTests(unittest.TestCase):
    def test_auto_arg_is_suppressed_by_user_key(self) -> None:
        result = render_auto_args(
            ["--run={RUN}", "--strategy={STRATEGY}", "plain"],
            ["--strategy=manual"],
            {"RUN": "r1", "STRATEGY": "ddp"},
        )
        self.assertEqual(result, ["--run=r1", "plain"])

    def test_checkpoint_search_is_scoped_to_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = root / "runs" / "r1"
            other = root / "runs" / "r2"
            (run / "checkpoints").mkdir(parents=True)
            (other / "checkpoints").mkdir(parents=True)
            a = run / "checkpoints" / "a.ckpt"
            b = other / "checkpoints" / "b.ckpt"
            a.write_text("a")
            b.write_text("b")
            os.utime(b, (time.time() + 100, time.time() + 100))
            state = {"run_dir": str(run), "config": {"execution": {"checkpoint_glob": "checkpoints/*.ckpt"}}}
            self.assertEqual(newest_checkpoint(state), a)

    def test_detect_ngpu_does_not_probe_hardware_in_cpu_slurm_job(self) -> None:
        with patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}, clear=False):
            os.environ.pop("SLURM_GPUS_ON_NODE", None)
            with patch("hummel_submit.runner.shutil.which") as which:
                self.assertEqual(detect_ngpu(), 0)
                which.assert_not_called()

    def test_detect_ngpu_uses_slurm_allocation(self) -> None:
        with patch.dict(
            os.environ,
            {"SLURM_JOB_ID": "12345", "SLURM_GPUS_ON_NODE": "2"},
            clear=False,
        ):
            with patch("hummel_submit.runner.shutil.which") as which:
                self.assertEqual(detect_ngpu(), 2)
                which.assert_not_called()


if __name__ == "__main__":
    unittest.main()
