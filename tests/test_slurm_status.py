from __future__ import annotations

import unittest

from hummel_submit.slurm import parse_sacct_status, parse_squeue_status


class SlurmStatusTests(unittest.TestCase):
    def test_parse_squeue(self) -> None:
        parsed = parse_squeue_status("100|RUNNING\n101|PENDING\n")
        self.assertTrue(parsed["100"].running)
        self.assertTrue(parsed["101"].pending)

    def test_parse_sacct_ignores_steps(self) -> None:
        parsed = parse_sacct_status(
            "100|COMPLETED|0:0\n100.batch|COMPLETED|0:0\n101|FAILED|3:0\n",
            requested={"100", "101"},
        )
        self.assertEqual(set(parsed), {"100", "101"})
        self.assertTrue(parsed["100"].finished)
        self.assertTrue(parsed["101"].failed)
        self.assertEqual(parsed["101"].exit_code, 3)


if __name__ == "__main__":
    unittest.main()
