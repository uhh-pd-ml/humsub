from __future__ import annotations

import copy
import unittest

from hummel_submit.config import DEFAULTS, ConfigError, check_signal_window, slurm_time_seconds


class SlurmTimeTests(unittest.TestCase):
    def test_formats(self) -> None:
        self.assertEqual(slurm_time_seconds("90"), 90 * 60)
        self.assertEqual(slurm_time_seconds("30:00"), 30 * 60)
        self.assertEqual(slurm_time_seconds("04:00:00"), 4 * 3600)
        self.assertEqual(slurm_time_seconds("1-12"), 36 * 3600)
        self.assertEqual(slurm_time_seconds("1-12:30"), 36 * 3600 + 30 * 60)
        self.assertEqual(slurm_time_seconds("1-00:00:05"), 86400 + 5)
        self.assertIsNone(slurm_time_seconds("UNLIMITED"))

    def test_invalid(self) -> None:
        for bad in ("", "soon", "1:2:3:4", "-5"):
            with self.assertRaises(ConfigError, msg=bad):
                slurm_time_seconds(bad)


class SignalWindowTests(unittest.TestCase):
    def cfg(self, **slurm):
        config = copy.deepcopy(DEFAULTS)
        config["slurm"].update(slurm)
        return config

    def test_short_limit_with_continuation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigError, "longer than signal_seconds"):
            check_signal_window(self.cfg(time_limit="00:10:00", signal_seconds=600, max_hops=2), resubmit=True)

    def test_ok_when_limit_exceeds_signal(self) -> None:
        check_signal_window(self.cfg(time_limit="00:10:00", signal_seconds=60, max_hops=2), resubmit=True)
        check_signal_window(self.cfg(time_limit="24:00:00"), resubmit=True)

    def test_not_checked_without_continuation(self) -> None:
        check_signal_window(self.cfg(time_limit="00:05:00", signal_seconds=600, max_hops=2), resubmit=False)
        check_signal_window(self.cfg(time_limit="00:05:00", signal_seconds=600, max_hops=1), resubmit=True)


if __name__ == "__main__":
    unittest.main()
