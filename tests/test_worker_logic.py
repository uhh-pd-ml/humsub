from __future__ import annotations

import unittest


def should_queue(hop: int, max_hops: int) -> bool:
    return hop + 1 < max_hops


class HopTests(unittest.TestCase):
    def test_max_hops_is_not_off_by_one(self) -> None:
        queued_followers = [hop for hop in range(20) if should_queue(hop, 20)]
        self.assertEqual(queued_followers[-1], 18)
        self.assertEqual(len(queued_followers), 19)
        self.assertFalse(should_queue(19, 20))


if __name__ == "__main__":
    unittest.main()
