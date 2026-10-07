from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
import unittest

from hummel_submit.cleanup import cleanup_submission_cache, inspect_submission_cache, stale_orphan_cache_dirs


class CleanupTests(unittest.TestCase):
    def make_submission(self, root: Path, status: str) -> Path:
        output = root / "output"
        cache = root / "cache"
        sid = "submission-test"
        spec_dir = output / ".hummel-submit" / "submissions" / sid
        chain_dir = output / ".hummel-submit" / "chains" / "chain-test"
        spec_dir.mkdir(parents=True)
        chain_dir.mkdir(parents=True)
        spec = {
            "submission_id": sid,
            "output_dir": str(output),
            "config": {"execution": {"cache_dir": str(cache)}},
            "chain_ids": ["chain-test"],
        }
        spec_path = spec_dir / "submission.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        state = {
            "chain_id": "chain-test",
            "output_dir": str(output),
            "jobs": [],
            "status": status,
            "exit_code": 0 if status == "completed" else 1,
        }
        state_path = chain_dir / "state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        if status != "completed":
            (chain_dir / "done").write_text("failed\n", encoding="utf-8")
        for kind in ("stages", "payload-work"):
            path = cache / kind / sid
            path.mkdir(parents=True)
            (path / "data").write_bytes(b"1234")
        return spec_path

    def test_terminal_success_cleanup_preserves_state(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spec = self.make_submission(root, "completed")
            info = inspect_submission_cache(spec)
            self.assertEqual(info.state, "successful")
            self.assertEqual(len(info.cache_paths), 2)
            cleanup_submission_cache(spec)
            self.assertTrue(spec.exists())
            self.assertTrue((root / "output/.hummel-submit/chains/chain-test/state.json").exists())
            self.assertFalse((root / "cache/stages/submission-test").exists())
            self.assertFalse((root / "cache/payload-work/submission-test").exists())

    def test_failed_cleanup_is_allowed(self):
        with tempfile.TemporaryDirectory() as td:
            spec = self.make_submission(Path(td), "failed-60")
            info = inspect_submission_cache(spec)
            self.assertEqual(info.state, "failed")
            cleanup_submission_cache(spec)
            self.assertFalse(any(entry.path.exists() for entry in info.cache_paths))

    def test_orphan_detection_is_age_guarded(self):
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            orphan = cache / "stages" / "orphan"
            orphan.mkdir(parents=True)
            (orphan / "x").write_text("x", encoding="utf-8")
            old = time.time() - 72 * 3600
            os.utime(orphan, (old, old))
            found = stale_orphan_cache_dirs(cache, set(), older_than_hours=48)
            self.assertEqual([entry.path for entry in found], [orphan])


if __name__ == "__main__":
    unittest.main()
