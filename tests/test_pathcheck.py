from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hummel_submit.pathcheck import PathCheckError, check_compute_writable, check_payload_args


def test_tmp_base() -> str:
    """Use a neutral filesystem root that pathcheck does not classify as Hummel storage."""
    return os.environ.get("HUMSUB_TEST_TMPDIR", "/var/tmp")


class PathCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(dir=test_tmp_base())
        self.root = Path(self.tmp.name)
        self.home = self.root / "home"
        self.usw = self.root / "usw"
        self.beegfs = self.root / "beegfs"
        self.ssd = self.root / "ssd"
        for path in (self.home, self.usw, self.beegfs, self.ssd):
            path.mkdir()
        self.env = {
            "HOME": str(self.home),
            "USW": str(self.usw),
            "BEEGFS": str(self.beegfs),
            "SSD": str(self.ssd),
        }

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_home_output_is_rejected_even_if_frontend_can_write_it(self) -> None:
        with patch.dict(os.environ, self.env, clear=False), self.assertRaisesRegex(PathCheckError, "read-only"):
            check_compute_writable(self.home / "results", "output")

    def test_beegfs_output_is_accepted(self) -> None:
        with patch.dict(os.environ, self.env, clear=False):
            result = check_compute_writable(self.beegfs / "jobs", "output", must_be_shared=True)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.storage, "beegfs")

    def test_symlink_into_ssd_is_classified_by_target(self) -> None:
        project = self.home / "project"
        project.mkdir()
        link = project / "scratch"
        link.symlink_to(self.ssd, target_is_directory=True)
        with patch.dict(os.environ, self.env, clear=False):
            result = check_compute_writable(link / "cache", "cache")
        self.assertEqual(result.storage, "ssd")

    def test_existing_home_path_can_be_an_input(self) -> None:
        project = self.home / "project"
        project.mkdir()
        input_file = self.home / "data.root"
        input_file.write_text("data", encoding="utf-8")
        with patch.dict(os.environ, self.env, clear=False):
            checks = check_payload_args([f"--input={input_file}"], project)
        self.assertEqual(checks[0].status, "ok")
        self.assertIn("treated as an input", checks[0].detail)

    def test_output_like_argument_under_home_is_rejected(self) -> None:
        project = self.home / "project"
        project.mkdir()
        out = self.home / "existing-output"
        out.mkdir()
        with patch.dict(os.environ, self.env, clear=False), self.assertRaisesRegex(PathCheckError, "read-only"):
            check_payload_args(["--output_dir", str(out)], project)

    def test_nonexistent_relative_path_is_treated_as_prospective_output(self) -> None:
        project = self.home / "project"
        project.mkdir()
        with patch.dict(os.environ, self.env, clear=False), self.assertRaisesRegex(PathCheckError, "read-only"):
            check_payload_args(["results/new.root"], project)

    def test_project_specific_writable_arg_is_checked(self) -> None:
        project = self.home / "project"
        project.mkdir()
        target = self.home / "artifacts"
        target.mkdir()
        with patch.dict(os.environ, self.env, clear=False), self.assertRaisesRegex(PathCheckError, "read-only"):
            check_payload_args(["--artifact-store", str(target)], project, extra_writable=["--artifact-store"])


if __name__ == "__main__":
    unittest.main()
