from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hummel_submit.submission import create_submission_spec


class SubmissionTests(unittest.TestCase):
    def _make_venv(self, root: Path, *, symlink_python: bool = False) -> tuple[Path, Path]:
        venv = root / "venv"
        bindir = venv / "bin"
        bindir.mkdir(parents=True)
        law = bindir / "law"
        law.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        law.chmod(0o755)
        if symlink_python:
            real = root / "sw" / "bin" / "python3"
            real.parent.mkdir(parents=True)
            real.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            real.chmod(0o755)
            python = bindir / "python3"
            python.symlink_to(real)
        else:
            python = bindir / "python3"
            python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            python.chmod(0o755)
        return python, law

    def test_law_bootstrap_uses_frontend_venv_bin(self) -> None:
        with tempfile.TemporaryDirectory(dir="/var/tmp") as td:
            root = Path(td)
            project = root / "project"
            project.mkdir()
            output = root / "output"
            python, law = self._make_venv(root)
            with patch.dict(os.environ, {"VIRTUAL_ENV": str(root / "venv")}, clear=False):
                spec, _ = create_submission_spec(
                    output_dir=output,
                    project_dir=project,
                    config={"execution": {}, "slurm": {}, "validation": {}},
                    run_name="run-test",
                    user_args=[],
                    resubmit=True,
                    python_executable=str(python),
                )
            bootstrap = Path(spec["law_bootstrap"]).read_text(encoding="utf-8")
            self.assertEqual(spec["python_executable"], str(python))
            self.assertEqual(spec["law_executable"], str(law))
            self.assertIn(str(root / "venv" / "bin"), bootstrap)
            self.assertIn("$PATH", bootstrap)

    def test_virtualenv_python_symlink_is_not_resolved(self) -> None:
        with tempfile.TemporaryDirectory(dir="/var/tmp") as td:
            root = Path(td)
            project = root / "project"
            project.mkdir()
            python, law = self._make_venv(root, symlink_python=True)
            spec, _ = create_submission_spec(
                output_dir=root / "output",
                project_dir=project,
                config={"execution": {}, "slurm": {}, "validation": {}},
                run_name="run-test",
                user_args=[],
                resubmit=True,
                python_executable=str(python),
            )
            self.assertEqual(spec["python_executable"], str(python))
            self.assertEqual(spec["law_executable"], str(law))
            self.assertNotIn(str(root / "sw" / "bin"), Path(spec["law_bootstrap"]).read_text())


if __name__ == "__main__":
    unittest.main()
