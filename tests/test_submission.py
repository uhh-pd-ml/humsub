from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from hummel_submit.submission import create_submission_spec


class SubmissionTests(unittest.TestCase):
    def test_law_bootstrap_uses_frontend_venv_bin(self) -> None:
        with tempfile.TemporaryDirectory(dir="/var/tmp") as td:
            root = Path(td)
            project = root / "project"
            project.mkdir()
            output = root / "output"
            python_executable = str(root / "venv" / "bin" / "python")
            spec, _ = create_submission_spec(
                output_dir=output,
                project_dir=project,
                config={"execution": {}, "slurm": {}, "validation": {}},
                run_name="run-test",
                user_args=[],
                resubmit=True,
                python_executable=python_executable,
            )
            bootstrap = Path(spec["law_bootstrap"]).read_text(encoding="utf-8")
            self.assertIn(str(root / "venv" / "bin"), bootstrap)
            self.assertIn("$PATH", bootstrap)


if __name__ == "__main__":
    unittest.main()
