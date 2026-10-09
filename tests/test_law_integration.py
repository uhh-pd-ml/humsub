"""Contract test against the installed law: renders a real remote-job file."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

try:
    import law  # noqa: F401
except ImportError:  # pragma: no cover
    law = None

from hummel_submit.submission import create_submission_spec


@unittest.skipIf(law is None, "law is not installed")
class LawIntegrationTests(unittest.TestCase):
    def _workflow(self, root: Path):
        from hummel_submit.law_payload import PayloadWorkflow

        project = root / "project"
        project.mkdir()
        config = {
            "execution": {"output_dir": str(root / "out"), "cache_dir": str(root / "cache")},
            "slurm": {"job_name": "t", "partition": "std", "account": "a",
                      "reservation": None, "time_limit": "01:00:00"},
            "validation": {},
        }
        _, path = create_submission_spec(
            output_dir=root / "out", project_dir=project, config=config, run_name="r",
            user_args=["true"], resubmit=True, python_executable=sys.executable,
        )
        return PayloadWorkflow(
            humsub_spec=str(path), workflow="slurm", no_poll=True, retries=0,
            tasks_per_job=1, job_workers=1,
        )

    def test_job_manager_is_not_grouped(self) -> None:
        # law >= 0.2 submits Slurm job arrays when the manager declares grouping;
        # a humsub chain is one job, so grouping must stay off.
        from hummel_submit.contrib.hummel import HummelJobManager

        for attr in ("job_grouping_submit", "job_grouping_cancel",
                     "job_grouping_cleanup", "job_grouping_query"):
            self.assertFalse(getattr(HummelJobManager, attr))

    def test_renders_job_file(self) -> None:
        with tempfile.TemporaryDirectory(dir="/var/tmp") as td:
            proxy = self._workflow(Path(td)).workflow_proxy
            proxy.job_file_factory = proxy.create_job_file_factory()
            data = proxy.create_job_file(0, [0])
            text = Path(str(data["job"])).read_text(encoding="utf-8")
            self.assertIn("#SBATCH --job-name=t-law-0", text)
            self.assertIn("#SBATCH --partition=std", text)
            self.assertTrue(Path(str(data["job"])).is_relative_to(Path(td)))
            # python/law executables of the submission venv reach the rendered job
            job_dir = Path(str(data["job"])).parent
            rendered = "".join(p.read_text(encoding="utf-8") for p in job_dir.glob("law_job_*.sh"))
            self.assertIn(f'local python_exe="{sys.executable}"', rendered)


if __name__ == "__main__":
    unittest.main()
