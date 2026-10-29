"""Contract test against the installed law: renders a real remote-job file."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

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
            humsub_spec=str(path), workflow="hummel", no_poll=True, retries=0,
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

    def test_hummel_is_its_own_workflow_type(self) -> None:
        with tempfile.TemporaryDirectory(dir="/var/tmp") as td:
            wf = self._workflow(Path(td))
            self.assertEqual(wf.workflow_proxy.workflow_type, "hummel")

    def test_coexists_with_law_slurm_workflow(self) -> None:
        # A task may offer both backends; --workflow selects one, neither shadows the other.
        from law.contrib.slurm import SlurmWorkflow
        from law.workflow.local import LocalWorkflow
        from hummel_submit.contrib.hummel import HummelJobManager, HummelWorkflow

        class Both(HummelWorkflow, SlurmWorkflow, LocalWorkflow):
            def create_branch_map(self):
                return {0: None}

            def slurm_output_directory(self):
                return "/tmp"

        self.assertEqual(Both.find_workflow_cls("hummel").workflow_proxy_cls.workflow_type, "hummel")
        self.assertEqual(Both.find_workflow_cls("slurm").workflow_proxy_cls.workflow_type, "slurm")
        self.assertFalse(issubclass(HummelWorkflow, SlurmWorkflow))
        self.assertFalse(hasattr(HummelWorkflow, "slurm_partition"))
        self.assertIsNot(Both.slurm_create_job_manager, Both.hummel_create_job_manager)
        self.assertNotEqual(HummelJobManager, SlurmWorkflow.slurm_job_manager_cls)

    def test_job_manager_takes_threads_and_retries_from_law_defaults(self) -> None:
        with tempfile.TemporaryDirectory(dir="/var/tmp") as td:
            proxy = self._workflow(Path(td)).workflow_proxy
            # law >= 0.2 (master since 2026-10) creates the manager without arguments
            manager = proxy.create_job_manager()
            self.assertEqual((manager.threads, manager.retries, manager.retry_delay), (4, 1, 5.0))


@unittest.skipIf(law is None, "law is not installed")
class HummelJobManagerRetryTests(unittest.TestCase):
    """law passes retries/retry_delay to submit/query/cancel/cleanup; -1 means the instance default."""

    def _manager(self, root: Path, **kwargs):
        from hummel_submit.contrib.hummel import HummelJobManager

        with patch("hummel_submit.contrib.hummel.job.load_submission_spec", return_value={"output_dir": str(root)}):
            return HummelJobManager(root / "submission.json", threads=1, retry_delay=0, **kwargs)

    @staticmethod
    def _flaky(failures: int, result=None):
        calls = []

        def func(*args, **kwargs):
            calls.append(args)
            if len(calls) <= failures:
                raise RuntimeError("slurmctld timeout")
            return result
        return func, calls

    def test_submit_uses_instance_retries_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            manager = self._manager(Path(td), retries=2)
            func, calls = self._flaky(2, "chain-1")
            with patch("hummel_submit.contrib.hummel.job.submit_chain", side_effect=func):
                self.assertEqual(manager.submit("job.sh"), "chain-1")
            self.assertEqual(len(calls), 3)

    def test_explicit_retries_override_instance_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            manager = self._manager(Path(td), retries=5)
            func, calls = self._flaky(5)
            with patch("hummel_submit.contrib.hummel.job.cancel_chain", side_effect=func):
                with self.assertRaises(RuntimeError):
                    manager.cancel("chain-1", retries=1)
            self.assertEqual(len(calls), 2)

    def test_query_retries_scheduler_errors_and_silent_returns_none(self) -> None:
        from hummel_submit.chain import ChainStatus

        with tempfile.TemporaryDirectory() as td:
            manager = self._manager(Path(td), retries=1)
            func, calls = self._flaky(1, ChainStatus("chain-1", "running"))
            with patch("hummel_submit.contrib.hummel.job.query_chain", side_effect=func):
                self.assertEqual(manager.query("chain-1")["status"], manager.RUNNING)
            self.assertEqual(len(calls), 2)

            func, calls = self._flaky(10)
            with patch("hummel_submit.contrib.hummel.job.query_chain", side_effect=func):
                self.assertIsNone(manager.query("chain-1", retries=0, silent=True))
            self.assertEqual(len(calls), 1)

    def test_cleanup_retries_and_only_cancels_live_chains(self) -> None:
        from hummel_submit.chain import ChainStatus

        with tempfile.TemporaryDirectory() as td:
            manager = self._manager(Path(td), retries=1)
            func, calls = self._flaky(1, ChainStatus("chain-1", "finished", code=0))
            with patch("hummel_submit.contrib.hummel.job.query_chain", side_effect=func), \
                    patch("hummel_submit.contrib.hummel.job.cancel_chain") as cancel:
                manager.cleanup("chain-1")
            self.assertEqual(len(calls), 2)
            cancel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
