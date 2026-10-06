from __future__ import annotations

from pathlib import Path
from typing import Any

import luigi
from law.contrib.slurm.job import SlurmJobFileFactory
from law.contrib.slurm.workflow import SlurmWorkflow
from law.job.base import JobInputFile
from law.target.local import LocalDirectoryTarget

from .job import HummelJobManager
from ...submission import load_submission_spec


class HummelWorkflow(SlurmWorkflow):
    """law remote workflow whose scheduler-facing jobs are humsub chains.

    The implementation intentionally reuses law's Slurm *job-file renderer* but
    not its SlurmJobManager.  Rendered remote-job scripts are stored on the
    persistent output filesystem and handed to :class:`HummelJobManager`, which
    wraps each one in an autonomous Hummel chain and returns a stable chain id to
    law.
    """

    humsub_spec = luigi.Parameter(
        description="absolute path to the frozen humsub submission specification",
    )

    # HummelJobManager handles actual queue interaction.  Keep all law-generated
    # job files persistent because autonomous successor jobs may need them long
    # after the submission-side law process has exited.
    slurm_job_file_factory_defaults = {"cleanup": False}

    _humsub_spec_cache: dict[str, Any] | None = None

    def humsub_submission_spec(self) -> dict[str, Any]:
        if self._humsub_spec_cache is None:
            self._humsub_spec_cache = load_submission_spec(Path(str(self.humsub_spec)))
        return self._humsub_spec_cache

    def slurm_output_directory(self):
        spec = self.humsub_submission_spec()
        return LocalDirectoryTarget(str(Path(spec["law_dir"]) / "control"))

    def slurm_log_directory(self):
        # The outer Hummel Slurm jobs own logging.  The inner rendered law job
        # script runs as an ordinary shell payload, so law does not need a second
        # independent log transport.
        return None

    def slurm_bootstrap_file(self):
        spec = self.humsub_submission_spec()
        return JobInputFile(spec["law_bootstrap"], share=True)

    def slurm_use_local_scheduler(self) -> bool:
        # Remote branches should be self-contained and must not depend on a
        # luigi scheduler process surviving on the login node.
        return True

    def slurm_create_job_manager(self, **kwargs):
        return HummelJobManager(
            submission_spec=Path(str(self.humsub_spec)),
            threads=int(kwargs.get("threads", 1)),
        )

    def slurm_create_job_file_factory(self, **kwargs):
        spec = self.humsub_submission_spec()
        job_root = Path(spec["law_dir"]) / "job-files"
        job_root.mkdir(parents=True, exist_ok=True)

        factory_kwargs = dict(self.slurm_job_file_factory_defaults or {})
        factory_kwargs.update(kwargs)
        factory_kwargs.setdefault("dir", str(job_root))
        factory_kwargs.setdefault("mkdtemp", True)
        factory_kwargs["cleanup"] = False
        return SlurmJobFileFactory(**factory_kwargs)

    def slurm_job_config(self, config, job_num: int, branches: list[int]):
        # These directives are documentary when the rendered file is executed by
        # a humsub chain rather than submitted directly with sbatch.  Mirroring
        # the effective Hummel config still makes the frozen payload intelligible
        # when inspected manually.
        spec = self.humsub_submission_spec()
        slurm = spec["config"]["slurm"]
        config.job_name = f"{slurm['job_name']}-law-{job_num}"
        config.partition = slurm["partition"]

        # Render the exact submission-side host executables into law_job.sh.
        # In particular, do not resolve a virtualenv Python symlink into /sw/env:
        # the law console script lives in the virtualenv bin directory itself.
        config.render_variables["python_exe"] = spec["python_executable"]
        if spec.get("law_executable"):
            config.render_variables["law_exe"] = spec["law_executable"]
        return config

    def slurm_check_job_completeness(self) -> bool:
        # A chain is only scientifically successful when its law branch targets
        # exist.  This catches unusual cases where the outer chain itself exits
        # cleanly but a branch failed to materialize its target.
        return True
