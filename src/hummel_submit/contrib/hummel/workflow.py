"""Hummel workflow implementation for law.

A dedicated remote workflow type (``workflow_type = "hummel"``, ``--workflow hummel``) modeled on law's
``SlurmWorkflowProxy``/``SlurmWorkflow``, but independent of them: a task can inherit from both
:class:`HummelWorkflow` and ``law.slurm.SlurmWorkflow`` and select either with ``--workflow``, and
the ``slurm_*`` parameters, hooks and ``[job]`` config options of law's Slurm backend do not apply.

Every law remote job becomes one humsub chain (no job arrays).  The job script is rendered with
law's ``SlurmJobFileFactory``; its ``#SBATCH`` directives are only documentary, because the chain
executes the rendered script as a plain shell payload inside its own Slurm jobs.
"""

from __future__ import annotations

__all__ = ["HummelWorkflow"]

import contextlib
import os
import pathlib
from typing import Any, Generator

import luigi
from law.contrib.slurm.job import SlurmJobFileFactory
from law.job.base import JobArguments, JobInputFile
from law.target.file import FileSystemDirectoryTarget, get_path
from law.target.local import LocalDirectoryTarget
from law.task.proxy import ProxyCommand
from law.util import DotDict, InsertableDict, law_src_path, merge_dicts
from law.workflow.remote import BaseRemoteWorkflow, BaseRemoteWorkflowProxy, PollData

from ...submission import load_submission_spec
from ...tick import TICK_ENV
from .job import HummelJobManager


class HummelWorkflowProxy(BaseRemoteWorkflowProxy):

    workflow_type: str = "hummel"

    def poll(self) -> None:
        # law counts retries in memory only (``_job_retries``) but persists ``attempts`` in the jobs file.  A pass
        # (hummel_submit.tick) is a fresh process, so without this every pass would grant ``retries`` new attempts.
        for job_num, attempts in self.job_data.attempts.items():
            self._job_retries[int(job_num)] = max(self._job_retries[int(job_num)], int(attempts))
        super().poll()

    def create_job_manager(self, **kwargs) -> HummelJobManager:
        return self.task.hummel_create_job_manager(**kwargs)

    def create_job_file_factory(self, **kwargs) -> SlurmJobFileFactory:
        return self.task.hummel_create_job_file_factory(**kwargs)

    def create_job_file(self, job_num: int, branches: list[int]) -> dict[str, Any]:
        task: HummelWorkflow = self.task

        # the file postfix is pythonic range made from branches, e.g. [0, 1, 2, 4] -> "_0To5"
        postfix = f"_{branches[0]}To{branches[-1] + 1}"

        c = self.job_file_factory.get_config()  # type: ignore[union-attr]
        c.input_files = {}
        c.render_variables = {}
        c.custom_content = []

        # the job file is rendered locally and executed directly by the chain
        law_job_file = task.hummel_job_file()
        if not isinstance(law_job_file, JobInputFile):
            law_job_file = JobInputFile(get_path(law_job_file))
        law_job_file.copy = True
        law_job_file.share = False
        law_job_file.postfix = True
        law_job_file.render_local = True
        law_job_file.render_job = False
        c.input_files["job_file"] = law_job_file
        c.executable = law_job_file

        # collect task parameters
        exclude_args = (
            task.exclude_params_branch |
            task.exclude_params_workflow |
            task.exclude_params_remote_workflow |
            task.exclude_params_hummel_workflow |
            {"workflow", "effective_workflow"}
        )
        proxy_cmd = ProxyCommand(
            task.as_branch(branches[0]),
            exclude_task_args=list(exclude_args),
            exclude_global_args=["workers", "local-scheduler", f"{task.task_family}-*"],
        )
        if task.hummel_use_local_scheduler():
            proxy_cmd.add_arg("--local-scheduler", "True", overwrite=True)
        for key, value in dict(task.hummel_cmdline_args()).items():
            proxy_cmd.add_arg(key, value, overwrite=True)

        c.arguments = JobArguments(
            task_cls=task.__class__,
            task_params=proxy_cmd.build(skip_run=True),
            branches=branches,
            workers=task.job_workers,
            auto_retry=False,
            dashboard_data=(
                self.dashboard.remote_hook_data(job_num, self.job_data.attempts.get(job_num, 0))
                if self.dashboard is not None
                else None
            ),
        ).join()

        bootstrap_file = task.hummel_bootstrap_file()
        if bootstrap_file:
            c.input_files["bootstrap_file"] = bootstrap_file

        stageout_file = task.hummel_stageout_file()
        if stageout_file:
            c.input_files["stageout_file"] = stageout_file

        if self.dashboard is not None:
            dashboard_file = self.dashboard.remote_hook_file()
            if dashboard_file:
                c.input_files["dashboard_file"] = dashboard_file

        # The chain's Slurm jobs own stdout/stderr and the logs (<output_dir>/logs); law does not
        # transfer a second copy, so transfer_logs has no effect here.
        c.stdout = None
        c.stderr = None
        c.custom_log_file = None

        # the control directory is local (the chain needs the files long after submission)
        output_dir = task.hummel_output_directory()
        if not isinstance(output_dir, FileSystemDirectoryTarget):
            output_dir = LocalDirectoryTarget(get_path(output_dir))
        output_dir.touch()
        c.absolute_paths = True
        c.custom_content.append(("chdir", output_dir.abspath))

        c.job_name = f"{task.live_task_id}{postfix}"

        # keep law's tmp directory short (multiprocessing socket paths are limited to ~100 chars)
        if not c.render_variables.get("law_job_tmp"):
            c.render_variables["law_job_tmp"] = "/tmp/law_$( basename \"$LAW_JOB_HOME\" )"

        c = task.hummel_job_config(c, job_num, branches)

        job_file, c = self.job_file_factory(postfix=postfix, **c.__dict__)  # type: ignore[misc]

        return {"job": job_file, "config": c, "log": None}

    def destination_info(self) -> InsertableDict:
        info = super().destination_info()
        return self.task.hummel_destination_info(info)


class HummelWorkflow(BaseRemoteWorkflow):
    """law remote workflow whose scheduler-facing jobs are humsub chains.

    Every remote job is wrapped by :class:`HummelJobManager` in an autonomous Hummel chain, and law
    only sees the stable chain id.  Rendered job scripts are stored on the persistent output
    filesystem because continuation hops need them long after the submitting law process exited.
    The ``hummel_*`` hooks mirror the ``slurm_*`` hooks of ``law.slurm.SlurmWorkflow`` where they
    make sense for chains; Slurm resources (account, partition, GPUs, time limit) come from the
    frozen humsub submission specification, not from law parameters.

    .. code-block:: python

        class MyTask(HummelWorkflow, law.slurm.SlurmWorkflow, law.LocalWorkflow):
            ...  # --workflow hummel | slurm | local

    humsub drives the workflow by *passes* (:mod:`hummel_submit.tick`): law re-run with ``no_poll`` on an existing
    submission polls once, resubmits failed jobs within ``retries``, submits pending ones and returns.
    """

    workflow_proxy_cls = HummelWorkflowProxy

    hummel_workflow_run_decorators: list | None = None
    hummel_job_manager_defaults: dict | None = None
    hummel_job_file_factory_defaults: dict | None = None

    humsub_spec = luigi.Parameter(
        description="absolute path to the frozen humsub submission specification",
    )

    hummel_job_kwargs: list[str] = []
    hummel_job_kwargs_submit: dict | None = None
    hummel_job_kwargs_cancel: dict | None = None
    hummel_job_kwargs_query: dict | None = None

    exclude_params_hummel_workflow: set[str] = set()

    exclude_index = True

    _humsub_spec_cache: dict[str, Any] | None = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # ``no_poll`` is not passed on by law's req() (exclude_params_req_set), so a workflow required by another task
        # would poll in a blocking loop; inside a pass every Hummel workflow polls exactly once instead.
        if os.environ.get(TICK_ENV) == "1":
            self.no_poll = True

    def humsub_submission_spec(self) -> dict[str, Any]:
        if self._humsub_spec_cache is None:
            self._humsub_spec_cache = load_submission_spec(pathlib.Path(str(self.humsub_spec)))
        return self._humsub_spec_cache

    def hummel_output_directory(self) -> str | pathlib.Path | FileSystemDirectoryTarget:
        """Location of law's control files (job data json); defaults to ``<law_dir>/control``."""
        return LocalDirectoryTarget(str(pathlib.Path(self.humsub_submission_spec()["law_dir"]) / "control"))

    def hummel_output_postfix(self) -> str:
        return ""

    @contextlib.contextmanager
    def hummel_workflow_run_context(self) -> Generator[None, None, None]:
        yield

    def hummel_workflow_requires(self) -> DotDict:
        return DotDict()

    def hummel_job_resources(self, job_num: int, branches: list[int]) -> dict[str, int]:
        return {}

    def hummel_bootstrap_file(self) -> str | pathlib.Path | JobInputFile | None:
        return JobInputFile(self.humsub_submission_spec()["law_bootstrap"], share=True)

    def hummel_job_file(self) -> str | pathlib.Path | JobInputFile:
        return JobInputFile(law_src_path("job", "law_job.sh"), copy=True, share=True, render_job=True)

    def hummel_stageout_file(self) -> str | pathlib.Path | JobInputFile | None:
        return None

    def hummel_job_manager_cls(self) -> type[HummelJobManager]:
        return HummelJobManager

    def hummel_create_job_manager(self, **kwargs) -> HummelJobManager:
        kwargs = merge_dicts(
            {"submission_spec": pathlib.Path(str(self.humsub_spec))},
            self.hummel_job_manager_defaults,
            kwargs,
        )
        return self.hummel_job_manager_cls()(**kwargs)

    def hummel_job_file_factory_cls(self) -> type[SlurmJobFileFactory]:
        return SlurmJobFileFactory

    def hummel_create_job_file_factory(self, **kwargs) -> SlurmJobFileFactory:
        # Job files stay on the persistent output filesystem: autonomous successor hops re-run them
        # long after the submission-side law process has exited.
        job_root = pathlib.Path(self.humsub_submission_spec()["law_dir"]) / "job-files"
        job_root.mkdir(parents=True, exist_ok=True)
        kwargs = merge_dicts({"dir": str(job_root), "mkdtemp": True}, self.hummel_job_file_factory_defaults, kwargs)
        kwargs["cleanup"] = False
        return self.hummel_job_file_factory_cls()(**kwargs)

    def hummel_job_config(
        self,
        config: SlurmJobFileFactory.Config,
        job_num: int,
        branches: list[int],
    ) -> SlurmJobFileFactory.Config:
        # The directives are documentary (the chain runs the file as a shell payload); mirroring
        # the effective Hummel config keeps the frozen payload intelligible when inspected.
        spec = self.humsub_submission_spec()
        slurm = spec["config"]["slurm"]
        config.job_name = f"{slurm['job_name']}-law-{job_num}"
        config.partition = slurm["partition"]

        # Render the exact submission-side host executables into law_job.sh.  In particular, do
        # not resolve a virtualenv Python symlink into /sw/env: the law console script lives in
        # the virtualenv bin directory itself.
        config.render_variables["python_exe"] = spec["python_executable"]
        if spec.get("law_executable"):
            config.render_variables["law_exe"] = spec["law_executable"]
        return config

    def hummel_dump_intermediate_job_data(self) -> bool:
        return True

    def hummel_post_submit_delay(self) -> int | float:
        return self.poll_interval * 60

    def hummel_check_job_completeness(self) -> bool:
        # A chain is only scientifically successful when its law branch targets exist.  This
        # catches chains that exit cleanly although a branch did not materialize its target.
        return True

    def hummel_check_job_completeness_delay(self) -> float | int:
        return 0.0

    def hummel_poll_callback(self, poll_data: PollData) -> bool | None:
        return None

    def hummel_use_local_scheduler(self) -> bool:
        # Remote branches must not depend on a luigi scheduler surviving on the login node.
        return True

    def hummel_cmdline_args(self) -> dict[str, str]:
        return {}

    def hummel_destination_info(self, info: InsertableDict) -> InsertableDict:
        return info
