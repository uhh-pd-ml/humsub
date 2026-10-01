from __future__ import annotations

from pathlib import Path

from law.target.local import LocalFileTarget
from law.task.base import Task as LawTask
from law.workflow.local import LocalWorkflow

from .contrib.hummel import HummelWorkflow
from .runner import run_application
from .submission import load_submission_spec


class PayloadTask(LawTask):
    """Base task for the generic humsub command-line payload workflow."""

    def run(self):
        raise NotImplementedError


class PayloadWorkflow(PayloadTask, HummelWorkflow, LocalWorkflow):
    """Single-branch law workflow used by the backwards-compatible humsub CLI.

    More specialized applications should define their own law workflows and
    inherit :class:`hummel_submit.contrib.hummel.HummelWorkflow` directly.  This
    built-in workflow exists so the original ``humsub -- <command args>`` user
    interface still benefits from law without requiring users to write task
    classes themselves.
    """

    def create_branch_map(self):
        return {0: None}

    def output(self):
        spec = load_submission_spec(Path(str(self.humsub_spec)))
        return LocalFileTarget(spec["success_marker"])

    def run(self):
        spec = load_submission_spec(Path(str(self.humsub_spec)))
        rc = run_application(spec)
        if rc != 0:
            raise RuntimeError(f"payload failed with exit code {rc}")
        output = self.output()
        output.parent.touch()
        output.touch()
