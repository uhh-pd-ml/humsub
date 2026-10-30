"""A *pass* over a submission's law task graph: law, not humsub, decides what to (re)submit.

A pass re-runs the submission's top law task with every Hummel workflow in ``no_poll`` mode.  On an existing
submission law then polls each workflow once (chain states via :class:`HummelJobManager`), resubmits failed jobs
within ``retries`` (counted across passes from the jobs file), submits jobs still waiting for ``parallel_jobs``
room, and returns; tasks whose requirements are complete run, the others wait for a later pass.  The supervisor job
runs one pass per ``supervisor_interval`` until the top task is complete or law gives up.

The top task is recorded in the submission as ``law_task`` = ``{"module", "class", "params"}``, so any law task
built on :class:`HummelWorkflow` can be driven the same way.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import fcntl
import importlib
import os
from pathlib import Path
from typing import Any, Iterator

from .submission import load_submission_spec

LOCK_NAME = ".tick.lock"
#: Set during a pass: every HummelWorkflow instantiated then runs with ``no_poll`` (see HummelWorkflow.__init__).
TICK_ENV = "HUMSUB_TICK"


@dataclass
class TickResult:
    state: str  # "complete" | "progress" | "failed" | "busy"
    errors: list[str] = field(default_factory=list)


def manifest_law_task(
    spec_path: Path, opts: dict[str, Any], cls: str = "ManifestPayloadWorkflow"
) -> dict[str, Any]:
    """A law task of a manifest submission (the branch workflow, or ManifestFinalizeWorkflow)."""
    return {
        "module": "hummel_submit.manifest_workflow",
        "class": cls,
        "params": {
            "humsub_spec": str(spec_path),
            "workflow": "slurm",
            "retries": int(opts.get("retries", 0)),
            "tasks_per_job": int(opts.get("tasks_per_job", 1)),
            "parallel_jobs": int(opts.get("parallel_jobs", 0)),
            "job_workers": 1,
        },
    }


def law_task_of(spec_path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    # submissions made before passes existed are manifest workflows
    return spec.get("law_task") or manifest_law_task(spec_path, spec.get("submit_options", {}))


def instantiate(desc: dict[str, Any]):
    cls = getattr(importlib.import_module(desc["module"]), desc["class"])
    return cls(**desc["params"])


_failures: list[tuple[str, BaseException]] = []
_handler_installed = False


def _install_failure_handler() -> None:
    global _handler_installed
    if _handler_installed:
        return
    import luigi

    @luigi.Task.event_handler(luigi.Event.FAILURE)
    def _record(task, exception):  # noqa: ANN001
        _failures.append((task.task_id, exception))

    _handler_installed = True


def _is_waiting(exc: BaseException) -> bool:
    # luigi refuses to run a task whose requirements are still incomplete; in a pass that only means "not yet"
    return isinstance(exc, RuntimeError) and "Unfulfilled" in str(exc)


@contextmanager
def _pass_lock(spec_path: Path) -> Iterator[bool]:
    with (spec_path.parent / LOCK_NAME).open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _tick_env() -> Iterator[None]:
    previous = os.environ.get(TICK_ENV)
    os.environ[TICK_ENV] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(TICK_ENV, None)
        else:
            os.environ[TICK_ENV] = previous


def run_tick(spec_path: Path) -> TickResult:
    """One pass over the submission's law task graph (never two at once for the same submission)."""
    import luigi
    from luigi.task_register import Register

    spec_path = spec_path.resolve()
    spec = load_submission_spec(spec_path)
    with _pass_lock(spec_path) as locked:
        if not locked:
            return TickResult("busy", ["another pass over this submission is running"])
        with _tick_env():
            # fresh task instances (and with them fresh law proxies): everything is re-read from disk
            Register.clear_instance_cache()
            _install_failure_handler()
            _failures.clear()
            task = instantiate(law_task_of(spec_path, spec))
            luigi.build([task], local_scheduler=True, workers=1)
            complete = task.complete()
    errors = [f"{task_id}: {exc}" for task_id, exc in _failures if not _is_waiting(exc)]
    if complete:
        return TickResult("complete")
    return TickResult("failed" if errors else "progress", errors)
