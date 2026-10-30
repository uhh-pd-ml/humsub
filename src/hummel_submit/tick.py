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
import importlib
import os
from pathlib import Path
from typing import Any, Iterator

from .submission import load_submission_spec

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
            "workflow": "hummel",
            "retries": int(opts.get("retries") or 0),
            "tasks_per_job": int(opts.get("tasks_per_job", 1)),
            "parallel_jobs": int(opts.get("parallel_jobs", 0)),
            "job_workers": 1,
        },
    }


def law_task_of(spec_path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    # submissions made before passes existed are manifest workflows
    return spec.get("law_task") or manifest_law_task(spec_path, spec.get("submit_options", {}))


def ensure_lock_generation(spec_path: Path) -> None:
    """Refuse to drive a submission whose (older) supervisor may still run without the cross-node control lock; once
    none runs, upgrade it: its next supervisor is built from this version (fresh snapshot) and takes the lock."""
    from .config import ConfigError
    from .slurm import live_job_ids
    from .submission import CONTROL_LOCK_GENERATION, update_submission_spec

    spec = load_submission_spec(spec_path)
    if int(spec.get("control_lock", 0)) >= CONTROL_LOCK_GENERATION:
        return
    control = Path(spec.get("law_dir", spec_path.parent / "law")) / "control"
    if list(control.glob("slurm_jobs_*.json")) and not list(control.glob("hummel_jobs_*.json")):
        # humsub <= 0.4 ran law with the slurm workflow type: a pass with the hummel type would not find law's job
        # data and submit every incomplete branch again, next to chains that may still be running
        raise ConfigError(
            "this submission was made with humsub <= 0.4 (law job data in slurm_jobs_*.json); drive it with the humsub "
            "version that made it, or resubmit its manifest (branches whose outputs exist are skipped)"
        )
    alive = live_job_ids() & {str(j) for j in spec.get("supervisor_job_ids", [])}
    if alive:
        raise ConfigError(
            f"supervisor job(s) {', '.join(sorted(alive))} of this submission run an older humsub without the "
            "cross-node control lock; wait for them to end (or scancel them), then try again"
        )
    for name in ("hummel-submit-worker.zip", "supervisor.sh"):
        (spec_path.parent / "supervisor" / name).unlink(missing_ok=True)
    update_submission_spec(spec_path, lambda data: data.update(control_lock=CONTROL_LOCK_GENERATION))


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


def run_tick(spec_path: Path, purpose: str = "law pass") -> TickResult:
    """One pass over the submission's law task graph, under the submission's control lock (re-entrant: a caller that
    already holds it, like the supervisor or resume, keeps it)."""
    import luigi
    from luigi.task_register import Register

    from .locks import ControlLock, describe

    spec_path = spec_path.resolve()
    spec = load_submission_spec(spec_path)
    lock = ControlLock(spec_path.parent, purpose)
    if not lock.try_acquire():
        return TickResult("busy", [f"the submission is busy: {describe(lock.holder() or {})}"])
    try:
        with _tick_env():
            # fresh task instances (and with them fresh law proxies): everything is re-read from disk
            Register.clear_instance_cache()
            _install_failure_handler()
            _failures.clear()
            task = instantiate(law_task_of(spec_path, spec))
            luigi.build([task], local_scheduler=True, workers=1)
            complete = task.complete()
    finally:
        lock.release()
    errors = [f"{task_id}: {exc}" for task_id, exc in _failures if not _is_waiting(exc)]
    if complete:
        return TickResult("complete")
    return TickResult("failed" if errors else "progress", errors)
