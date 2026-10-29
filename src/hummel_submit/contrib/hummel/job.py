from __future__ import annotations

import pathlib
import time
from typing import Any, Callable, TypeVar

from law.config import Config
from law.job.base import BaseJobManager

from ...chain import cancel_chain, query_chain, submit_chain
from ...submission import load_submission_spec

T = TypeVar("T")

# Defaults of the ``hummel_manager_*`` options of law's ``[job]`` config section, chosen like law's
# ``slurm_manager_*`` defaults.
DEFAULT_THREADS = 4
DEFAULT_RETRIES = 1
DEFAULT_RETRY_DELAY = 5.0


class HummelJobManager(BaseJobManager):
    """law job manager that exposes autonomous humsub chains as stable jobs.

    law only ever sees the returned chain id.  The changing Slurm job ids that
    implement an autonomous continuation chain are private to humsub.

    *threads*, *retries* and *retry_delay* default to the ``hummel_manager_threads``,
    ``hummel_manager_retries`` and ``hummel_manager_retry_delay`` options of law's ``[job]``
    config section.  Like law's own managers, :py:meth:`submit`, :py:meth:`query`,
    :py:meth:`cancel` and :py:meth:`cleanup` accept ``retries=-1``/``retry_delay=-1`` meaning
    "use the instance default".
    """

    chunk_size_submit = 0
    chunk_size_cancel = 0
    chunk_size_cleanup = 0
    chunk_size_query = 0

    def __init__(
        self,
        submission_spec: str | pathlib.Path,
        threads: int | None = None,
        retries: int | None = None,
        retry_delay: float | int | None = None,
    ) -> None:
        cfg = Config.instance()
        if threads is None:
            threads = cfg.get_expanded_int("job", "hummel_manager_threads", DEFAULT_THREADS)
        if retries is None:
            retries = cfg.get_expanded_int("job", "hummel_manager_retries", DEFAULT_RETRIES)
        if retry_delay is None:
            retry_delay = cfg.get_expanded_float("job", "hummel_manager_retry_delay", DEFAULT_RETRY_DELAY)

        super().__init__(threads=threads)

        self.retries = max(0, retries)
        self.retry_delay = max(0, retry_delay)
        self.submission_spec = pathlib.Path(submission_spec).resolve()
        self._spec = load_submission_spec(self.submission_spec)
        self.output_dir = pathlib.Path(self._spec["output_dir"])

    @classmethod
    def cast_job_id(cls, job_id):
        # Chain ids are deliberately opaque strings, unlike Slurm's numeric ids.
        return str(job_id)

    def _with_retries(self, func: Callable[[], T], retries: int, retry_delay: float | int) -> T:
        if retries < 0:
            retries = self.retries
        if retry_delay < 0:
            retry_delay = self.retry_delay
        while True:
            try:
                return func()
            except Exception:
                if retries <= 0:
                    raise
                retries -= 1
                time.sleep(retry_delay)

    def submit(
        self,
        job_file,
        retries: int = -1,
        retry_delay: float | int = -1,
        silent: bool = False,
        **kwargs,
    ):
        # A retry is clean: submit_chain removes the frozen chain state again when sbatch fails.
        try:
            return self._with_retries(
                lambda: submit_chain(pathlib.Path(job_file), self.submission_spec),
                retries, retry_delay,
            )
        except Exception:
            if silent:
                return None
            raise

    def cancel(
        self,
        job_id,
        retries: int = -1,
        retry_delay: float | int = -1,
        silent: bool = False,
        **kwargs,
    ):
        # cancel_chain is idempotent (marker, status, scancel), so retrying it is safe.
        try:
            self._with_retries(
                lambda: cancel_chain(self.output_dir, str(job_id), reason="cancelled-by-law"),
                retries, retry_delay,
            )
        except Exception:
            if not silent:
                raise
        return None

    def cleanup(
        self,
        job_id,
        retries: int = -1,
        retry_delay: float | int = -1,
        silent: bool = False,
        **kwargs,
    ):
        # Persistent chain state, frozen workers and logs are intentional
        # provenance.  cleanup therefore only cancels a still-running chain; it
        # does not delete artifacts behind law's back.
        def cleanup_chain() -> None:
            status = query_chain(self.output_dir, str(job_id))
            if not status.terminal:
                cancel_chain(self.output_dir, str(job_id), reason="cleaned-up-by-law")

        try:
            self._with_retries(cleanup_chain, retries, retry_delay)
        except Exception:
            if not silent:
                raise
        return None

    def query(
        self,
        job_id,
        retries: int = -1,
        retry_delay: float | int = -1,
        silent: bool = False,
        **kwargs,
    ) -> dict[str, Any] | None:
        # A missing chain state is not an error here but a definite "failed" status from
        # query_chain, so only scheduler hiccups (squeue/sacct) are retried.
        try:
            status = self._with_retries(
                lambda: query_chain(self.output_dir, str(job_id)),
                retries, retry_delay,
            )
        except Exception:
            if silent:
                return None
            raise

        if status.state == "pending":
            law_status = self.PENDING
        elif status.state == "running":
            law_status = self.RUNNING
        elif status.state == "finished":
            law_status = self.FINISHED
        else:
            law_status = self.FAILED

        return self.job_status_dict(
            job_id=str(job_id),
            status=law_status,
            code=status.code,
            error=status.error,
            extra={"chain_id": str(job_id)},
        )
