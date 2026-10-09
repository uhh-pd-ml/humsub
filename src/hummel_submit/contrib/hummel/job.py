from __future__ import annotations

import pathlib
import time
from typing import Any

from law.job.base import BaseJobManager

from ...chain import cancel_chain, query_chain, submit_chain
from ...submission import load_submission_spec


class HummelJobManager(BaseJobManager):
    """law job manager that exposes autonomous humsub chains as stable jobs.

    law only ever sees the returned chain id.  The changing Slurm job ids that
    implement an autonomous continuation chain are private to humsub.
    """

    chunk_size_submit = 0
    chunk_size_cancel = 0
    chunk_size_cleanup = 0
    chunk_size_query = 0

    def __init__(self, submission_spec: str | pathlib.Path, threads: int = 1) -> None:
        super().__init__(threads=threads)
        self.submission_spec = pathlib.Path(submission_spec).resolve()
        self._spec = load_submission_spec(self.submission_spec)
        self.output_dir = pathlib.Path(self._spec["output_dir"])

    @classmethod
    def cast_job_id(cls, job_id):
        # Chain ids are deliberately opaque strings, unlike Slurm's numeric ids.
        return str(job_id)

    def submit(
        self,
        job_file,
        retries: int = 0,
        retry_delay: float | int = 3,
        silent: bool = False,
        **kwargs,
    ):
        while True:
            try:
                return submit_chain(pathlib.Path(job_file), self.submission_spec)
            except Exception:
                if retries > 0:
                    retries -= 1
                    time.sleep(retry_delay)
                    continue
                if silent:
                    return None
                raise

    def cancel(self, job_id, silent: bool = False, **kwargs):
        try:
            cancel_chain(self.output_dir, str(job_id), reason="cancelled-by-law")
        except Exception:
            if not silent:
                raise
        return None

    def cleanup(self, job_id, silent: bool = False, **kwargs):
        # Persistent chain state, frozen workers and logs are intentional
        # provenance.  cleanup therefore only cancels a still-running chain; it
        # does not delete artifacts behind law's back.
        try:
            status = query_chain(self.output_dir, str(job_id))
            if not status.terminal:
                cancel_chain(self.output_dir, str(job_id), reason="cleaned-up-by-law")
        except Exception:
            if not silent:
                raise
        return None

    def query(self, job_id, silent: bool = False, **kwargs) -> dict[str, Any] | None:
        try:
            status = query_chain(self.output_dir, str(job_id))
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
