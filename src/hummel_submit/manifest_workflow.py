from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time

import law

from .contrib.hummel import HummelWorkflow
from .scratch import (
    effective_scratch_kind,
    hop_timing,
    keep_small_files,
    remove_scratch,
    reset_scratch,
    retry_allowed,
    scratch_dirs,
)
from .submission import load_submission_spec


_ENV_KEY_RE = re.compile(r"[^A-Za-z0-9]+")


def _stage_env_name(name: str) -> str:
    return "HUMSUB_STAGE_" + _ENV_KEY_RE.sub("_", name).strip("_").upper()


class ManifestPayloadWorkflow(HummelWorkflow, law.LocalWorkflow):
    """Generic law workflow that maps a frozen JSON manifest to an executable payload.

    Each law branch receives one immutable JSON context file.  The payload is
    application-owned; humsub owns branch scheduling, staging, scratch, retries,
    autonomous Slurm continuation, and output-completeness integration with law.
    """

    def _spec(self):
        return load_submission_spec(Path(str(self.humsub_spec)))

    def _workflow(self):
        spec = self._spec()
        workflow = spec.get("manifest_workflow")
        if not isinstance(workflow, dict):
            raise RuntimeError("submission has no manifest_workflow configuration")
        return workflow

    def _manifest(self):
        return json.loads(Path(self._workflow()["manifest"]).read_text(encoding="utf-8"))

    def create_branch_map(self):
        return {int(branch["id"]): branch for branch in self._manifest()["branches"]}

    def output(self):
        targets = [law.LocalFileTarget(path) for path in self.branch_data["outputs"]]
        return targets[0] if len(targets) == 1 else targets

    def run(self):
        soft_file = os.environ.get("HUMSUB_SOFT_STOP_FILE")
        if soft_file and Path(soft_file).exists():
            # The hop is past its soft deadline (the previous branch finished inside the grace window): starting this
            # branch would only get it killed.  The follower hop starts it.
            raise RuntimeError("soft-stop notice already given: branch not started, the next hop runs it")
        spec = self._spec()
        workflow = self._workflow()
        branch_id = int(self.branch)
        branch_file = Path(workflow["branch_files"][str(branch_id)])
        payload = Path(workflow["payload"])

        targets = self.output()
        target_list = targets if isinstance(targets, list) else [targets]

        def clear_outputs() -> None:
            # Only a branch that law considers incomplete runs; remove any partial subset left by
            # a previous failed attempt before invoking the application payload again.
            for target in target_list:
                Path(target.path).unlink(missing_ok=True)

        clear_outputs()

        slurm_id = os.environ.get("SLURM_JOB_ID", "local")
        hint = self._manifest().get("humsub") or {}
        kind = effective_scratch_kind(spec["config"], hint, workflow.get("scratch"))
        dirs = scratch_dirs(spec["config"], spec["submission_id"], slurm_id, branch_id, kind)
        reset_scratch(dirs)
        retries = int(workflow.get("retry_payload", 0))
        leftovers = Path(spec["run_dir"]) / "leftovers" / f"{slurm_id}-{branch_id}"

        base_env = os.environ.copy()
        base_env.update({
            "HUMSUB_SUBMISSION_ID": spec["submission_id"],
            "HUMSUB_RUN_NAME": spec["run_name"],
            "HUMSUB_RUN_DIR": spec["run_dir"],
            "HUMSUB_BRANCH": str(branch_id),
            "HUMSUB_BRANCH_FILE": str(branch_file),
            "HUMSUB_SCRATCH": str(dirs.bulk),
            "HUMSUB_SCRATCH_FAST": str(dirs.fast),
            "HUMSUB_PAYLOAD": str(payload),
            "HUMSUB_ATTEMPT": os.environ.get("LAW_JOB_ATTEMPT", "1"),
        })
        for name, path in workflow.get("stages", {}).items():
            base_env[_stage_env_name(name)] = str(path)

        interrupted = threading.Event()
        child: list[subprocess.Popen | None] = [None]

        def on_term(signum: int, frame: object) -> None:
            # The chain worker sends SIGTERM shortly before the time limit.  Pass it on to the
            # payload (which runs in its own session), give it a moment, and let ``finally`` below
            # remove the scratch: a killed worker would otherwise leave hundreds of MB behind.
            interrupted.set()
            proc = child[0]
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

        previous = None
        try:
            previous = signal.signal(signal.SIGTERM, on_term)
        except ValueError:  # not the main thread (e.g. unit tests): no handler
            previous = None

        print(f"[humsub] branch={branch_id} context={branch_file}", flush=True)
        print(f"[humsub] payload={payload}", flush=True)
        print(f"[humsub] scratch={dirs.bulk} fast={dirs.fast} retry_payload={retries}", flush=True)

        durations: list[float] = []
        failed = 0
        try:
            while True:
                env = dict(base_env, HUMSUB_PAYLOAD_ATTEMPT=str(failed + 1))
                started = time.time()
                proc = subprocess.Popen(
                    [str(payload), str(branch_file)],
                    cwd=spec["project_dir"],
                    env=env,
                    start_new_session=True,
                )
                child[0] = proc
                if interrupted.is_set():  # signal arrived between spawn and registration
                    on_term(signal.SIGTERM, None)
                rc = _wait_with_grace(proc, interrupted)
                child[0] = None
                durations.append(time.time() - started)
                if interrupted.is_set():
                    raise RuntimeError("branch interrupted by the pre-timeout signal; scratch removed, the next hop re-runs it")
                missing = [] if rc != 0 else [Path(t.path) for t in target_list if not Path(t.path).exists()]
                if rc == 0 and not missing:
                    return
                if rc != 0 and soft_file and Path(soft_file).exists():
                    # The payload gave up on the soft-stop notice (e.g. checkpointed and exited): that is a hop boundary,
                    # not a failure, so no retry and no failure accounting; the next hop re-runs the branch.
                    raise RuntimeError(f"payload stopped after the soft-stop notice (exit {rc}); the next hop re-runs it")
                failed += 1
                reason = (
                    f"payload failed with exit code {rc}" if rc != 0
                    else "payload returned success but did not materialize declared output(s): " + ", ".join(map(str, missing))
                )
                elapsed, usable = hop_timing()
                allowed, why = retry_allowed(
                    failed_attempts=failed,
                    max_retries=retries,
                    elapsed=elapsed,
                    usable=usable,
                    attempt_durations=durations,
                )
                print(f"[humsub] attempt {failed} failed: {reason}; {why}", flush=True)
                if not allowed:
                    raise RuntimeError(reason)
                clear_outputs()
                reset_scratch(dirs)
        except BaseException:
            clear_outputs()
            kept = keep_small_files(dirs, leftovers)
            if kept:
                print(f"[humsub] kept {len(kept)} small file(s) of the failed branch in {leftovers}", flush=True)
            raise
        finally:
            remove_scratch(dirs)
            if previous is not None:
                signal.signal(signal.SIGTERM, previous)


def _wait_with_grace(proc: subprocess.Popen, interrupted: threading.Event, grace: float = 45.0) -> int:
    """Wait for the payload; after an interruption allow *grace* seconds, then SIGKILL its group."""
    deadline: float | None = None
    while True:
        try:
            return proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        if interrupted.is_set():
            if deadline is None:
                deadline = time.time() + grace
            elif time.time() > deadline:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                return proc.wait()
