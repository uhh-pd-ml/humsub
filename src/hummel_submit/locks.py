"""The control lock of a submission: at most one process runs a law pass or changes the supervisor schedule.

Held by the supervisor job for its whole run, and by ``humsub tick`` / ``resume`` / ``submit-manifest`` while they run a
pass or queue a supervisor.  It must work across nodes: on Hummel-2's BeeGFS ``flock`` is node-local (a compute node
acquires a lock the login node holds), while creating a file with ``O_EXCL`` is atomic cluster-wide.  The lock is
therefore the *existence* of ``<submission>/control.lock``, which records its owner.

A crashed owner cannot remove its lock (SIGKILL, OOM, node failure), so a lock is also released by its *staleness*,
decided without the owner:

* owner is a Slurm job: stale as soon as Slurm no longer lists the job (if Slurm cannot be asked, it counts as alive);
* owner is a process on the same host: stale when the pid is gone;
* any owner: stale when the file was not refreshed for ``LEASE_SECONDS`` (the holder touches it every
  ``REFRESH_SECONDS`` from a background thread).

A stale lock is broken by an atomic rename to a unique name, so two processes cannot both break it and both believe
they own it.  The lock is re-entrant for its owner (same Slurm job, or same host and pid).

:func:`mutex` is the same principle for short critical sections (read-modify-write of submission.json and of a chain's
state.json, chain creation), which therefore also exclude each other across nodes.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
from typing import Any, Iterator
import uuid

LOCK_NAME = "control.lock"
LEASE_SECONDS = 600
REFRESH_SECONDS = 60


class LockBusy(RuntimeError):
    def __init__(self, holder: dict[str, Any]) -> None:
        self.holder = holder
        super().__init__(f"the submission is busy: {describe(holder)}")


def describe(holder: dict[str, Any]) -> str:
    who = f"Slurm job {holder['job_id']}" if holder.get("job_id") else f"pid {holder.get('pid')} on {holder.get('host')}"
    return f"{holder.get('purpose', '?')} ({who}, since {time.strftime('%H:%M:%S', time.localtime(holder.get('since', 0)))})"


def me(purpose: str) -> dict[str, Any]:
    return {"job_id": os.environ.get("SLURM_JOB_ID") or None, "host": socket.gethostname(), "pid": os.getpid(),
            "purpose": purpose, "since": time.time(), "token": uuid.uuid4().hex}


def same_owner(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return bool(a.get("token")) and a.get("token") == b.get("token")


def slurm_job_alive(job_id: str) -> bool:
    """False only if Slurm positively does not list the job as pending/running/completing."""
    try:
        proc = subprocess.run(["squeue", "-h", "-j", str(job_id), "-o", "%T"], text=True, capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return True
    if proc.returncode != 0:
        # an unknown (purged) job id is an error for squeue -j; anything else means Slurm could not be asked
        return "invalid job id" not in (proc.stderr or "").lower()
    return bool(proc.stdout.strip())


def is_stale(holder: dict[str, Any], mtime: float, *, now: float | None = None) -> bool:
    now = time.time() if now is None else now
    if now - mtime > LEASE_SECONDS:
        return True
    if holder.get("job_id"):
        return not slurm_job_alive(str(holder["job_id"]))
    if holder.get("host") == socket.gethostname() and holder.get("pid"):
        try:
            os.kill(int(holder["pid"]), 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
    return False


def read_holder(path: Path) -> tuple[dict[str, Any], float] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8") or "{}"), path.stat().st_mtime
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        # being written right now (create, then write): treat as a fresh lock of an unknown owner
        return {"purpose": "starting"}, time.time()


def break_stale(path: Path, stale: dict[str, Any]) -> None:
    """Move a stale lock file aside; only one breaker wins the rename.  If the lock was released and taken by a live
    owner between the staleness check and the rename, put it back (``link`` never overwrites a lock created meanwhile)."""
    aside = path.with_name(f"{path.name}.stale-{uuid.uuid4().hex[:8]}")
    try:
        os.rename(path, aside)
    except FileNotFoundError:
        return
    moved = read_holder(aside)
    if moved and moved[0].get("token") != stale.get("token"):
        try:
            os.link(aside, path)
        except FileExistsError:
            pass
    aside.unlink(missing_ok=True)


#: A short-held mutex older than this is the leftover of a holder that was killed inside its critical section.
MUTEX_STALE_SECONDS = 60


@contextmanager
def mutex(path: Path, *, timeout: float = 120, stale: float = MUTEX_STALE_SECONDS) -> Iterator[None]:
    """Cross-node mutual exclusion for *short* critical sections (read-modify-write of a JSON file): the same O_EXCL
    principle as the control lock, without a refresh thread; a lock file older than MUTEX_STALE_SECONDS is broken.
    Not re-entrant.  Raises TimeoutError after *timeout* seconds of waiting."""
    path = Path(path)
    owner = me("mutex")
    deadline = time.time() + timeout
    delay = 0.02
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
        except FileExistsError:
            found = read_holder(path)
            if found is None:
                continue
            holder, mtime = found
            if time.time() - mtime > stale:
                break_stale(path, holder)
                continue
            if time.time() > deadline:
                raise TimeoutError(f"{path} has been held for more than {timeout:.0f} s ({describe(holder)})")
            time.sleep(delay)
            delay = min(delay * 2, 0.5)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(owner, handle)
        break
    try:
        yield
    finally:
        found = read_holder(path)
        if found and same_owner(found[0], owner):
            path.unlink(missing_ok=True)


class ControlLock:
    """``with ControlLock(submission_dir, "supervisor pass"):`` - see the module docstring."""

    _held: dict[str, int] = {}          # lock path -> nesting depth in this process

    def __init__(self, submission_dir: Path, purpose: str) -> None:
        self.path = Path(submission_dir) / LOCK_NAME
        self.owner = me(purpose)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- acquiring ---------------------------------------------------------------------------------------------------
    def try_acquire(self) -> bool:
        key = str(self.path)
        if self._held.get(key):
            self._held[key] += 1
            return True
        for _ in range(5):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
            except FileExistsError:
                found = read_holder(self.path)
                if found is None:
                    continue                                 # released meanwhile: try again
                holder, mtime = found
                if not is_stale(holder, mtime):
                    return False
                self._break(holder)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.owner, handle)
            break
        else:
            return False
        self._held[key] = 1
        self._stop.clear()
        self._thread = threading.Thread(target=self._refresh, daemon=True)
        self._thread.start()
        return True

    def _break(self, stale: dict[str, Any]) -> None:
        break_stale(self.path, stale)

    def acquire(self, *, wait: float = 0, poll: float = 5) -> None:
        deadline = time.time() + wait
        while not self.try_acquire():
            if time.time() >= deadline:
                raise LockBusy(self.holder() or {})
            time.sleep(poll)

    def holder(self) -> dict[str, Any] | None:
        found = read_holder(self.path)
        return found[0] if found else None

    # -- holding and releasing ---------------------------------------------------------------------------------------
    def _refresh(self) -> None:
        while not self._stop.wait(REFRESH_SECONDS):
            found = read_holder(self.path)
            if not found or not same_owner(found[0], self.owner):
                return                                       # not ours any more: never keep someone else's lock alive
            try:
                os.utime(self.path)
            except OSError:
                return

    def release(self) -> None:
        key = str(self.path)
        depth = self._held.get(key, 0)
        if depth > 1:
            self._held[key] = depth - 1
            return
        self._held.pop(key, None)
        self._stop.set()
        found = read_holder(self.path)
        if found and same_owner(found[0], self.owner):
            self.path.unlink(missing_ok=True)

    def __enter__(self) -> "ControlLock":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


@contextmanager
def control_lock(spec_path: Path, purpose: str, *, wait: float = 0) -> Iterator[ControlLock]:
    lock = ControlLock(Path(spec_path).parent, purpose)
    lock.acquire(wait=wait)
    try:
        yield lock
    finally:
        lock.release()


def held_here(spec_path: Path) -> bool:
    """Does this process hold the control lock of the submission?"""
    return bool(ControlLock._held.get(str(Path(spec_path).parent / LOCK_NAME)))
