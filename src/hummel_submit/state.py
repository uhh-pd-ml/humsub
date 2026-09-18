from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import zipfile
import uuid
from typing import Any

from . import __version__


def make_chain_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]


def default_run_name() -> str:
    return "run-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]


def chain_root(output_dir: Path) -> Path:
    return output_dir / ".hummel-submit" / "chains"


def chain_dir(output_dir: Path, chain_id: str) -> Path:
    return chain_root(output_dir) / chain_id


def state_path(output_dir: Path, chain_id: str) -> Path:
    return chain_dir(output_dir, chain_id) / "state.json"


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def create_state(
    *,
    output_dir: Path,
    project_dir: Path,
    config: dict[str, Any],
    run_name: str,
    user_args: list[str],
    resubmit: bool,
    python_executable: str,
    package_dir: Path,
) -> tuple[dict[str, Any], Path]:
    chain_id = make_chain_id()
    cdir = chain_dir(output_dir, chain_id)
    cdir.mkdir(parents=True, exist_ok=False)
    snapshot_path = cdir / "hummel-submit-worker.zip"
    with zipfile.ZipFile(snapshot_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(package_dir.rglob("*.py")):
            relative = source.relative_to(package_dir)
            archive.write(source, Path("hummel_submit") / relative)
    worker = cdir / "worker.sh"
    shutil.copy2(package_dir / "worker.sh", worker)

    run_dir = output_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "schema": 1,
        "package_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chain_id": chain_id,
        "project_dir": str(project_dir),
        "output_dir": str(output_dir),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "config": config,
        "user_args": user_args,
        "resubmit": resubmit,
        "python_executable": python_executable,
        "snapshot_path": str(snapshot_path),
        "worker_script": str(worker),
        "jobs": [],
        "status": "submitted",
    }
    path = cdir / "state.json"
    atomic_write_json(path, data)
    return data, path


def _locked_update(path: Path, update: Any) -> dict[str, Any]:
    lock_path = path.parent / ".state.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        data = load_state(path)
        update(data)
        atomic_write_json(path, data)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return data


def append_job(path: Path, job_id: str) -> dict[str, Any]:
    """Append a job id without losing a concurrent update from a fast-starting worker."""
    def update(data: dict[str, Any]) -> None:
        if job_id not in data["jobs"]:
            data["jobs"].append(job_id)
        data["last_job_id"] = job_id
        if data.get("status") == "submitted":
            data["status"] = "queued"
    return _locked_update(path, update)


def mark_status(path: Path, status: str, **fields: Any) -> dict[str, Any]:
    def update(data: dict[str, Any]) -> None:
        data["status"] = status
        data.update(fields)
    return _locked_update(path, update)


def done_marker(path: Path) -> Path:
    return path.parent / "done"


def continue_marker(path: Path, hop: int) -> Path:
    return path.parent / f"continue-{hop}"
