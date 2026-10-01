from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
from typing import Any
import uuid

from . import __version__
from .state import atomic_write_json


def make_submission_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]


def submission_root(output_dir: Path) -> Path:
    return output_dir / ".hummel-submit" / "submissions"


def create_submission_spec(
    *,
    output_dir: Path,
    project_dir: Path,
    config: dict[str, Any],
    run_name: str,
    user_args: list[str],
    resubmit: bool,
    python_executable: str,
) -> tuple[dict[str, Any], Path]:
    submission_id = make_submission_id()
    sdir = submission_root(output_dir) / submission_id
    sdir.mkdir(parents=True, exist_ok=False)

    run_dir = output_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    law_dir = sdir / "law"
    (law_dir / "job-files").mkdir(parents=True, exist_ok=True)
    (law_dir / "control").mkdir(parents=True, exist_ok=True)

    law_bootstrap = law_dir / "bootstrap.sh"
    python_bin = Path(python_executable).expanduser().absolute().parent
    law_bootstrap.write_text(
        "#!/usr/bin/env bash\n"
        f"export PATH={str(python_bin)!r}:$PATH\n"
        "export PYTHONNOUSERSITE=1\n",
        encoding="utf-8",
    )

    data: dict[str, Any] = {
        "schema": 1,
        "package_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "submission_id": submission_id,
        "project_dir": str(project_dir),
        "output_dir": str(output_dir),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "success_marker": str(run_dir / ".humsub-complete"),
        "law_dir": str(law_dir),
        "law_bootstrap": str(law_bootstrap),
        "config": config,
        "user_args": user_args,
        "resubmit": resubmit,
        "python_executable": python_executable,
        "chain_ids": [],
    }
    path = sdir / "submission.json"
    atomic_write_json(path, data)
    return data, path


def load_submission_spec(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_submission_chain(path: Path, chain_id: str) -> dict[str, Any]:
    lock_path = path.parent / ".submission.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        data = load_submission_spec(path)
        if chain_id not in data["chain_ids"]:
            data["chain_ids"].append(chain_id)
        atomic_write_json(path, data)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return data
