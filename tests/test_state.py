from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
import zipfile

import hummel_submit
from hummel_submit.state import create_chain_state


def test_tmp_base() -> str:
    """Use a neutral filesystem root that pathcheck does not classify as Hummel storage."""
    return os.environ.get("HUMSUB_TEST_TMPDIR", "/var/tmp")


class StateTests(unittest.TestCase):
    def test_worker_snapshot_is_single_importable_zip(self) -> None:
        with tempfile.TemporaryDirectory(dir=test_tmp_base()) as td:
            root = Path(td)
            output = root / "output"
            project = root / "project"
            project.mkdir()
            payload = root / "law-job.sh"
            payload.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
            package_dir = Path(hummel_submit.__file__).resolve().parent
            config = {
                "execution": {"checkpoint_glob": ""},
                "slurm": {},
                "validation": {},
            }
            state, state_path = create_chain_state(
                output_dir=output,
                project_dir=project,
                config=config,
                run_name="run-test",
                resubmit=True,
                python_executable="/usr/bin/python3",
                package_dir=package_dir,
                payload_script=payload,
                submission_id="submission-test",
            )
            snapshot = Path(state["snapshot_path"])
            self.assertTrue(snapshot.is_file())
            self.assertFalse((state_path.parent / "snapshot").exists())
            self.assertEqual(Path(state["payload_script"]), payload.resolve())
            with zipfile.ZipFile(snapshot) as archive:
                names = set(archive.namelist())
            self.assertIn("hummel_submit/worker.py", names)
            self.assertIn("hummel_submit/chain_runner.py", names)
            self.assertIn("hummel_submit/pathcheck.py", names)


if __name__ == "__main__":
    unittest.main()


def test_snapshot_contains_shell_scripts_and_modules(tmp_path):
    import zipfile
    import hummel_submit
    from hummel_submit.state import write_worker_snapshot

    package_dir = Path(hummel_submit.__file__).parent
    snapshot = tmp_path / "snap.zip"
    write_worker_snapshot(package_dir, snapshot)
    names = set(zipfile.ZipFile(snapshot).namelist())
    assert {"hummel_submit/worker.py", "hummel_submit/worker.sh", "hummel_submit/supervisor.sh"} <= names
