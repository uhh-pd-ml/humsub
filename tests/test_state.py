from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import zipfile

import hummel_submit
from hummel_submit.state import create_state


class StateTests(unittest.TestCase):
    def test_worker_snapshot_is_single_importable_zip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output = root / "output"
            project = root / "project"
            project.mkdir()
            package_dir = Path(hummel_submit.__file__).resolve().parent
            config = {
                "execution": {"checkpoint_glob": ""},
                "slurm": {},
                "validation": {},
            }
            state, state_path = create_state(
                output_dir=output,
                project_dir=project,
                config=config,
                run_name="run-test",
                user_args=[],
                resubmit=True,
                python_executable="/usr/bin/python3",
                package_dir=package_dir,
            )
            snapshot = Path(state["snapshot_path"])
            self.assertTrue(snapshot.is_file())
            self.assertFalse((state_path.parent / "snapshot").exists())
            with zipfile.ZipFile(snapshot) as archive:
                names = set(archive.namelist())
            self.assertIn("hummel_submit/worker.py", names)
            self.assertIn("hummel_submit/pathcheck.py", names)


if __name__ == "__main__":
    unittest.main()
