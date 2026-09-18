from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import hummel_submit.config as config_mod
from hummel_submit.config import ConfigError, load_config, validate_run_name


class ConfigTests(unittest.TestCase):
    def test_project_overrides_user(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            home_cfg = root / "user.toml"
            project = root / "project"
            project.mkdir()
            home_cfg.write_text('[execution]\ncommand=["user"]\n[slurm]\ntime_limit="12:00:00"\n', encoding="utf-8")
            (project / ".hummel-submit.toml").write_text('[execution]\ncommand=["project"]\n[slurm]\ntime_limit="04:00:00"\n', encoding="utf-8")
            with patch.object(config_mod, "USER_CONFIG", home_cfg):
                cfg, sources = load_config(project)
            self.assertEqual(cfg["execution"]["command"], ["project"])
            self.assertEqual(cfg["slurm"]["time_limit"], "04:00:00")
            self.assertEqual(sources, [home_cfg, project / ".hummel-submit.toml"])

    def test_environment_override_wins(self) -> None:
        with tempfile.TemporaryDirectory() as td, patch.object(config_mod, "USER_CONFIG", Path(td) / "missing"):
            project = Path(td) / "project"
            project.mkdir()
            (project / ".hummel-submit.toml").write_text('[execution]\ncommand=["train"]\n[slurm]\ntime_limit="04:00:00"\n', encoding="utf-8")
            with patch.dict(os.environ, {"HUMMEL_TIME_LIMIT": "02:00:00"}, clear=False):
                cfg, _ = load_config(project)
            self.assertEqual(cfg["slurm"]["time_limit"], "02:00:00")

    def test_checkpoint_cannot_escape_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td, patch.object(config_mod, "USER_CONFIG", Path(td) / "missing"):
            project = Path(td)
            (project / ".hummel-submit.toml").write_text('[execution]\ncommand=["train"]\ncheckpoint_glob="../*.ckpt"\n', encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(project)

    def test_reserved_sbatch_option_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td, patch.object(config_mod, "USER_CONFIG", Path(td) / "missing"):
            project = Path(td)
            (project / ".hummel-submit.toml").write_text('[execution]\ncommand=["train"]\n[slurm]\nextra_args=["--export=ALL"]\n', encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(project)

    def test_run_name_validation(self) -> None:
        self.assertEqual(validate_run_name("abc-1.2_x"), "abc-1.2_x")
        with self.assertRaises(ConfigError):
            validate_run_name("../../oops")


if __name__ == "__main__":
    unittest.main()
