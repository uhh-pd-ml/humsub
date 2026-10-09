from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from hummel_submit.envfile import EnvFileError, load_env_file


class EnvFileTests(unittest.TestCase):
    def _load(self, text: str):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / ".env"
            path.write_text(text, encoding="utf-8")
            return load_env_file(path)

    def test_missing_file_is_empty(self) -> None:
        self.assertEqual(load_env_file(Path("/nonexistent/.env")), {})

    def test_subset_of_dotenv(self) -> None:
        env = self._load("# c\n\nA=1\nexport B = two\nC=\"x y\"\nD='$HOME'\nE=a=b\n")
        self.assertEqual(env, {"A": "1", "B": "two", "C": "x y", "D": "$HOME", "E": "a=b"})

    def test_no_shell_evaluation(self) -> None:
        self.assertEqual(self._load("X=$(touch /nonexistent)\n"), {"X": "$(touch /nonexistent)"})

    def test_errors_name_file_and_line(self) -> None:
        with self.assertRaisesRegex(EnvFileError, r":2: expected KEY=VALUE"):
            self._load("A=1\njunk\n")
        with self.assertRaisesRegex(EnvFileError, r":1: invalid environment variable name"):
            self._load("1BAD=x\n")


if __name__ == "__main__":
    unittest.main()
