from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from hummel_submit.staging import parse_stage_spec, stage_inputs


class StagingTests(unittest.TestCase):
    def test_stage_file_and_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            one = root / "one.txt"
            one.write_text("one", encoding="utf-8")
            tree = root / "tree"
            tree.mkdir()
            (tree / "two.txt").write_text("two", encoding="utf-8")
            staged = stage_inputs(
                {"file": one, "tree": tree},
                cache_dir=root / "cache",
                submission_id="sub",
            )
            self.assertEqual(Path(staged["file"]).read_text(), "one")
            self.assertEqual((Path(staged["tree"]) / "two.txt").read_text(), "two")

    def test_stage_spec(self) -> None:
        name, path = parse_stage_spec("source=./somewhere")
        self.assertEqual(name, "source")
        self.assertTrue(path.is_absolute())
        with self.assertRaises(ValueError):
            parse_stage_spec("missing-equals")


if __name__ == "__main__":
    unittest.main()
