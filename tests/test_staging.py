from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from hummel_submit.staging import parse_stage_exclude, parse_stage_spec, stage_inputs


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

    def test_stage_directory_excludes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tree = root / "tree"
            tree.mkdir()
            (tree / "keep.txt").write_text("keep", encoding="utf-8")
            build = tree / "build"
            build.mkdir()
            (build / "large.o").write_text("object", encoding="utf-8")
            staged = stage_inputs(
                {"tree": tree},
                cache_dir=root / "cache",
                submission_id="sub",
                excludes={"tree": ["build"]},
            )
            staged_tree = Path(staged["tree"])
            self.assertEqual((staged_tree / "keep.txt").read_text(), "keep")
            self.assertFalse((staged_tree / "build").exists())

    def test_stage_exclude_spec(self) -> None:
        name, pattern = parse_stage_exclude("source=build")
        self.assertEqual(name, "source")
        self.assertEqual(pattern, "build")
        with self.assertRaises(ValueError):
            parse_stage_exclude("missing-equals")

    def test_unknown_stage_exclude_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            source.mkdir()
            with self.assertRaises(ValueError):
                stage_inputs(
                    {"source": source},
                    cache_dir=root / "cache",
                    submission_id="sub",
                    excludes={"other": ["build"]},
                )


if __name__ == "__main__":
    unittest.main()
