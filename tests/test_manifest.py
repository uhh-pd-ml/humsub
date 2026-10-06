from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from hummel_submit.manifest import ManifestError, freeze_manifest_workflow, load_manifest
from hummel_submit.state import atomic_write_json


class ManifestTests(unittest.TestCase):
    def test_validate_and_sort(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "manifest.json"
            path.write_text(json.dumps({
                "schema": 1,
                "common": {"answer": 42},
                "branches": [
                    {"id": 2, "data": {"x": 2}, "outputs": [str(root / "b")]},
                    {"id": 0, "data": {"x": 0}, "outputs": [str(root / "a")]},
                ],
            }), encoding="utf-8")
            data = load_manifest(path)
            self.assertEqual([b["id"] for b in data["branches"]], [0, 2])

    def test_relative_output_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest.json"
            path.write_text(json.dumps({
                "schema": 1,
                "branches": [{"id": 0, "outputs": ["relative.root"]}],
            }), encoding="utf-8")
            with self.assertRaises(ManifestError):
                load_manifest(path)

    def test_freeze_writes_branch_context(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "input.json"
            output = root / "out.root"
            manifest.write_text(json.dumps({
                "schema": 1,
                "common": {"kind": "demo"},
                "branches": [{"id": 0, "data": {"value": 7}, "outputs": [str(output)]}],
            }), encoding="utf-8")
            payload = root / "payload.sh"
            payload.write_text("#!/bin/sh\n", encoding="utf-8")
            spec_dir = root / "submission"
            spec_dir.mkdir()
            spec_path = spec_dir / "submission.json"
            atomic_write_json(spec_path, {
                "submission_id": "abc",
                "run_name": "demo",
                "run_dir": str(root / "run"),
            })
            spec = freeze_manifest_workflow(
                spec_path,
                manifest_source=manifest,
                payload_source=payload,
                stages={"source": "/tmp/source"},
            )
            branch_file = Path(spec["manifest_workflow"]["branch_files"]["0"])
            context = json.loads(branch_file.read_text())
            self.assertEqual(context["common"]["kind"], "demo")
            self.assertEqual(context["data"]["value"], 7)
            self.assertEqual(context["stages"]["source"], "/tmp/source")


if __name__ == "__main__":
    unittest.main()
