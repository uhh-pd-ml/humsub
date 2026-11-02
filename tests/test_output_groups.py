"""Output groups: when several inputs share a branch, a failure keeps the outputs of the inputs that were finished."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hummel_submit.make_manifest import build_manifest
from hummel_submit.manifest import ManifestError, load_manifest, output_groups
from hummel_submit.manifest_workflow import clear_incomplete_outputs


def _manifest(tmp_path: Path, branch: dict) -> Path:
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"schema": 1, "branches": [{"id": 0, **branch}]}))
    return path


def test_output_groups_are_parsed_and_default_to_one_group(tmp_path: Path) -> None:
    m = load_manifest(_manifest(tmp_path, {"outputs": ["/a/1", "/a/2", "/a/3"], "output_groups": [["/a/1", "/a/2"], ["/a/3"]]}))
    assert output_groups(m["branches"][0]) == [["/a/1", "/a/2"], ["/a/3"]]
    plain = load_manifest(_manifest(tmp_path, {"outputs": ["/a/1", "/a/2"]}))["branches"][0]
    assert "output_groups" not in plain and output_groups(plain) == [["/a/1", "/a/2"]]


@pytest.mark.parametrize("groups", [
    [["/a/1"]],                      # /a/2 in no group
    [["/a/1", "/a/2"], ["/a/2"]],    # /a/2 twice
    [["/a/1", "/a/2", "/a/9"]],      # /a/9 is not an output
    [["/a/1", "/a/2"], []],          # empty group
    [],                              # no groups
    "notalist",
    [["/a/1", 2]],
])
def test_output_groups_must_partition_the_outputs(tmp_path: Path, groups) -> None:
    with pytest.raises(ManifestError):
        load_manifest(_manifest(tmp_path, {"outputs": ["/a/1", "/a/2"], "output_groups": groups}))


def test_make_manifest_groups_the_outputs_per_input(tmp_path: Path) -> None:
    files = [tmp_path / f"in{i}.parquet" for i in range(3)]
    m = build_manifest(files, outputs=["{out}/{stem}.a", "{out}/{stem}.b"], files_per_branch=2, values={"out": "/o"})
    assert m["branches"][0]["output_groups"] == [["/o/in0.a", "/o/in0.b"], ["/o/in1.a", "/o/in1.b"]]
    assert m["branches"][1]["output_groups"] == [["/o/in2.a", "/o/in2.b"]]
    path = tmp_path / "m.json"
    path.write_text(json.dumps(m))
    assert load_manifest(path)["branches"][0]["output_groups"] == m["branches"][0]["output_groups"]


def _files(tmp_path: Path, names: list[str]) -> list[str]:
    for name in names:
        (tmp_path / name).write_text("x")
    return [str(tmp_path / n) for n in names]


def test_a_failure_keeps_the_complete_groups_and_removes_the_partial_ones(tmp_path: Path) -> None:
    # inputs 0 and 1 finished (both outputs), input 2 crashed after one of its two outputs, input 3 never started
    groups = [[str(tmp_path / f"{i}.{x}") for x in "ab"] for i in range(4)]
    _files(tmp_path, ["0.a", "0.b", "1.a", "1.b", "2.a"])
    clear_incomplete_outputs(groups)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["0.a", "0.b", "1.a", "1.b"]


def test_a_failure_with_every_group_complete_keeps_nothing(tmp_path: Path) -> None:
    # e.g. exit != 0 after writing everything: keeping it all would make the failed branch look complete
    groups = [_files(tmp_path, ["0.a"]), _files(tmp_path, ["1.a"])]
    clear_incomplete_outputs(groups)
    assert list(tmp_path.iterdir()) == []


def test_without_groups_a_failure_is_all_or_nothing_as_before(tmp_path: Path) -> None:
    outputs = [str(tmp_path / n) for n in ("0.a", "1.a", "2.a")]
    _files(tmp_path, ["0.a", "1.a"])
    clear_incomplete_outputs(output_groups({"outputs": outputs}))
    assert list(tmp_path.iterdir()) == []
