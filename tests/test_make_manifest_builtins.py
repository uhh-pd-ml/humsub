from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from hummel_submit import cli
from hummel_submit.config import ConfigError
from hummel_submit.make_manifest import build_manifest, find_inputs
from hummel_submit.manifest import ManifestError, load_manifest
from hummel_submit.payloads import container, resolve


# --- command line -------------------------------------------------------------------------------------------------

def test_unknown_command_is_an_error_and_never_submits(capsys) -> None:
    with patch.object(cli, "cmd_submit") as submit:
        assert cli.main(["stat", "x"]) == 2
    submit.assert_not_called()
    err = capsys.readouterr().err
    assert "unknown command 'stat'" in err and "did you mean 'stats'" in err


def test_dash_dash_and_options_still_mean_submit() -> None:
    with patch.object(cli, "cmd_submit", return_value=0) as submit:
        assert cli.main(["--", "--epochs=3"]) == 0
        assert cli.main(["--dry-run", "--", "x"]) == 0
    assert submit.call_count == 2


def test_every_parser_command_is_dispatched() -> None:
    parser = cli._parser()
    names = {n for a in parser._subparsers._group_actions for n in a.choices}
    assert {"make-manifest", "stats", "doctor", "prune"} <= names


# --- manifest finalize section ------------------------------------------------------------------------------------

def _manifest(tmp_path: Path, **extra) -> Path:
    path = tmp_path / "m.json"
    data = {"schema": 1, "branches": [{"id": 0, "outputs": [str(tmp_path / "o0")]}], **extra}
    path.write_text(json.dumps(data))
    return path


def test_manifest_finalize_is_validated_and_kept(tmp_path: Path) -> None:
    m = load_manifest(_manifest(tmp_path, finalize={"data": {"x": 1}, "outputs": [str(tmp_path / "merged")]}))
    assert m["finalize"] == {"data": {"x": 1}, "outputs": [str(tmp_path / "merged")]}
    for bad in ({"outputs": []}, {"outputs": ["relative"]}, {"outputs": [str(tmp_path / "o0")]},
                {"outputs": [str(tmp_path / "x")], "typo": 1}, []):
        with pytest.raises(ManifestError):
            load_manifest(_manifest(tmp_path, finalize=bad))


# --- make-manifest ------------------------------------------------------------------------------------------------

def _tree(tmp_path: Path, names: list[str]) -> Path:
    raw = tmp_path / "raw"
    for i, name in enumerate(names):
        p = raw / f"sub{i % 2}" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    return raw


def test_find_inputs_require_and_select(tmp_path: Path) -> None:
    raw = _tree(tmp_path, ["A-1.parquet", "B-2.parquet", "A-3.parquet"])
    files, found = find_inputs([raw], "*.parquet", require=r"[AB]-\d\.parquet", select=r"A-\d\.parquet")
    assert found == 3 and [f.name for f in files] == ["A-1.parquet", "A-3.parquet"]
    with pytest.raises(ConfigError, match="do not match --require"):
        find_inputs([raw], "*.parquet", require=r"A-\d\.parquet")


def test_build_manifest_packs_files_and_adds_finalize(tmp_path: Path) -> None:
    files = [tmp_path / f"in{i}.parquet" for i in range(5)]
    m = build_manifest(files, outputs=["{out}/_m/{stem}.json"], files_per_branch=2, values={"out": "/b/o"},
                       common={"c": 1}, finalize_outputs=["{out}/manifest.json"], finalize_data={"command": ["merge"]})
    assert [len(b["data"]["inputs"]) for b in m["branches"]] == [2, 2, 1]
    assert m["branches"][2]["outputs"] == ["/b/o/_m/in4.json"]
    assert m["common"] == {"c": 1, "out": "/b/o"}
    assert m["finalize"] == {"data": {"command": ["merge"]}, "outputs": ["/b/o/manifest.json"]}


def test_build_manifest_rejects_colliding_outputs_and_unknown_fields(tmp_path: Path) -> None:
    files = [tmp_path / "a" / "x.parquet", tmp_path / "b" / "x.parquet"]
    with pytest.raises(ConfigError, match="would both write"):
        build_manifest(files, outputs=["/o/{stem}.json"])
    with pytest.raises(ConfigError, match="unknown field"):
        build_manifest(files[:1], outputs=["{nope}/{stem}.json"])


def test_make_manifest_command_writes_a_valid_manifest(tmp_path: Path, capsys) -> None:
    raw = _tree(tmp_path, ["A-1.parquet", "A-2.parquet", "A-3.parquet"])
    out = tmp_path / "manifest.json"
    rc = cli.main(["make-manifest", "--search", str(raw), "--glob", "*.parquet", "--output", "{out}/{stem}.json",
                   "--set", f"out={tmp_path / 'o'}", "--files-per-branch", "2", "--common-json", '{"command": ["x"]}',
                   "--finalize-output", "{out}/all.json", "-o", str(out)])
    assert rc == 0
    m = load_manifest(out)
    assert len(m["branches"]) == 2 and m["common"]["command"] == ["x"] and "finalize" in m
    assert "2 branch(es), 3 input file(s)" in capsys.readouterr().out


# --- builtin:container ----------------------------------------------------------------------------------------------

def test_builtin_payload_resolution() -> None:
    assert resolve("./payload.py") is None
    assert resolve("builtin:container").name == "container.py"
    with pytest.raises(ValueError, match="available: container"):
        resolve("builtin:nope")


def _ctx(**data) -> dict:
    return {
        "branch": 3, "run_dir": "/r", "outputs": ["/o/a.json", "/o/b.json"],
        "stages": {"code": "/ssd/stages/s1/code"},
        "common": {
            "out": "/o",
            "command": ["python", "conv.py", "{inputs}", "--out", "{out}"],
            "container": {"image": "/img.sif", "apptainer": "/bin/apptainer", "workdir": "{stage_code}",
                          "env": {"PYTHONPATH": "{stage_code}"}},
        },
        "data": data,
    }


def test_container_command_expands_inputs_fields_binds_and_env() -> None:
    cmd, env = container.build_command(
        _ctx(inputs=["/raw/a.parquet", "/raw/b.parquet"]),
        {"PYTHONPATH": "/worker.zip", "HUMSUB_SCRATCH": "/beegfs/s", "HUMSUB_SCRATCH_FAST": "/ssd/f", "HOME": "/h"},
    )
    assert cmd == [
        "/bin/apptainer", "exec", "--bind", "/beegfs,/ssd/stages/s1/code,/beegfs/s,/ssd/f",
        "--pwd", "/ssd/stages/s1/code", "--env", "PYTHONPATH=/ssd/stages/s1/code", "/img.sif",
        "python", "conv.py", "/raw/a.parquet", "/raw/b.parquet", "--out", "/o",
    ]
    assert "PYTHONPATH" not in env and env["HOME"] == "/h"


def test_container_branch_command_wins_and_outputs_expand() -> None:
    cmd, _ = container.build_command(_ctx(command=["merge", "{outputs}", "{branch}"], finalize=True), {})
    assert cmd[-4:] == ["merge", "/o/a.json", "/o/b.json", "3"]


def test_container_unknown_field_is_reported() -> None:
    with pytest.raises(KeyError, match="unknown field"):
        container.build_command(_ctx(command=["{typo}"]), {})
