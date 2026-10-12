from __future__ import annotations

import json
from pathlib import Path

import pytest

from hummel_submit.chain_runner import hop_timing_env
from hummel_submit.config import DEFAULTS, ConfigError, validate_config
from hummel_submit.manifest import ManifestError, load_manifest
from hummel_submit.manifest_submit import SubmitOptions
from hummel_submit.slurm import base_sbatch_args, sbatch_command
import copy


def _state(**slurm) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    cfg["slurm"].update(partition="std", gpus=0, account="a", time_limit="03:00:00", signal_seconds=600, **slurm)
    return {"project_dir": "/p", "output_dir": "/o", "config": cfg, "resubmit": True,
            "snapshot_path": "/s.zip", "worker_script": "/w.sh", "python_executable": "/py"}


def test_nice_is_on_by_default_and_visible_in_sbatch_args() -> None:
    assert "--nice=1000000" in base_sbatch_args(_state())
    assert not any(a.startswith("--nice") for a in base_sbatch_args(_state(nice=0)))


def test_nice_is_validated_and_reserved() -> None:
    cfg = copy.deepcopy(DEFAULTS)
    cfg["slurm"]["nice"] = -1
    with pytest.raises(ConfigError, match="nice"):
        validate_config(cfg, require_command=False)
    cfg["slurm"]["nice"] = 5
    cfg["slurm"]["extra_args"] = ["--nice=3"]
    with pytest.raises(ConfigError, match="managed by humsub"):
        validate_config(cfg, require_command=False)


def test_scratch_kind_is_validated() -> None:
    cfg = copy.deepcopy(DEFAULTS)
    cfg["execution"]["scratch"] = "nfs"
    with pytest.raises(ConfigError, match="scratch"):
        validate_config(cfg, require_command=False)


def test_lane_dependency_only_on_first_hop() -> None:
    state = _state()
    state["lane_after"] = "777"
    first = sbatch_command(state, Path("/st.json"), 0)
    assert "--dependency=afterany:777" in first
    follower = sbatch_command(state, Path("/st.json"), 1, dependency="900")
    assert "--dependency=afterany:900" in follower and "--dependency=afterany:777" not in follower


def test_signal_is_requested_when_only_failure_hops_exist() -> None:
    state = _state(max_hops=1)
    assert not any(a.startswith("--signal") for a in base_sbatch_args(state))
    state["failure_budget"] = 1
    assert any(a.startswith("--signal") for a in base_sbatch_args(state))


def test_safety_margin_excludes_other_retry_mechanisms() -> None:
    SubmitOptions(safety_margin=0.1).validate()
    with pytest.raises(ConfigError, match="mutually exclusive"):
        SubmitOptions(safety_margin=0.1, retry_payload=2).validate()
    with pytest.raises(ConfigError, match="mutually exclusive"):
        SubmitOptions(safety_margin=0.1, retries=1, wait=True).validate()
    with pytest.raises(ConfigError, match="no-resubmit"):
        SubmitOptions(safety_margin=0.1, resubmit=False).validate()


def test_failure_budget_scales_with_chain_length() -> None:
    assert SubmitOptions().failure_budget() == 0
    assert SubmitOptions(safety_margin=0.1, tasks_per_job=1).failure_budget() == 1
    assert SubmitOptions(safety_margin=0.1, tasks_per_job=4).failure_budget() == 1
    assert SubmitOptions(safety_margin=0.5, tasks_per_job=4).failure_budget() == 2
    assert SubmitOptions(safety_margin=0.1, tasks_per_job=25).failure_budget() == 3


def test_other_option_validation() -> None:
    with pytest.raises(ConfigError):
        SubmitOptions(retries=1).validate()  # needs --wait
    with pytest.raises(ConfigError):
        SubmitOptions(max_concurrent=-1).validate()
    with pytest.raises(ConfigError):
        SubmitOptions(scratch="nfs").validate()
    with pytest.raises(ConfigError):
        SubmitOptions(supervisor_job=True, supervisor_interval=5).validate()


def test_options_roundtrip_ignores_unknown_keys() -> None:
    opts = SubmitOptions(retry_payload=2, max_concurrent=50, scratch="ssd")
    again = SubmitOptions.from_dict({**opts.to_dict(), "future_option": 1})
    assert again == opts


def test_hop_timing_env_accounts_for_signal_and_failure_hops() -> None:
    state = _state(max_hops=1)
    env = hop_timing_env(state, now=100.0)
    assert env["HUMSUB_HOP_USABLE_SECONDS"] == str(3 * 3600)  # single hop: whole limit
    state = _state(max_hops=4)
    assert hop_timing_env(state, now=100.0)["HUMSUB_HOP_USABLE_SECONDS"] == str(3 * 3600 - 600)
    state = _state(max_hops=1)
    state["failure_budget"] = 1
    assert hop_timing_env(state, now=100.0)["HUMSUB_HOP_USABLE_SECONDS"] == str(3 * 3600 - 600)


def test_manifest_hint_validation(tmp_path: Path) -> None:
    def write(extra: dict) -> Path:
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"schema": 1, "branches": [{"id": 0, "outputs": ["/abs/o"]}], **extra}))
        return path

    assert load_manifest(write({"humsub": {"scratch": "ssd", "scratch_bytes_per_branch": 5}}))["humsub"]["scratch"] == "ssd"
    assert "humsub" not in load_manifest(write({}))
    for bad in ({"scratch": "tape"}, {"scratch_bytes_per_branch": -1}, {"nope": 1}):
        with pytest.raises(ManifestError):
            load_manifest(write({"humsub": bad}))
