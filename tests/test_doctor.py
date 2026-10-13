from __future__ import annotations

import copy
import stat
from pathlib import Path
from types import SimpleNamespace

import hummel_submit.doctor as doctor
from hummel_submit.config import DEFAULTS
from hummel_submit.slurm import SlurmJobStatus


def _config(tmp: Path, **slurm) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    cfg["execution"].update(output_dir=str(tmp / "out"), cache_dir=str(tmp / "ssd"))
    cfg["slurm"].update(account="acct", partition="std", gpus=0, **slurm)
    return cfg


def test_probe_script_checks_paths_nested_sbatch_and_imports() -> None:
    script = doctor.probe_script([Path("/beegfs/g/proj/out"), Path("/ssd/cache")], "/venv/bin/python", "acct", "std", 0)
    assert "source /sw/batch/init.sh" in script
    for needle in ("/beegfs/g/proj/out", "/ssd/cache", "sbatch --parsable --hold --account=acct --partition=std", "scancel", "/venv/bin/python -c", "import law, hummel_submit"):
        assert needle in script, needle
    assert "--gpus" not in script
    assert "--gpus=2" in doctor.probe_script([Path("/x")], "python", "acct", "gpu", 2), "the probe validates the GPU request too"
    assert "bash" not in script.splitlines()[0] or script.startswith("#!/bin/bash")


def test_probe_log_parsing() -> None:
    log = "noise\nDOCTOR info batch node: n001, now\nDOCTOR ok writable from the batch node: /x\nDOCTOR fail nested sbatch: a batch job cannot submit jobs (denied)\n"
    checks = doctor.parse_probe_log(log)
    assert [(c.status, c.name) for c in checks] == [("info", "probe: batch node"), ("ok", "probe: writable from the batch node"), ("fail", "probe: nested sbatch")]
    assert "denied" in checks[2].detail


def test_local_checks_report_problems_without_a_cluster(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(doctor, "slurm_command", lambda name: "/bin/true")
    monkeypatch.setattr(doctor, "quota_usage", lambda p: (60 * 2**30, 100 * 2**30))

    def fake_check(path, label, must_be_shared=False):
        if "bad" in str(path):
            raise doctor.PathCheckError("not writable")
        return SimpleNamespace(status="ok", resolved=path, detail="writable")
    monkeypatch.setattr(doctor, "check_compute_writable", fake_check)
    checks = doctor.local_checks(_config(tmp_path, nice=0), [Path("/bad/place"), Path("/good/place")])
    by_name = {c.name: c for c in checks}
    assert by_name["output_dir"].status == "ok" and by_name["scheduler"].status == "ok"
    assert by_name["path /bad/place"].status == "fail" and by_name["path /good/place"].status == "ok"
    assert by_name["SSD quota"].status == "warn" and "60 %" in by_name["SSD quota"].detail
    assert by_name["nice"].status == "warn"
    assert by_name["law"].status == "ok"


def _fake_sbatch(tmp: Path, reply: str, code: int = 0) -> Path:
    path = tmp / "sbatch"
    path.write_text(f"#!/bin/sh\necho '{reply}'\nexit {code}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_probe_end_to_end_with_a_fake_scheduler(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "out"
    monkeypatch.setattr(doctor, "slurm_command", lambda name: str(_fake_sbatch(tmp_path, "4242;cluster")))
    monkeypatch.setattr(doctor, "query_jobs", lambda ids: {"4242": SlurmJobStatus("4242", "COMPLETED", 0)})
    (out / "logs").mkdir(parents=True)
    (out / "logs" / "humsub-doctor_4242.log").write_text("DOCTOR ok writable from the batch node: /out\nDOCTOR ok nested sbatch: fine\n")
    checks = doctor.run_probe(_config(tmp_path), out, [out], timeout=20)
    assert [c.status for c in checks] == ["ok", "ok"]
    assert not list((out / "logs").glob(".doctor-probe-*")), "the probe script is removed"


def test_probe_reports_a_refused_submission(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(doctor, "slurm_command", lambda name: str(_fake_sbatch(tmp_path, "Invalid account", 1)))
    checks = doctor.run_probe(_config(tmp_path), tmp_path / "out", [tmp_path], timeout=5)
    assert checks[0].status == "fail" and "refused" in checks[0].detail


def test_probe_that_never_starts_is_a_warning_not_a_failure(tmp_path: Path, monkeypatch) -> None:
    cancelled = []
    monkeypatch.setattr(doctor, "slurm_command", lambda name: str(_fake_sbatch(tmp_path, "7")))
    monkeypatch.setattr(doctor, "query_jobs", lambda ids: {"7": SlurmJobStatus("7", "PENDING")})
    monkeypatch.setattr(doctor, "cancel_jobs", lambda ids: cancelled.append(ids))
    monkeypatch.setattr(doctor.time, "sleep", lambda s: None)
    t = iter(range(0, 1000, 10))
    monkeypatch.setattr(doctor.time, "time", lambda: next(t))
    checks = doctor.run_probe(_config(tmp_path), tmp_path / "out", [tmp_path], timeout=30)
    assert checks[0].status == "warn" and cancelled == [["7"]]


def test_probe_uses_nice_zero_unless_asked(tmp_path: Path, monkeypatch) -> None:
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return SimpleNamespace(stdout="1\n", stderr="")
    monkeypatch.setattr(doctor.subprocess, "run", fake_run)
    monkeypatch.setattr(doctor, "query_jobs", lambda ids: {"1": SlurmJobStatus("1", "COMPLETED", 0)})
    doctor.run_probe(_config(tmp_path), tmp_path / "out", [tmp_path], timeout=5)
    assert not any(a.startswith("--nice") for a in seen["cmd"])
    doctor.run_probe(_config(tmp_path), tmp_path / "out", [tmp_path], timeout=5, keep_nice=True)
    assert "--nice=1000000" in seen["cmd"]
