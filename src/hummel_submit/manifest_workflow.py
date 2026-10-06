from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import law

from .contrib.hummel import HummelWorkflow
from .submission import load_submission_spec


_ENV_KEY_RE = re.compile(r"[^A-Za-z0-9]+")


def _stage_env_name(name: str) -> str:
    return "HUMSUB_STAGE_" + _ENV_KEY_RE.sub("_", name).strip("_").upper()


class ManifestPayloadWorkflow(HummelWorkflow, law.LocalWorkflow):
    """Generic law workflow that maps a frozen JSON manifest to an executable payload.

    Each law branch receives one immutable JSON context file.  The payload is
    application-owned; humsub owns branch scheduling, staging, scratch, retries,
    autonomous Slurm continuation, and output-completeness integration with law.
    """

    def _spec(self):
        return load_submission_spec(Path(str(self.humsub_spec)))

    def _workflow(self):
        spec = self._spec()
        workflow = spec.get("manifest_workflow")
        if not isinstance(workflow, dict):
            raise RuntimeError("submission has no manifest_workflow configuration")
        return workflow

    def _manifest(self):
        return json.loads(Path(self._workflow()["manifest"]).read_text(encoding="utf-8"))

    def create_branch_map(self):
        return {int(branch["id"]): branch for branch in self._manifest()["branches"]}

    def output(self):
        targets = [law.LocalFileTarget(path) for path in self.branch_data["outputs"]]
        return targets[0] if len(targets) == 1 else targets

    def run(self):
        spec = self._spec()
        workflow = self._workflow()
        branch_id = int(self.branch)
        branch_file = Path(workflow["branch_files"][str(branch_id)])
        payload = Path(workflow["payload"])

        targets = self.output()
        target_list = targets if isinstance(targets, list) else [targets]
        # A branch is only executed when law considers it incomplete.  Remove any
        # partial subset left by a previous failed attempt before invoking the
        # application payload again.
        for target in target_list:
            Path(target.path).unlink(missing_ok=True)

        slurm_id = os.environ.get("SLURM_JOB_ID", "local")
        scratch = (
            Path(spec["config"]["execution"]["cache_dir"])
            / "payload-work"
            / spec["submission_id"]
            / f"{slurm_id}-{branch_id}"
        )
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.update({
            "HUMSUB_SUBMISSION_ID": spec["submission_id"],
            "HUMSUB_RUN_NAME": spec["run_name"],
            "HUMSUB_RUN_DIR": spec["run_dir"],
            "HUMSUB_BRANCH": str(branch_id),
            "HUMSUB_BRANCH_FILE": str(branch_file),
            "HUMSUB_SCRATCH": str(scratch),
            "HUMSUB_PAYLOAD": str(payload),
            "HUMSUB_ATTEMPT": os.environ.get("LAW_JOB_ATTEMPT", "1"),
        })
        for name, path in workflow.get("stages", {}).items():
            env[_stage_env_name(name)] = str(path)

        print(f"[humsub] branch={branch_id} context={branch_file}", flush=True)
        print(f"[humsub] payload={payload}", flush=True)
        try:
            proc = subprocess.run(
                [str(payload), str(branch_file)],
                cwd=spec["project_dir"],
                env=env,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"payload failed with exit code {proc.returncode}")

            missing = [Path(target.path) for target in target_list if not Path(target.path).exists()]
            if missing:
                raise RuntimeError(
                    "payload returned success but did not materialize declared output(s): "
                    + ", ".join(map(str, missing))
                )
        except Exception:
            for target in target_list:
                Path(target.path).unlink(missing_ok=True)
            raise
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
