#!/usr/bin/env python3
"""Payload: humsub runs this once per branch as `payload.py BRANCH_CONTEXT.json`."""
import json
import os
from pathlib import Path
import sys

context = json.loads(Path(sys.argv[1]).read_text())
n = context["data"]["n"]                                  # this branch's data
lookup = Path(context["stages"]["lookup"]) / "factor.txt"  # a staged (frozen) input file
factor = int(lookup.read_text())
final = Path(context["outputs"][0])

print(f'{context["common"]["greeting"]} from branch {context["branch"]} '
      f'(attempt {os.environ["HUMSUB_ATTEMPT"]}, hop {os.environ.get("HUMSUB_HOP", "?")}, '
      f'job {os.environ.get("SLURM_JOB_ID", "-")})', flush=True)

result = {"n": n, "sum": factor * n * (n + 1) // 2, "factor": factor}

# Write beside the final path, then rename: a half-written file must never look like a
# finished output.  (HUMSUB_SCRATCH is for big temporaries; it is on the SSD, a different
# filesystem than BeeGFS, so it cannot be renamed into place -- copy from it instead.)
final.parent.mkdir(parents=True, exist_ok=True)
tmp = final.with_name(f".{final.name}.tmp")
tmp.write_text(json.dumps(result) + "\n")
os.replace(tmp, final)
print("wrote", final, flush=True)
