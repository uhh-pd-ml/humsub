#!/usr/bin/env python3
"""Write manifest.json: one branch per number N; branch i computes sum(1..N).

usage: make_manifest.py OUTPUT_DIR [N ...]
OUTPUT_DIR must be on shared storage ($BEEGFS), e.g. $BEEGFS/humsub-examples/hello
"""
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1]).absolute()
numbers = [int(x) for x in sys.argv[2:]] or [10, 100, 1000, 10000]

manifest = {
    "schema": 1,
    # "common" is copied unchanged into every branch context
    "common": {"greeting": "hello"},
    "branches": [
        {
            "id": i,                                      # unique integer
            "data": {"n": n},                             # private to this branch
            "outputs": [str(out_dir / f"sum-{i}.json")],  # absolute paths; complete = all exist
        }
        for i, n in enumerate(numbers)
    ],
}
Path("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"wrote manifest.json with {len(numbers)} branches; outputs go to {out_dir}")
