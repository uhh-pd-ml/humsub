#!/bin/bash
source /sw/batch/init.sh
set -euo pipefail

PYTHON="$1"
SNAPSHOT="$2"
SPEC="$3"
ROUND="$4"
# The supervisor creates new chains, which needs the package as real files (worker.sh, ...), not from a zip.
CODE="${TMPDIR:-/tmp}/humsub-code-${SLURM_JOB_ID:-$$}"
"$PYTHON" -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$SNAPSHOT" "$CODE"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
trap 'rm -rf "$CODE"' EXIT
"$PYTHON" -m hummel_submit.supervisor "$SPEC" "$ROUND"
