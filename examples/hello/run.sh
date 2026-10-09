#!/usr/bin/env bash
# End-to-end demo; run from this directory on a Hummel frontend:  ./run.sh [RUN_NAME]
set -euo pipefail
cd "$(dirname "$0")"
RUN="${1:-hello-$(date +%H%M%S)}"
OUT="$BEEGFS/humsub-examples/$RUN"

mkdir -p lookup && echo 3 > lookup/factor.txt             # something to stage
python3 make_manifest.py "$OUT"                           # 1. manifest
humsub submit-manifest --manifest manifest.json --payload ./payload.py \
    --stage lookup=./lookup --run-name "$RUN"             # 2. stage + submit (returns immediately)
