#!/bin/bash
source /sw/batch/init.sh
set -euo pipefail

PYTHON="$1"
SNAPSHOT="$2"
STATE_PATH="$3"
HOP="$4"
export PYTHONPATH="$SNAPSHOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m hummel_submit.worker "$STATE_PATH" "$HOP"
