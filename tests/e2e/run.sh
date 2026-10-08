#!/usr/bin/env bash
# Run the end-to-end tests inside the hummel-slurm-ci image (a Slurm configured
# like Hummel-2).  Needs Docker; the container runs --privileged.
#
#   tests/e2e/run.sh                      all e2e tests except the slow ones
#   tests/e2e/run.sh -m slow              only the slow ones (minutes: real time limits)
#   tests/e2e/run.sh -k roundtrip -x      any pytest arguments
#
# Environment:
#   IMAGE            image to use (default ghcr.io/uhh-pd-ml/hummel-slurm-ci:latest)
#   WHEELHOUSE       directory of wheels (law, luigi, pytest, setuptools, wheel and
#                    their dependencies); installs offline from it instead of from
#                    the network, e.g. when the container has no working DNS
#   KEEP_CONTAINER=1 leave the container running afterwards (docker exec -u testuser ...)
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"
IMAGE="${IMAGE:-ghcr.io/uhh-pd-ml/hummel-slurm-ci:latest}"
NAME="${CONTAINER_NAME:-humsub-e2e}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
mounts=(-v "$REPO:/src:ro")
[ -n "${WHEELHOUSE:-}" ] && mounts+=(-v "$WHEELHOUSE:/wheelhouse:ro")
docker run -d --privileged --init --name "$NAME" -h slurmctl "${mounts[@]}" "$IMAGE" >/dev/null
cleanup() { [ "${KEEP_CONTAINER:-0}" = 1 ] || docker rm -f "$NAME" >/dev/null 2>&1; }
trap cleanup EXIT

for _ in $(seq 1 180); do
    docker exec "$NAME" test -e /run/hummel-ready 2>/dev/null && break
    sleep 1
done
docker exec "$NAME" test -e /run/hummel-ready || { docker logs --tail 40 "$NAME"; echo "cluster not ready"; exit 1; }

# pytest arguments are passed through a file to survive quoting
printf '%s\0' "$@" | docker exec -i "$NAME" bash -c 'cat > /tmp/pytest-args'

docker exec -u testuser -w /beegfs/uu/testuser/testuser -e WHEELHOUSE="${WHEELHOUSE:+/wheelhouse}" "$NAME" bash -lc '
set -euo pipefail
V=$USW/venv
python3.12 -m venv "$V"
if [ -n "${WHEELHOUSE:-}" ]; then
    pipopts=(--no-index --find-links "$WHEELHOUSE")
    "$V/bin/pip" install -q "${pipopts[@]}" setuptools wheel law pytest
    pipopts+=(--no-build-isolation --no-deps)
else
    pipopts=()
    "$V/bin/pip" install -q pytest
fi
# build from a writable copy: /src is mounted read-only
rm -rf "$BEEGFS/src" && mkdir "$BEEGFS/src"
cp -r /src/pyproject.toml /src/README.md /src/src "$BEEGFS/src/"
"$V/bin/pip" install -q "${pipopts[@]}" "$BEEGFS/src"
rm -rf "$BEEGFS/tests" && cp -r /src/tests "$BEEGFS/tests"
cd "$BEEGFS"
mapfile -d "" args < /tmp/pytest-args
if [ ${#args[@]} -eq 0 ]; then args=(-m "not slow"); fi
export HUMSUB_E2E=1 PATH="$V/bin:$PATH"
"$V/bin/python" -m pytest tests/e2e -p no:cacheprovider -v "${args[@]}"
'
