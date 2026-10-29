#!/usr/bin/env bash
# Run the end-to-end tests inside the hummel-slurm-ci image (a Slurm configured
# like Hummel-2).  Needs Docker; the container runs --privileged.
#
#   tests/e2e/run.sh                      all e2e tests except the slow ones
#   tests/e2e/run.sh -m slow              only the slow ones (minutes: real time limits)
#   tests/e2e/run.sh -k roundtrip -x      any pytest arguments
#
# Tests run in parallel (pytest-xdist, E2E_WORKERS workers); every test submits under its own
# Slurm job name, so tests only see and cancel their own jobs.
#
# Environment:
#   IMAGE            image to use (default ghcr.io/uhh-pd-ml/hummel-slurm-ci:latest)
#   WHEELHOUSE       directory of wheels (law, luigi, pytest, pytest-xdist, setuptools, wheel and
#                    their dependencies); installs offline from it instead of from
#                    the network.  If unset, one is built on the *host* (which has network)
#                    into ${XDG_CACHE_HOME:-~/.cache}/humsub-e2e-wheelhouse-<law commit> on first
#                    use and re-used afterwards -- containers usually have no working DNS.
#                    NO_WHEELHOUSE=1 installs from the network inside the container instead.
#   E2E_WORKERS      parallel pytest workers (default 4; 0 = serial).  The simulated cluster has
#                    16 CPUs in `std`; many more workers make hops queue and slow tests time out.
#   DOCKER_RUN_FLAGS extra `docker run` flags, e.g. --cpus=2 to mimic a small CI runner
#   KEEP_CONTAINER=1 leave the container running afterwards (docker exec -u testuser ...)
#   COVERAGE=1       measure coverage of hummel_submit in *all* processes (law, chain
#                    hops, payload workers) and run the unit tests in the same container,
#                    so the report is the combined unit + e2e coverage.  Writes
#                    .coverage-e2e/ (data, coverage.txt, html/) next to this repository.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"
# the image: IMAGE, else a locally available hummel-slurm-ci, else the one on ghcr.io
if [ -z "${IMAGE:-}" ]; then
    if docker image inspect hummel-slurm-ci:latest >/dev/null 2>&1; then IMAGE=hummel-slurm-ci:latest
    else IMAGE=ghcr.io/uhh-pd-ml/hummel-slurm-ci:latest; fi
fi
# the wheelhouse: build it on the host once (see above)
# law exactly as pinned in pyproject.toml; the wheelhouse is keyed by the pinned commit
LAW_REQ=$(sed -n 's/^ *"\(law @ [^"]*\)",*$/\1/p' pyproject.toml)
[ -n "$LAW_REQ" ] || { echo "no law requirement found in pyproject.toml"; exit 1; }
if [ -z "${WHEELHOUSE:-}" ] && [ "${NO_WHEELHOUSE:-0}" != 1 ]; then
    WHEELHOUSE="${XDG_CACHE_HOME:-$HOME/.cache}/humsub-e2e-wheelhouse-${LAW_REQ##*@}"
    WHEELS=("$LAW_REQ" pytest pytest-xdist setuptools wheel coverage)
    # (re)build when the wanted set changed, e.g. after adding a test dependency
    if [ "$(cat "$WHEELHOUSE/.requirements" 2>/dev/null)" != "${WHEELS[*]}" ]; then
        echo "building the wheelhouse in $WHEELHOUSE ..."
        mkdir -p "$WHEELHOUSE"
        python3 -m pip wheel -q -w "$WHEELHOUSE" "${WHEELS[@]}"
        echo "${WHEELS[*]}" > "$WHEELHOUSE/.requirements"
    fi
fi
NAME="${CONTAINER_NAME:-humsub-e2e}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
mounts=(-v "$REPO:/src:ro")
[ -n "${WHEELHOUSE:-}" ] && mounts+=(-v "$WHEELHOUSE:/wheelhouse:ro")
docker run -d --privileged --init --name "$NAME" -h slurmctl ${DOCKER_RUN_FLAGS:-} "${mounts[@]}" "$IMAGE" >/dev/null
cleanup() { [ "${KEEP_CONTAINER:-0}" = 1 ] || docker rm -f "$NAME" >/dev/null 2>&1; }
fetch_coverage() {
    [ "${COVERAGE:-0}" = 1 ] || return 0
    rm -rf "$REPO/.coverage-e2e"
    docker cp "$NAME:/beegfs/uu/testuser/testuser/cov" "$REPO/.coverage-e2e" 2>/dev/null \
        && echo "coverage written to $REPO/.coverage-e2e (coverage.txt, html/)"
}
trap 'fetch_coverage; cleanup' EXIT

for _ in $(seq 1 180); do
    docker exec "$NAME" test -e /run/hummel-ready 2>/dev/null && break
    sleep 1
done
docker exec "$NAME" test -e /run/hummel-ready || { docker logs --tail 40 "$NAME"; echo "cluster not ready"; exit 1; }

# pytest arguments are passed through a file to survive quoting
printf '%s\0' "$@" | docker exec -i "$NAME" bash -c 'cat > /tmp/pytest-args'

docker exec -u testuser -w /beegfs/uu/testuser/testuser -e WHEELHOUSE="${WHEELHOUSE:+/wheelhouse}" -e COVERAGE="${COVERAGE:-0}" -e E2E_WORKERS="${E2E_WORKERS:-4}" "$NAME" bash -lc '
set -euo pipefail
V=$USW/venv
python3.12 -m venv "$V"
if [ -n "${WHEELHOUSE:-}" ]; then
    pipopts=(--no-index --find-links "$WHEELHOUSE")
    "$V/bin/pip" install -q "${pipopts[@]}" setuptools wheel law pytest pytest-xdist
    pipopts+=(--no-build-isolation --no-deps)
else
    pipopts=()
    "$V/bin/pip" install -q pytest pytest-xdist
fi
# build from a writable copy: /src is mounted read-only
rm -rf "$BEEGFS/src" && mkdir "$BEEGFS/src"
cp -r /src/pyproject.toml /src/README.md /src/src "$BEEGFS/src/"
"$V/bin/pip" install -q "${pipopts[@]}" "$BEEGFS/src"
rm -rf "$BEEGFS/tests" && cp -r /src/tests "$BEEGFS/tests"
cd "$BEEGFS"
mapfile -d "" args < /tmp/pytest-args
if [ ${#args[@]} -eq 0 ]; then args=(-m "not slow"); fi
if [ "$E2E_WORKERS" != 0 ]; then args=(-n "$E2E_WORKERS" "${args[@]}"); fi
export HUMSUB_E2E=1 PATH="$V/bin:$PATH"
target=tests/e2e
if [ "${COVERAGE:-0}" = 1 ]; then
    target=tests
    cov=$BEEGFS/cov; rm -rf "$cov"; mkdir -p "$cov"
    "$V/bin/pip" install -q ${WHEELHOUSE:+--no-index --find-links "$WHEELHOUSE"} coverage
    # Slurm jobs start with --export=NONE, so the config path is baked into a .pth file
    # that starts coverage in every Python process of this venv.
    cat > "$cov/coveragerc" <<RC
[run]
branch = True
parallel = True
sigterm = True
data_file = $cov/data
source = hummel_submit
[paths]
src =
    $BEEGFS/src/src/hummel_submit
    $V/lib/python3.12/site-packages/hummel_submit
    */hummel-submit-worker.zip/hummel_submit
RC
    printf "import os; os.environ.setdefault(\"COVERAGE_PROCESS_START\", \"$cov/coveragerc\"); import coverage; coverage.process_startup()\n" \
        > "$V/lib/python3.12/site-packages/zz_coverage.pth"
fi
set +e
"$V/bin/python" -m pytest "$target" -p no:cacheprovider -v "${args[@]}"
rc=$?
if [ "${COVERAGE:-0}" = 1 ]; then
    rm -f "$V/lib/python3.12/site-packages/zz_coverage.pth"
    ( cd "$cov" && "$V/bin/python" -m coverage combine --rcfile="$cov/coveragerc" >/dev/null 2>&1
      "$V/bin/python" -m coverage report --rcfile="$cov/coveragerc" -m --skip-empty > "$cov/coverage.txt"
      "$V/bin/python" -m coverage html --rcfile="$cov/coveragerc" -d "$cov/html" --skip-empty >/dev/null 2>&1
      cat "$cov/coverage.txt" | cut -c1-200 )
fi
exit $rc
'
