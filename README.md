# hummel-submit (`humsub`)

`humsub` runs applications on the Hummel-2 Slurm cluster as **autonomous job chains**: a job that
would exceed the partition's time limit is stopped shortly before the limit and continued in a
follow-up Slurm job, without you resubmitting it.  Workflows are expressed with
[law](https://github.com/riga/law); `humsub` is law's batch-system backend for Hummel plus a small
command line.

Two ways to use it:

| mode | command | for |
|---|---|---|
| **manifest** (recommended) | `humsub submit-manifest` | many independent units of work ("branches"), each producing declared output files, run by *your* payload script |
| **single command** | `humsub submit -- ARGS` | one long restartable command (e.g. GPU training that resumes from checkpoints), optionally in an Apptainer container |

`humsub` owns scheduling, staging of inputs, branch scratch, Slurm submission, continuation,
status/log following and cleanup.  Your payload owns what a branch *does*.  Nothing in `humsub`
knows about any particular science application.

## Terminology

| term | meaning |
|---|---|
| **submission** | one `submit`/`submit-manifest` invocation (id `YYYYMMDD-HHMMSS-xxxxxxxx`); owns frozen manifest, payload, staged inputs |
| **branch** | logical law unit of work: one manifest entry with its data and declared outputs |
| **chain** | stable identity (chain id) of one law remote job across continuation hops; what `status`, `follow`, `cancel` take |
| **job** | an actual Slurm job (Slurm job id).  A chain consists of one or more jobs |
| **hop** | one Slurm allocation in a chain (hop 0 = first job) |
| **continuation** | the *expected* extension of the same logical execution: a hop is stopped by the pre-timeout signal and the next hop re-runs the same law job |
| **retry** | a *new attempt after a failure* (law-level, `--wait --retries N`); a failed chain does not continue |
| **stage** | named input file/directory frozen (copied) at submission, so later edits do not affect running jobs |

`--tasks-per-job K` runs K branches one after another inside one chain (default 1: one chain per branch).

## Install

Needs Python ≥ 3.11 and `rsync` (present on Hummel).  Install software under `$USW`:

```bash
python3 -m venv "$USW/venvs/hummel-submit"
"$USW/venvs/hummel-submit/bin/pip" install .          # also installs law (master branch) from GitHub
mkdir -p "$HOME/.local/bin"
ln -sf "$USW/venvs/hummel-submit/bin/humsub" "$HOME/.local/bin/humsub"
humsub --version
```

The venv's interpreter and its `law` executable must be visible at the same absolute path on
the compute nodes (`$USW` is; it is read-only there, which is fine).  The install is a copy:
**re-run `pip install --no-deps .` after changing the source**.  Chains already submitted keep the
frozen worker they were started with.

## Quickstart (no CMS, no GPU)

[`examples/hello`](examples/hello) is a complete minimal application: four branches, each computes
`factor * n(n+1)/2` using a staged input file, and writes one JSON file.

```bash
cd examples/hello            # contains .hummel-submit.toml, make_manifest.py, payload.py, run.sh
./run.sh my-first-run
```

which does the following (edit `account` in `.hummel-submit.toml` first):

```bash
# 1. manifest: which branches exist, with which data, producing which files
python3 make_manifest.py "$BEEGFS/humsub-examples/my-first-run"      # -> manifest.json
# 2. stage the input and submit (returns as soon as the Slurm jobs are queued)
humsub submit-manifest --manifest manifest.json --payload ./payload.py \
       --stage lookup=./lookup --run-name my-first-run
```

Then:

```bash
humsub submission-status SUBMISSION_ID   # one line per chain: pending / running / finished / failed
humsub status CHAIN_ID                   # jobs, queue state, log path of every hop
humsub follow CHAIN_ID                   # tail -f the active log, switching to the next hop automatically
cat "$BEEGFS/humsub-examples/my-first-run"/sum-*.json            # the outputs, where the manifest said
humsub cleanup SUBMISSION_ID             # delete the staged inputs on the SSD once finished
```

`SUBMISSION_ID` and the chain ids are printed by `submit-manifest`.

## The manifest

```json
{
  "schema": 1,
  "common": {"anything": "shared by all branches"},
  "branches": [
    {"id": 0, "data": {"n": 10}, "outputs": ["/beegfs/.../out-0.json"]},
    {"id": 1, "data": {"n": 100}, "outputs": ["/beegfs/.../out-1.json", "/beegfs/.../out-1.log"]}
  ]
}
```

* `schema` must be `1`.  `common` is optional, free-form JSON.
* `branches` is a non-empty list; `id` is a unique non-negative integer; `data` is optional free-form JSON.
* `outputs` is a non-empty list of **absolute paths on shared, batch-writable storage** (`$BEEGFS`, or `$SSD`; `/home`, `/usw`,
  `/tmp` and `/dev/shm` are rejected at submission).  **A branch is complete exactly when all of its outputs
  exist.**  This is law's completeness test, so it also makes submission idempotent: re-submitting a
  manifest launches only branches with missing outputs, and says "nothing to submit" if none are missing.

## The payload

`--payload` is any executable (a script with a shebang is fine).  `humsub` copies it into the
submission and runs it once per branch, in the project directory, as

```text
payload BRANCH_CONTEXT.json
```

`BRANCH_CONTEXT.json` is frozen at submission and contains:

```json
{"schema": 1, "submission_id": "…", "run_name": "…", "run_dir": "…", "branch": 3,
 "common": {…}, "data": {…}, "outputs": ["/abs/out-3.json"], "stages": {"lookup": "/staged/path"}}
```

Environment (in addition to what the Hummel batch environment provides; jobs start from `--export=NONE`):

| variable | value |
|---|---|
| `HUMSUB_BRANCH_FILE` | path of the context JSON (= `argv[1]`) |
| `HUMSUB_BRANCH` | branch id |
| `HUMSUB_SUBMISSION_ID`, `HUMSUB_RUN_NAME`, `HUMSUB_RUN_DIR` | identity of the submission; `RUN_DIR` = `<output_dir>/runs/<run>` (created, empty) |
| `HUMSUB_SCRATCH` | private empty directory for this branch, under `cache_dir` on the SSD; **removed when the branch ends** (success or failure) |
| `HUMSUB_STAGE_<NAME>` | staged path of `--stage NAME=…` (name upper-cased, non-alphanumerics → `_`) |
| `HUMSUB_PAYLOAD` | path of the frozen payload |
| `HUMSUB_ATTEMPT` | law attempt number (1 on the first try, +1 per `--retries`) |
| `HUMSUB_CHAIN_ID`, `HUMSUB_HOP`, `HUMSUB_STATE_PATH` | chain id, hop number (0-based) and state file of the surrounding chain |

Rules:

1. **Exit 0 and create every declared output**, otherwise the branch fails.  Before each run
   `humsub` deletes any outputs left by a failed attempt, and after a failure it deletes whatever the payload
   wrote to the declared paths, so a partial output cannot make a retry look complete.
2. **Write atomically** (temporary name next to the target, then rename): outputs on BeeGFS and scratch on the SSD
   are different filesystems, so copy from scratch to a temporary name beside the target and rename that.
3. **Be restartable.**  A hop ends with SIGTERM to the payload's process group shortly before the time limit; the
   next hop re-runs the whole branch from the beginning (branches that already completed are skipped, a
   branch that was running is restarted).  Branches should therefore be much shorter than `time_limit`, or
   checkpoint on their own.
4. **Containers belong inside the payload** (law and the chain run on the host).  A payload that needs a
   runtime calls it itself, e.g. `cmsexec …` or `apptainer exec …`.
5. `$HOME` and `$USW` are read-only in batch; write to `$BEEGFS`, `$SSD` or `HUMSUB_SCRATCH`.

### Stages

`--stage NAME=PATH` (repeatable) copies a file or directory into `<cache_dir>/stages/<submission>/NAME`
(immutable for the life of the submission) and exposes it as `context["stages"]["NAME"]` and
`HUMSUB_STAGE_NAME`.  `--stage-exclude NAME=PATTERN` (repeatable) skips matching files while copying a
directory; patterns use `rsync --exclude` syntax, so a leading `/` anchors at the stage root:

```bash
--stage src=$HOME/my-source --stage-exclude 'src=/build/' --stage-exclude 'src=*.root'
```

Stages live on the SSD and are kept after the run so failures stay reproducible; remove them with
`humsub cleanup` / `humsub gc` (below).

## Configuration

Settings come from TOML files, lowest to highest precedence: built-in defaults → `~/.config/hummel-submit/config.toml`
→ `./.hummel-submit.toml` (current directory) → `HUMMEL_*` environment variables → command-line options.
`humsub init` writes annotated templates (never overwrites); `humsub config` shows the resolved result.

```toml
[execution]
output_dir = "${BEEGFS}/jobs"                  # run state, chains, logs, runs/<run>; shared + persistent
cache_dir  = "${SSD}/.hummel-submit/cache"     # stages, branch scratch; SSD
# single-command mode only:
image = "none"                                 # Apptainer image or "none"
command = []                                   # argv prefix, e.g. ["python", "train.py"]
auto_args = []                                 # --key={RUN}/{RUN_DIR}/{NGPU}/{STRATEGY}/{CKPT} placeholders
checkpoint_glob = ""                           # relative to the run dir; newest match is {CKPT} on hops > 0
env_file = ".env"                              # KEY=VALUE lines, never executed as shell
binds = ["${BEEGFS}", "${USW}", "${SSD}"]
nv = true                                      # apptainer --nv

[slurm]
job_name = "job"      account = "kasieczka_gpu"   partition = "gpu"   nodes = 1   gpus = 1
time_limit = "24:00:00"     # per hop
signal_seconds = 600        # a hop is stopped this long BEFORE time_limit; must be < time_limit
max_hops = 20               # at most this many Slurm jobs per chain
mail = ""   reservation = ""
extra_args = []             # e.g. ["--cpus-per-task=8"]; --mem* is forbidden on Hummel
[validation]
writable_args = []          # single-command mode: option names whose value must be writable
```

* In **manifest mode** only `output_dir`, `cache_dir` and `[slurm]` matter; `image`, `command`, `binds` … are ignored.
* `time_limit` must exceed `signal_seconds` whenever continuation is on (`max_hops > 1` and no
  `--no-resubmit`); otherwise the stop signal arrives right after the job starts and every hop is cut
  short.  `humsub` rejects such a configuration.
* Environment overrides: `HUMMEL_IMAGE`, `HUMMEL_OUTPUT_DIR`, `HUMMEL_CACHE_DIR`, `HUMMEL_ACCOUNT`,
  `HUMMEL_PARTITION`, `HUMMEL_TIME_LIMIT`, `HUMMEL_MAIL`, `HUMMEL_RESERVATION`, `HUMMEL_MAX_HOPS`.
* Always set by `humsub` (not overridable via `extra_args`): `--export=NONE`, `--signal`, `--output`, `--chdir`,
  account/partition/nodes/gpus/time, dependency.  Jobs source `/sw/batch/init.sh` first, use no `srun`.

Storage on Hummel-2: `$HOME` and `$USW` are read-only in batch; `$BEEGFS` is for large persistent data and
logs; `$SSD` for small-file/random I/O and caches (no backup; 100 GiB quota); `/tmp` is RAM and counts as job memory.

## Command reference

```text
humsub init [--project-only | --user-only]     write config templates
humsub config [--json]                         show resolved configuration
humsub submit-manifest --manifest M --payload P [--stage N=PATH]… [--stage-exclude N=PATTERN]…
        [--run-name NAME] [--wait] [--retries R] [--tasks-per-job K] [--parallel-jobs J] [--dry-run]
        [--no-resubmit] [--time T] [--account A] [--partition P] [--reservation R] [--mail M]
        [--max-hops H] [--sbatch-arg ARG]… [--skip-path-checks]
humsub submit [same Slurm options] [--dry-run] [--run-name NAME] -- ARGS…      single-command mode
humsub submission-status SUBMISSION            chains of a submission and their states
humsub status CHAIN                            one chain: state, jobs, queue, log paths
humsub follow CHAIN [-n LINES]                 follow the active log across hops; exit code = chain result
humsub cancel CHAIN                            scancel all jobs of the chain and mark it failed
humsub cleanup SUBMISSION [--dry-run] [--force]
humsub gc [--apply] [--successful-after-hours H] [--failed-after-hours H] [--orphans-after-hours H]
humsub help                                    same as --help
```

`CHAIN` is the chain id or any of its Slurm job ids.  Any first argument that is not a subcommand is
treated as `submit` (`humsub -- --epochs=3` ≡ `humsub submit -- --epochs=3`).

### Submit options in detail

* Default: `submit-manifest` returns once all chains are queued ("fire and forget"); law is not kept alive.
* `--wait` keeps the submitting law process alive to poll the chains and to retry failed branches
  (`--retries R`, only with `--wait`) and to apply `--parallel-jobs J` (at most J chains active at once;
  default `0` = submit everything immediately, which is also the only sensible value without `--wait`).
  Use it in `tmux`/`screen`; if it dies, running chains are unaffected.
* `--no-resubmit`: no continuation — a single Slurm job per chain, no stop signal.
* `--dry-run`: validate the manifest, stages and paths and print the plan; submit nothing.
* Path checks run before anything is created (`--skip-path-checks` disables them): shared output
  directory, writable cache directory, every declared output location.

### Retrying

A failed chain (payload exit ≠ 0, or outputs missing) is **not** continued.  Either run with
`--wait --retries R` or, after fixing the cause, run the same `submit-manifest` again with a fresh `--run-name`:
branches whose outputs exist are skipped, the rest run again.

### Cleaning up

| what | where | removed by |
|---|---|---|
| frozen submission, chain state, rendered law jobs, Slurm logs | `<output_dir>/.hummel-submit/…`, `<output_dir>/logs/<job_name>_<jobid>.log` | never (provenance); delete by hand |
| staged inputs | `<cache_dir>/stages/<submission>` | `humsub cleanup`, `humsub gc` |
| branch scratch | `<cache_dir>/payload-work/<submission>/…` | the branch itself; `cleanup`/`gc` for leftovers |

`cleanup SUBMISSION` removes the two cache trees of one finished submission (active ones are refused without
`--force`).  `gc` considers all submissions of the current project: it only reports unless `--apply`; defaults:
successful after 24 h, failed after 168 h.  `--orphans-after-hours H` additionally removes cache directories
no submission of this `output_dir` refers to — submissions made by this version carry an owner marker, so
other projects' live staging in a shared `cache_dir` is spared, but **staging from older versions is
indistinguishable from an orphan**: use a large `H`.  CVMFS/`cmsexec` caches are not touched by `humsub` at all.

## Troubleshooting

| symptom | cause / action |
|---|---|
| `time_limit … must be longer than signal_seconds` | lower `signal_seconds` or raise `time_limit` (see Configuration) |
| chain ends `stopped-no-follower` | a hop hit the time limit but `max_hops` was exhausted: raise `max_hops`, or make branches shorter |
| chain ends `stopped-no-continuation-marker` | the follower started although the previous hop did not request continuation (e.g. it was killed): inspect the previous log, resubmit |
| chain `failed-<rc>` | payload/law exit code `rc`; read the log (`humsub status CHAIN` prints it) |
| `payload returned success but did not materialize declared output(s)` | the payload exited 0 without creating all `outputs` |
| `… must be on shared storage / is read-only in batch` | path check: use `$BEEGFS`/`$SSD`, not `$HOME`, `$USW`, `/tmp` |
| `unresolved environment variable` | run on a Hummel frontend (or define `BEEGFS`, `SSD`, `USW`) |
| `law executable not found` | the venv's `bin/law` must sit next to its `python`; reinstall the venv |
| old behaviour after editing the source | `pip install --no-deps .` again; running chains keep their frozen worker |

On disk, `<output_dir>/.hummel-submit/submissions/<id>/submission.json` and `chains/<id>/state.json` are
plain JSON and the first things to read when something looks wrong.

## How it works

```text
humsub submit-manifest
  → freeze manifest, payload, stages  → law workflow (one law branch per manifest branch)
  → law renders a remote-job script, HummelJobManager wraps it in a chain
  → chain hop = Slurm job: worker.sh → hummel_submit.worker
        runs the law job script; on SIGUSR1 (signal_seconds before the limit) it writes continue-<hop>,
        SIGTERMs the payload group and exits 0; a follower queued with `afterany` starts the next hop
        only if that marker exists; on success/failure the follower is cancelled
  → law sees only the stable chain id (pending / running / finished / failed)
```

Each chain stores a ZIP snapshot of the worker code, so upgrading the package cannot change a running
chain.  law and the chain run in the Hummel host environment; the payload decides about containers.
Custom law workflows can inherit `hummel_submit.contrib.hummel.HummelWorkflow` directly; independent
branches should prefer manifest mode.  The law integration is isolated in `hummel_submit/contrib/hummel`.

## Single-command mode

`humsub submit -- --epochs=100` runs `[execution].command` plus `auto_args` plus your arguments in one
chain, optionally inside `image` (Apptainer with `--nv` unless `nv = false`, `binds`).  On hops after the
first, `{CKPT}` expands to the newest `checkpoint_glob` match.  Arguments that look like output paths
are checked for writability before submission.  Per-submission overrides: `--time`, `--account`, `--partition`,
`--reservation`, `--mail`, `--max-hops`, `--sbatch-arg=…`, `--no-resubmit`, `--dry-run`.

## Development

```bash
pip install pytest && PYTHONPATH=src python -m pytest -q        # scheduler-independent unit tests
```

A real end-to-end test needs Slurm: run `examples/hello/run.sh` on a frontend.
