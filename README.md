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
| **retry** | a *new attempt after a failure* (law-level, `--retries N`, done by the supervisor's passes or `--wait`); a failed chain does not continue |
| **pass** | one run of law over a submission's task graph with `no_poll`: poll once, resubmit failed jobs, submit what is ready (`humsub tick`, the supervisor job) |
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
* `extra_outputs` (optional, same path rules, files **or directories**) are results that are good to have but **not part of the
  contract**: debug files, logs, and above all **checkpoints**.  humsub never counts them for completion, **never deletes them**
  (required outputs are wiped before every attempt and after every failure; extra outputs are not), never cleans them up, and lists
  them in the branch context (`"extra_outputs": [...]`).  Their paths are fixed in the manifest, so they are identical in every
  attempt: a checkpoint written in one hop is found by the next hop, by a retry, after `humsub resume`, and by a later
  study that points at the same path (e.g. to continue training on more data).  Retention is the manifest author's decision.
* `finalize` (optional) is a fan-in step that runs **once, after every branch is complete**, e.g. merging per-branch summaries:

  ```json
  "finalize": {"data": {"command": ["python", "merge.py"]}, "outputs": ["/beegfs/.../summary.json"]}
  ```

  It is a law workflow of its own (`ManifestFinalizeWorkflow`: one branch, the same payload, stages, `common` and Slurm
  settings) whose requirement is the branch workflow, so law submits it in the first **pass** (below) that finds every
  branch complete, and polls and retries it like any job.  Its context has `branch = "finalize"` and `data.finalize =
  true`; like a branch it is done exactly when all of its `outputs` exist.  A manifest with `finalize` switches
  `--supervisor-job` on (unless `--wait`), so the step runs `--supervisor-interval` after the last branch at the latest.
  `humsub tick SUBMISSION` runs a pass by hand; `submission-status` shows the step's state.

### Manifests from a folder search: `make-manifest`

For "input files → known output files" work, nothing needs to be listed per file:

```bash
humsub make-manifest --search /beegfs/.../raw --glob '*.parquet' \
    --require '.+-NEVENT[0-9]+-RS[0-9]+\.parquet' --select '(QCD|tt).+\.parquet' \
    --output '{out}/_manifests/{stem}.json' --set out=/beegfs/.../converted \
    --files-per-branch 50 --finalize-output '{out}/manifest.json' -o manifest.json
```

* every file below `--search` matching `--glob` must fullmatch `--require` (else it stops: an unexpected file is a problem to
  look at, not something to skip); of those, the ones matching `--select` become inputs;
* each input declares the `--output` templates (`{stem}`, `{name}`, `{parent}` of the input and every `--set` key); two inputs
  that would write the same output are an error;
* `--files-per-branch K` packs K inputs into one branch (`data.inputs`): one payload run handles all of them.  Prefer this over
  `--tasks-per-job` when every payload start is expensive (container start, imports): `--tasks-per-job` still starts the
  payload once per branch;
* `--set KEY=VALUE`, `--common FILE`, `--common-json JSON` fill `common`; `--finalize-output`/`--finalize-data-json` add a
  `finalize` step.

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
| `HUMSUB_SCRATCH` | private empty **bulk** scratch directory for this branch.  On BeeGFS by default (`[execution].scratch`, `--scratch`, or the manifest hint below choose `ssd` instead); **removed when the branch ends** — on success, failure *and* when the hop is stopped by the pre-timeout signal |
| `HUMSUB_SCRATCH_FAST` | private empty directory on the SSD (small, fast, 100 GiB quota for everything); same lifetime.  Equal to `HUMSUB_SCRATCH` when the bulk scratch is on the SSD |
| `HUMSUB_STAGE_<NAME>` | staged path of `--stage NAME=…` (name upper-cased, non-alphanumerics → `_`) |
| `HUMSUB_PAYLOAD` | path of the frozen payload |
| `HUMSUB_PAYLOAD_ATTEMPT` | 1 on the first run of the payload, +1 for every `--retry-payload` re-run inside the same job |
| `HUMSUB_SOFT_STOP_FILE` | path of a file that appears when the hop runs out of time (the soft-stop notice, see below); poll for its existence |
| `HUMSUB_SOFT_STOP_AT`, `HUMSUB_HARD_STOP_AT` | expected epoch seconds of the soft-stop notice and of the SIGTERM that follows it (Slurm delivers its signal up to 30 s early); only set when the chain can continue |
| `HUMSUB_HOP_START`, `HUMSUB_HOP_USABLE_SECONDS` | epoch seconds at which the hop started and how many seconds of it are usable (time limit minus the signal window); `humsub` uses them for the time-aware retry rule, a payload may use them too |
| `HUMSUB_ATTEMPT` | law's in-job attempt counter (`LAW_JOB_ATTEMPT`); **always 1**, because a `--retries` resubmission is a new job. Keep your own state if a payload must know it is being retried |
| `HUMSUB_CHAIN_ID`, `HUMSUB_HOP`, `HUMSUB_STATE_PATH` | chain id, hop number (0-based) and state file of the surrounding chain |

Rules:

1. **Exit 0 and create every declared output**, otherwise the branch fails (or is re-run, see `--retry-payload`).
   Before each run `humsub` deletes any outputs left by a failed attempt, and after a failure it deletes whatever
   the payload wrote to the declared paths, so a partial output cannot make a retry look complete.  After a failed
   or interrupted branch only a few small text files of the scratch (≤ 1 MiB each, ≤ 20 files) are kept, in
   `<run_dir>/leftovers/<jobid>-<branch>/`; the scratch itself is removed.
2. **Write atomically** (temporary name next to the target, then rename): scratch and the declared outputs may be
   on different filesystems, so copy from scratch to a temporary name beside the target and rename that.
3. **Be restartable.**  A hop ends in two steps (below): first a *soft-stop notice*, later SIGTERM to the payload's process group
   (it runs in its own session; `humsub` forwards the signal, waits up to 45 s, then SIGKILLs it and removes the scratch).
   The next hop re-runs the whole branch from the beginning (branches that already completed are skipped, a branch that was
   running is restarted).  Branches should therefore be much shorter than `time_limit`, or checkpoint on their own
   (keep the checkpoint in an `extra_outputs` path).
4. **Containers belong inside the payload** (law and the chain run on the host).  A payload that needs a
   runtime calls it itself, e.g. `cmsexec …` or `apptainer exec …`.
5. `$HOME` and `$USW` are read-only in batch; write to `$BEEGFS`, `$SSD`, `HUMSUB_SCRATCH` or `HUMSUB_SCRATCH_FAST`.
6. **Keep big files off the SSD.**  The SSD quota is shared by everything of yours (container runtimes, caches, all
   running branches).  Put per-branch output in `HUMSUB_SCRATCH` (BeeGFS unless you chose otherwise) and only small-file
   I/O in `HUMSUB_SCRATCH_FAST`.

An optional top-level `"humsub"` object in the manifest carries hints for the submission:

```json
{"schema": 1, "humsub": {"scratch": "ssd", "scratch_bytes_per_branch": 700000000}, "branches": [...]}
```

`scratch` (`beegfs`|`ssd`) selects the bulk scratch unless `--scratch` is given; `scratch_bytes_per_branch` feeds the
preflight quota check when the bulk scratch is on the SSD.

### Built-in payload: `builtin:container`

`--payload builtin:container` runs a command from the manifest inside an Apptainer image, so a containerised application
needs no payload script at all:

```json
"common": {
  "container": {"image": "/beegfs/.../app.sif", "workdir": "{stage_code}", "env": {"PYTHONPATH": "{stage_code}"}},
  "command": ["python", "scripts/convert.py", "{inputs}", "--out", "{out}"],
  "out": "/beegfs/.../converted"
}
```

* `container`: `image` (required), `binds` (default `["/beegfs"]`; the stages and the branch scratch are always bound),
  `env`, `workdir`, `apptainer` (default: on `PATH`, else the newest under `/sw/env/system-gcc/apptainer`), `nv` (GPUs).
* `command` comes from `data.command` if the branch has one (so `finalize` can run another command), else `common.command`.
  An element that is exactly `"{inputs}"`/`"{outputs}"` expands to the branch's `data.inputs`/declared outputs; `{field}`
  is replaced from the string values of `common` and `data`, `stage_<name>`, `run_dir`, `branch`, `scratch`, `scratch_fast`.
* humsub's own `PYTHONPATH` is not forwarded into the container.  Stage the application's code (`--stage code=...`) and
  point `workdir`/`PYTHONPATH` at `{stage_code}` to run exactly the submitted version of it.

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
cache_dir  = "${SSD}/.hummel-submit/cache"     # stages, fast branch scratch; SSD
scratch = "beegfs"                             # where HUMSUB_SCRATCH (bulk) lives: "beegfs" | "ssd"
bulk_scratch_dir = "${BEEGFS}/.hummel-submit/scratch"
fast_bytes_per_job = 250000000                 # SSD footprint of one running chain job (preflight check)
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
signal_seconds = 600        # the soft-stop notice comes this long BEFORE time_limit; must be < time_limit
grace_seconds = -1          # notice -> SIGTERM delay: -1 = signal_seconds - 150, 0 = no soft stop (SIGTERM at once)
max_hops = 20               # at most this many Slurm jobs per chain
nice = 1000000              # Slurm --nice of every chain job: other users' jobs go first; 0 = explicit opt-out
supervisor_partition = ""   # partition of --supervisor-job (default: same as `partition`)
supervisor_account = ""     # account of --supervisor-job (default: same as `account`); e.g. a CPU account for GPU chains
mail = ""   reservation = ""
extra_args = []             # e.g. ["--cpus-per-task=8"]; --mem* is forbidden on Hummel
[validation]
writable_args = []          # single-command mode: option names whose value must be writable
```

* **`nice` is on by default — on purpose.**  The design goal is that productions fill the cluster's idle time (nights, weekends,
  holidays) *without* impeding interactive and daytime users: every chain job, **including each continuation hop**, is queued
  below other users' jobs.  Consequently a chain can sit in the queue between two hops for hours, even for days on a busy cluster;
  that waiting is the intended behaviour, not a malfunction, and the reason why chains carry no wall-time limit of their own.
  If you need something done *now* (a deadline, a small test), choose it consciously: `--nice 0` (or `nice = 0` in
  `.hummel-submit.toml`) removes the penalty and is reported at submission; a smaller positive value only lowers your priority a bit.
  nice only orders the queue; it does **not** limit how many of your jobs run at once (see `--max-concurrent`).
  `--supervisor-job` is the one exception: the 1-CPU watchdog is never niced.
* In **manifest mode** only `output_dir`, `cache_dir`, `scratch`, `bulk_scratch_dir`, `fast_bytes_per_job` and `[slurm]` matter; `image`, `command`, `binds` … are ignored.
* Slurm delivers the pre-timeout signal anywhere from `signal_seconds + 30 s` to `signal_seconds` before the
  limit (its time-limit check runs on a 30 s tick).  Keep `time_limit - signal_seconds` well above 30 s plus
  the job's startup time, or the signal can reach a hop that has not started yet and kill it without
  requesting continuation.
* `time_limit` must exceed `signal_seconds` whenever continuation is on (`max_hops > 1` and no
  `--no-resubmit`); otherwise the stop signal arrives right after the job starts and every hop is cut
  short.  `humsub` rejects such a configuration.
* Environment overrides: `HUMMEL_IMAGE`, `HUMMEL_OUTPUT_DIR`, `HUMMEL_CACHE_DIR`, `HUMMEL_ACCOUNT`,
  `HUMMEL_PARTITION`, `HUMMEL_TIME_LIMIT`, `HUMMEL_MAIL`, `HUMMEL_RESERVATION`, `HUMMEL_MAX_HOPS`, `HUMMEL_NICE`,
  `HUMMEL_SCRATCH`.
* Always set by `humsub` (not overridable via `extra_args`): `--export=NONE`, `--signal`, `--output`, `--chdir`,
  account/partition/nodes/gpus/time, dependency, nice.  Jobs source `/sw/batch/init.sh` first, use no `srun`.

Storage on Hummel-2: `$HOME` and `$USW` are read-only in batch; `$BEEGFS` is for large persistent data and
logs; `$SSD` for small-file/random I/O and caches (no backup; 100 GiB quota); `/tmp` is RAM and counts as job memory.

## Command reference

```text
humsub init [--project-only | --user-only]     write config templates
humsub config [--json]                         show resolved configuration
humsub make-manifest --search DIR… [--glob G] [--require RE] [--select RE] --output TEMPLATE… [--files-per-branch K]
        [--set KEY=VALUE]… [--common FILE] [--common-json JSON] [--finalize-output TEMPLATE]… [--finalize-data-json JSON] -o M
humsub submit-manifest --manifest M --payload P|builtin:container [--stage N=PATH]… [--stage-exclude N=PATTERN]…
        [--run-name NAME] [--wait] [--retries R] [--tasks-per-job K] [--parallel-jobs J] [--dry-run]
        [--no-resubmit] [--time T] [--account A] [--partition P] [--reservation R] [--mail M]
        [--max-hops H] [--nice N] [--signal-seconds S] [--grace-seconds G] [--sbatch-arg ARG]… [--skip-path-checks]
        [--retry-payload N | --safety-margin F] [--max-concurrent N] [--scratch beegfs|ssd]
        [--supervisor-job [--supervisor-interval S] [--supervisor-trigger-delay S] [--supervisor-rounds R]] [--ignore-quota-check]
humsub resume SUBMISSION [--dry-run] [--retries R] [--parallel-jobs J] [--stage N=PATH]… [--supervisor-job]
        [--time T] [--account A] [--partition P] [--max-hops H] [--nice N] [--sbatch-arg ARG]…   continue after a give-up
humsub tick SUBMISSION                         one law pass: poll once, resubmit failed jobs, submit what is ready
humsub submit [same Slurm options] [--dry-run] [--run-name NAME] -- ARGS…      single-command mode
humsub submission-status SUBMISSION            chains of a submission, done/in-flight/failed branches, failed law jobs, finalize
humsub status CHAIN                            one chain: state, jobs, queue, log paths
humsub follow CHAIN [-n LINES]                 follow the active log across hops; exit code = chain result
humsub cancel CHAIN                            scancel all jobs of the chain and mark it failed
humsub cleanup SUBMISSION [--dry-run] [--force]
humsub gc [--apply | --submission ID… | -f] [--successful-after-hours H] [--failed-after-hours H] [--orphans-after-hours H] [--stale-scratch]
humsub stats SUBMISSION [--json]               what the production cost: jobs, runtimes, core-hours, queue waits, failures, peak concurrency
humsub prune (--run NAME | --id PREFIX | --before DATE)… [--apply] [--force]     delete the provenance of finished submissions
humsub doctor [--probe] [--check-path P]… [--manifest M] [--timeout S]    can this setup run a production? (--probe: from a batch node)
humsub help                                    same as --help
```

`CHAIN` is the chain id or any of its Slurm job ids.  A first argument starting with `-` is shorthand for
`submit` (`humsub -- --epochs=3` ≡ `humsub submit -- --epochs=3`); any other unknown word is an error (with a
"did you mean" hint), so a typo or a command your installed version lacks never submits a job.

### Submit options in detail

* Default: `submit-manifest` returns once all chains are queued ("fire and forget"); law is not kept alive.
* `--wait` keeps the submitting law process alive to poll the chains and to retry failed branches
  (`--retries R`) and to apply `--parallel-jobs J` (at most J chains active at once; default `0` = submit everything
  immediately).  `--supervisor-job` does the same without a login-node process, one law pass per interval.
  Use it in `tmux`/`screen`; if it dies, running chains are unaffected.
* `--no-resubmit`: no continuation — a single Slurm job per chain, no stop signal.
* `--dry-run`: validate the manifest, stages and paths and print the plan; submit nothing.
* Path checks run before anything is created (`--skip-path-checks` disables them): shared output
  directory, writable cache directory, every declared output location.

### Hop boundaries: the soft stop

At `time_limit - signal_seconds` Slurm signals the chain worker.  Earlier versions then killed the payload immediately, which wasted the
rest of the window and cost a whole extra hop (with the nice penalty: another wait in the queue) for a payload that needed three more
minutes.  Now:

1. **Soft-stop notice.**  The file named in `HUMSUB_SOFT_STOP_FILE` is created and the chain is marked for continuation.  *No signal is
   sent* (the default action of SIGUSR2 would kill payloads that do not know it); payloads poll the file, between epochs, files or events.
   A payload that can finish before `HUMSUB_HARD_STOP_AT` simply does.  A payload that cannot should checkpoint and exit non-zero
   (any code): once the notice exists this is treated as a hop boundary, **not as a failure** (no retry, no failure budget used).
2. **Next branch of a multi-branch chain is not started** after the notice (it would only be killed); the follower hop runs it.
3. **SIGTERM** follows `grace_seconds` after the notice (default `signal_seconds - 150`, so it still lands ~2 minutes before the limit;
   an explicit value must leave at least 90 s of the window).  `grace_seconds = 0` restores the old behaviour.
4. If the payload completes inside the grace window and nothing is left in the job, the chain ends as *completed* and the queued follower
   is cancelled; otherwise the follower continues.

`--signal-seconds` and `--grace-seconds` override both values per submission.

### Concurrency: `--max-concurrent` and `--parallel-jobs`

Two limits, for two different things:

| | `--max-concurrent N` (lanes) | `--parallel-jobs J` (law) |
|---|---|---|
| limits | chains **running** | chains **in the Slurm queue at all** (each running chain also has one pre-queued follower) |
| mechanism | when a chain is created, its first hop gets `--dependency=afterany` on the most recent job of the chain N positions earlier (round-robin lanes) | law keeps jobs back as "unsubmitted" and submits more in a pass while fewer than J are active |
| a free slot is refilled | at once, by Slurm | at the next pass (`--supervisor-interval`, or the `--wait` controller's poll interval) |
| queue footprint | every chain is queued from the start | at most J chains |
| retries / later chains | respect the lane if its predecessor is still alive | counted like any other active job |
| changeable later | no (fixed when a chain is created) | yes: `humsub resume --parallel-jobs J` |

* **Short branches** (minutes): prefer `--max-concurrent`; with `--parallel-jobs` and a 30-minute supervisor interval
  most slots would stay empty between passes.
* **Huge productions**: `--parallel-jobs` bounds what sits in the queue.  Hummel-2 has no per-user limit, but
  `MaxJobCount` (100000) is shared by *all* users; pack inputs into branches (`make-manifest --files-per-branch`) first.
* **Both together** means: at most J chains queued, at most N of them running, so it requires **J > N** (humsub refuses
  J ≤ N: the lanes would never hold anything back).  The queued backlog `J - N` must bridge one pass, roughly
  `J - N ≥ N × supervisor_interval / branch duration` (e.g. N = 200, 10-minute branches, 30-minute interval: J ≈ 800).
* The lane cap is "N plus chains currently in a later hop": continuation or retry hops of a chain can briefly overlap
  with the next chain of the lane.  A lane predecessor that has already ended gives no dependency (`afterany` on it is
  satisfied anyway; Slurm 24.11 also treats a dependency on a job it purged after `MinJobAge` as satisfied, but this
  does not rely on it); should `sbatch` ever reject the dependency, the chain is submitted without it.

### Scratch and the SSD quota

* Every running chain job holds an unpacked container runtime (≈ 150-250 MB) on the SSD, and with `--scratch ssd` its
  branch scratch as well.  Before submitting, `humsub` estimates `concurrent chains × (fast_bytes_per_job [+ scratch])`
  and refuses a submission that would use more than 80 % of the free SSD quota (read from `rrz-quota`), printing the
  estimate otherwise.  Typical fixes: `--max-concurrent N`, `--scratch beegfs`.  `--ignore-quota-check` disables it.

### Retrying and resuming

Inside a job (1, 2) and across jobs (3, 4); choose what fits, you do not need all of them.

1. **`--retry-payload N`** — a failing payload (exit ≠ 0 or declared outputs missing) is re-run up to N times inside the
   same job, with a fresh scratch.  Retries are **time-aware**: before an attempt the longest previous attempt (× 1.2)
   must still fit into the usable time of the hop (time limit minus `signal_seconds`); otherwise the retry is skipped
   and the branch fails.  For equal attempts this means the second retry is skipped after ≈ 50 % of the hop is used,
   the third after ≈ 66 %.  A branch interrupted by the pre-timeout signal is never retried in place (the next hop re-runs it).
2. **`--safety-margin F`** — a failed hop does not end the chain: the already-queued follower hop (which would otherwise be
   cancelled) re-runs it, and law skips every branch whose outputs exist.  The chain may spend `max(1, ceil(F ×
   tasks-per-job))` extra hops on failures, on top of `max_hops`; a spare follower exits immediately when everything is
   done.  The Slurm job of the failed hop stays `FAILED` in `sacct`, so the history remains readable.  It is a
   per-chain budget: a permanently broken input exhausts only its own chain's budget.  Mutually exclusive with
   `--retry-payload` (one in-chain mechanism); nests with law's `--retries` (below).
3. **`--supervisor-job` (law passes)** — the recommended way to run a production unattended.  A one-CPU job re-queues
   itself every `--supervisor-interval` seconds (min 60, default 1800) and runs one **pass**: law re-runs the
   submission's top task with `no_poll`, which polls every chain once, **resubmits failed jobs** (up to `--retries R`,
   default `--supervisor-rounds` = 3; counted across passes from law's jobs file), submits jobs still waiting for
   `--parallel-jobs` room, and runs every task whose requirements are complete (e.g. the manifest's `finalize`
   workflow).  It stops when the top task is complete and gives up when law reports jobs that failed with no retries
   left.  `humsub tick SUBMISSION` runs one pass by hand (the control lock, below, keeps passes from overlapping).
   **Chain ends bring passes forward:** when a chain ends (completed, failed, out of hops), its worker brings the pending
   supervisor job forward to start `--supervisor-trigger-delay` seconds later (default 60, gathering many chain ends into
   one pass; never later than planned), so a finalize step or a retry follows within a minute or two rather than after a
   whole interval.  If a supervisor is running a pass at that moment, the worker leaves a marker
   (`<submission>/pass-requested`) instead, and that supervisor queues its successor after the trigger delay.  The
   interval stays as the heartbeat for chains that end without running any code (node failure, SIGKILL), so it can be
   long.  The supervisor is
   never niced (a niced watchdog starves on a busy partition).  `scrontab` is disabled on Hummel-2, so this is the cron
   substitute inside the cluster; `humsub tick` from your own cron works as well.
4. **`--wait --retries R`** (controller based): law keeps polling and resubmits failed branches while the
   submitting process lives.

#### One supervisor, one pass at a time: the control lock

Every submission has a **control lock**, `<submission dir>/control.lock`.  Whoever runs a pass or changes the supervisor
schedule must hold it: the supervisor job (from its start to its end, including queuing its successor), `humsub tick`,
`humsub resume` and `submit-manifest` while it queues the first supervisor.  Supervisor jobs can only be queued under
it, so a supervisor is only ever queued by the running supervisor itself or while none runs; a supervisor that finds
another supervisor holding the lock ends at once **without** a successor, so two lines of supervisors can never form.
A supervisor that finds a `tick`/`resume` holding it waits (up to 20 minutes); `tick` reports `busy`, `resume` refuses.  A chain
whose end requests a pass changes the supervisor schedule only if it gets the lock, otherwise it leaves the marker.

* **Across nodes.**  On Hummel-2's BeeGFS `flock` is node-local (a compute node acquires a lock the login node holds),
  so the lock is the *existence* of the file, created with `O_EXCL`, which BeeGFS makes atomic cluster-wide.  The file
  records its owner (Slurm job id or host and pid, purpose, a unique token); `cat control.lock` shows who holds it.
* **Crashes never block the production.**  The holder removes the file when it ends (also on Slurm's SIGTERM), but
  nothing runs after SIGKILL, an OOM kill or a node failure, so a lock is also released by its **staleness**, decided
  without the owner: a Slurm job that Slurm no longer lists (if Slurm cannot be asked, the lock counts as alive), a
  process on the same host whose pid is gone, or any lock not refreshed for 10 minutes (the holder touches it every
  minute).  A stale lock is broken by an atomic rename, so two processes cannot both break it and both own it.
* **Short critical sections use the same principle.**  Read-modify-write of `submission.json` and of a chain's
  `state.json`, and chain creation, are protected by short-held `O_EXCL` mutexes (`.submission.mutex`, `.state.mutex`,
  `.submit-chain.mutex`; broken when older than 60 s, chain creation 5 min), so a worker on a compute node and a
  command on the login node cannot lose each other's updates.

#### In-chain retries (`--safety-margin`) versus law retries (`--retries`)

They nest: while a chain retries in its own follower hops, law sees it as running; only when its budget is used up
does law see the job fail, and then starts a **new chain** (with a fresh hop budget) for it.  A job therefore gets at
most `(1 + safety-margin hops) × (1 + retries)` attempts.  They differ in *when* and *how* the retry runs:

| | `--safety-margin` (follower hop) | law `--retries` (pass) |
|---|---|---|
| starts | at once: the follower was queued when the failed hop started | at the next pass (up to `--supervisor-interval`), then queued anew |
| queue priority | the follower's (it has been waiting since the hop started) | a fresh, niced job at the back of the queue |
| budget | `max_hops + margin` of the same chain | law's `attempts`; the new chain has its full `max_hops` |
| covers | the payload failing in a chain that still has a queued follower | anything that ends a chain: payload failure, budget exhausted, chain killed/cancelled, failed to start, lost follower |
| `HUMSUB_HOP` / logs | continues the chain's hop count; one chain history | a new chain starting at hop 0; law's jobs file links the chains |

Use the safety margin for **cheap, fast retries of transient failures** (a flaky node, a filesystem hiccup) in long
productions, and law retries (the supervisor's default: `--supervisor-rounds`) as the **safety net** that also covers
everything a chain cannot handle itself; `humsub resume` grants new law retries after a give-up.  E.g. a large
production: `--safety-margin 0.5 --supervisor-job` (law retries default to 3).

#### What survives a retry

The same in every mechanism, because it is defined by the manifest, not by the chain:

* **Branches whose outputs exist are never re-run** (law skips them within a job of several branches, `--tasks-per-job`).
* **The failed branch starts from the beginning**: its declared outputs are deleted before every attempt, and its
  scratch (`HUMSUB_SCRATCH`, `HUMSUB_SCRATCH_FAST`) is removed when it fails (a few small text files are kept in
  `<run_dir>/leftovers/` for diagnosis).
* **`extra_outputs` survive**, at the same fixed paths: they are the place for **checkpoints**.  A payload that
  checkpoints there resumes from the checkpoint after any kind of retry (in-job, follower hop, law, `resume`) and after
  a time-limit continuation; one that keeps its state in the scratch or derives it from `HUMSUB_HOP` does not.
* Single-command mode (`humsub submit`) differs: its `{CKPT}` is only filled in on hops > 0 of the same chain, so a
  retry in a *new* chain would start from scratch.  Single-command submissions have neither safety margins nor passes
  today; keep this in mind before adding them.

The dependency logic is law's own: the submission records its top task (`law_task` in `submission.json`), and a pass
is nothing but `luigi.build` of that task with every `HummelWorkflow` in `no_poll` mode, so any law task graph built on
`HummelWorkflow` can be driven this way.  law's jobs files (`<law_dir>/control/slurm_jobs_*.json`) are the record of
which chain each job currently runs on and how many retries it used; `humsub submission-status` reads them (done / in
flight / not yet submitted / failed branches, and every failed job with its retries).

**After a give-up: `humsub resume SUBMISSION`.**  When the passes gave up (law reports jobs with no retries left), fix
the cause and resume.  Everything stays in the same submission and the same law workflow: resume grants every failed job
(branch or finalize workflow) a fresh budget of `--retries` (default: as submitted, at least 1), applies changed settings
to the chains created from now on (`--time`, `--account`, `--partition`, `--max-hops`, `--nice`, `--sbatch-arg`, ...,
`--parallel-jobs`), re-copies the stages named with `--stage NAME=PATH` (e.g. a renewed proxy; a stage removed by `humsub
cleanup` is restored from its recorded source, all other stages stay frozen), removes the scratch of Slurm jobs that no
longer exist, runs one pass right away (which resubmits the failed jobs) and queues the supervisor again if the
submission is supervised (`--supervisor-job` adds one).  `--dry-run` shows what would change.  Branches whose outputs
exist are never redone.

Resubmitting the same manifest with a fresh `--run-name` still skips branches whose outputs exist *if the manifest's
output paths do not contain the run name*.

### Doctor: check the setup before a production

`humsub doctor` answers, in seconds, the questions that are cheap before and expensive after submitting thousands of jobs: does the configuration
resolve, are `output_dir` and `cache_dir` usable, does the scheduler answer, how full is the SSD quota, can law be imported?  With **`--probe`** it
also submits one tiny job (the account, partition and GPU request of your configuration, queued with `--nice 0` unless `--keep-nice`; waits `--timeout`
seconds, default 300) and reports what a *batch node* can do:

* write to `output_dir`, `cache_dir`, every `--check-path` and the common output location of a `--manifest` — group folders and other users'
  trees often look fine from the frontend and are not writable from a compute node;
* call `sbatch` (required by `--supervisor-job`), see `$HOME`/`$USW` read-only as expected, and import humsub and law with the interpreter the chains will use;
* the account/partition/GPU combination is accepted (the probe is submitted with exactly that request).

Exit status 1 if anything failed.  A probe that does not start within the timeout is a warning (busy cluster), not a failure.

### Stats: what did it cost?

`humsub stats SUBMISSION` (submission id or run name) sums up the submission — every chain law started for it, retries included, and the supervisor jobs — from the
chain state and `sacct`: job counts by state, runtimes of the completed jobs (min/median/max/total), core-hours of everything that ran, the share spent in failed or
cancelled jobs, queue waits, wall-clock, how many chains needed more than one job (continuations or retry hops) and the peak number of jobs running at once.  `--json` for scripts.

### Cleaning up provenance: prune

`gc` removes SSD caches; the submission records, chain state, run directories and Slurm logs are provenance and stay.  When a production has been verified and its results moved,
`humsub prune` deletes that provenance for *selected submissions only* (`--run NAME`, `--id PREFIX`, `--before YYYY-MM-DD`).
It deletes what the selected submissions **own** — their submission and chain directories, their run directory, the Slurm logs of exactly their job ids (supervisor jobs included) and their
staging and scratch caches — and nothing else: older runs, other projects' logs in the same `output_dir` and, always, every output declared in a manifest are untouched (a directory that
contains a declared output is kept with a note).  It is a dry run unless `--apply`; submissions that still look active are refused unless `--force`.

### Cleaning up

| what | where | removed by |
|---|---|---|
| frozen submission, chain state, rendered law jobs, Slurm logs | `<output_dir>/.hummel-submit/…`, `<output_dir>/logs/<job_name>_<jobid>.log` | never (provenance); delete by hand |
| staged inputs | `<cache_dir>/stages/<submission>` | `humsub cleanup`, `humsub gc` |
| fast branch scratch | `<cache_dir>/payload-work/<submission>/<jobid>-<branch>` | the branch itself (also on SIGTERM); `resume`, `cleanup`, `gc --stale-scratch` for leftovers of killed jobs |
| bulk branch scratch | `<bulk_scratch_dir>/<submission>/<jobid>-<branch>` | same |
| failure diagnostics | `<run_dir>/leftovers/<jobid>-<branch>/` | by hand (a few small text files) |

`cleanup SUBMISSION` removes the cache trees (stages, fast and bulk scratch) of one finished submission (active ones are refused without
`--force`).

`gc` collects what can go, **per submission**: the caches of finished submissions after the retention time (defaults: successful
after 24 h, failed after 168 h), with `--stale-scratch` the scratch of Slurm jobs that no longer exist (killed by SIGKILL, node
failure, `scancel`), and with `--orphans-after-hours H` cache directories no submission of this `output_dir` refers to
(submissions made by humsub ≥ 0.4 carry an owner marker, so other projects' live staging in a shared `cache_dir` is spared, but
**staging from older versions is indistinguishable from an orphan**: use a large `H`).  It lists every submission with what would
be removed, and then:

| | |
|---|---|
| `humsub gc` | lists only (dry run) |
| `humsub gc --apply` | asks which to clean (`1,3`, `2-4`, `all`, or Enter for none); without a terminal it deletes nothing |
| `humsub gc --submission ID` | cleans exactly the named submissions (id, id prefix or run name; repeatable), without asking |
| `humsub gc -f` | **cleans everything listed, without asking**: caches and scratch cannot be recovered, and failed submissions lose what you may still need to debug or resume them |

CVMFS/`cmsexec` caches and runtimes are not touched by `humsub` at all.

### Several projects and users side by side

* **Everything is keyed by submission.**  Submission ids (timestamp + random part) are unique, and so are the submission
  directory, its control lock, its supervisor, its chains, stages and scratch.  Projects that share `output_dir` and `cache_dir`
  (the defaults are the same for all projects of a user) never see each other's locks or supervisors; a supervisor only ever
  acts on the submission named in its arguments (its log starts with `submission <id> (run …, project …)`).
* **Liveness is decided cluster-wide, deletion per user.**  Scratch counts as stale only if its Slurm job id is not in
  `squeue` for *any* user, and only directories owned by the current user are ever removed, so a shared `bulk_scratch_dir` or
  `cache_dir` is safe.  If `squeue` cannot be asked, nothing is removed (an empty answer would mean "every job is dead").
* **A submission belongs to the user who made it** (`user`/`uid` in `submission.json`).  `tick`, `resume`, `cancel`,
  `cleanup`, `prune` and `gc` refuse other users' submissions, so a supervisor never runs under the wrong account; read-only
  commands (`status`, `submission-status`, `stats`, `follow`) work for everyone with read access, e.g. on a group folder.
* **Locking protocol versions.**  A submission records the control-lock generation of its supervisors.  `tick` and `resume`
  refuse a submission whose supervisor from an older humsub (without the cross-node control lock) is still running; once it
  has ended, they upgrade the submission, and its next supervisor is built from the current version.
* **Never use name-based Slurm features on humsub jobs** (`--dependency=singleton`, `scancel --name`, ...): job names
  (`<job_name>`, `<job_name>-supervisor`) are shared by all submissions with the same `job_name`.  humsub itself only ever
  addresses jobs by id.

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

## Tests

```bash
python -m pytest tests --ignore=tests/e2e     # unit tests, no cluster needed (~15 s)
tests/e2e/run.sh                              # end-to-end against a real Slurm configured like Hummel-2 (fast profile)
E2E_PROFILE=realistic tests/e2e/run.sh        # the same with Hummel's timing (real time limits, 60 s passes)
tests/e2e/run.sh -k finalize -x               # any pytest arguments
```

The end-to-end tests run inside the Docker image from
[hummel-slurm-ci](https://github.com/uhh-pd-ml/hummel-slurm-ci) (Slurm 24.11 with Hummel's partitions,
accounts, submit filter and `/sw/batch/init.sh`); they submit real manifests, chains, failures, retries,
supervisor passes, finalize steps, cancellations and pre-timeout continuations.

Almost all of their time is waiting for Slurm, so they are built to wait as little as possible:

* **in parallel** (`pytest-xdist`, `E2E_WORKERS`, default 6): every test cancels only its own jobs (never
  `scancel -u`), uses its own output, cache and scratch directories, and checks only its own queue entries;
* **two profiles** (`E2E_PROFILE`).  `fast` (default; every push and PR in CI): the image's fast scheduler loop, supervisor
  passes every 5 s (`HUMSUB_MIN_SUPERVISOR_INTERVAL` lowers humsub's 60 s floor for the tests only), and tests marked
  `deadline` send the pre-timeout signal themselves (`project.reach_deadline()`: `scancel --batch --signal=USR1`, what
  slurmctld sends) instead of waiting for real time limits.  `realistic` (weekly, on `main` and by hand in CI): Hummel's
  scheduler timing, real time limits and 60 s passes; `deadline` and `supervisor` tests are then marked `slow`;
* **no fixed sleeps**: wait for a condition (`project.wait_for_log()`, files) with a generous timeout;
* `--durations=0` and `[run.sh] Ns: ...` lines show where the time goes.

A new test that needs a hop to run out of time uses `project.configure_deadline(...)` and `project.reach_deadline(chain,
hop, after=...)`, where `after` is a line *the payload* prints (the worker installs its signal handler only right before
it starts the payload).

Set `WHEELHOUSE=dir` to install from local wheels when the container has no network.  Without `WHEELHOUSE`, `run.sh`
builds a wheelhouse on the *host* once (`~/.cache/humsub-e2e-wheelhouse`, needs `pip` and network) because the container
usually has no working DNS; `NO_WHEELHOUSE=1` installs from the network inside the container instead.  `IMAGE=` selects
the image; by default a locally available `hummel-slurm-ci:latest` is used, else the one on ghcr.io.
