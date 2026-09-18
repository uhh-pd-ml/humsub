# hummel-submit

A small, standard-library-only Python helper for submitting restartable jobs on the UHH Hummel-2 cluster.

It replaces the pattern of copying and editing a large submission shell script. The launcher is installed once; project and personal policy live in layered TOML files.

## Design goals

- Hummel-2 rules stay in one maintained implementation: `--export=NONE`, `/sw/batch/init.sh` first in the batch worker, and no `srun`.
- Personal defaults and project settings are separate.
- A chain uses a frozen snapshot of **both its resolved configuration and the installed worker code**, so edits/upgrades do not change a run halfway through.
- Every follower gets the same frozen SLURM options, rather than reconstructing only a subset of the first job's resources.
- A follower starts useful work only after the previous hop explicitly writes a continuation marker on `SIGUSR1`. Plain `scancel`, startup failures, and ordinary program failures therefore do not silently restart the job.
- Checkpoint discovery is restricted to the unique run directory.
- Application failures stop by default. Optional retry-on-failure is explicit and requires an existing checkpoint.
- No shell parsing is used for the payload command or SLURM extra arguments.

## Installation

Requires Python 3.11 or newer (Debian 12 provides Python 3.11).

From this repository:

```bash
python3 -m pip install --user .
```

or, in a virtual environment:

```bash
python3 -m venv ~/.venvs/hummel-submit
~/.venvs/hummel-submit/bin/pip install .
ln -s ~/.venvs/hummel-submit/bin/hummel-submit ~/.local/bin/hummel-submit
```

The Python interpreter used to invoke `hummel-submit` must be available at the same absolute path on the compute nodes. The package itself is snapshotted into each chain, so upgrading/reinstalling it while a chain is running is safe.

## First-time setup

Run in a project directory:

```bash
hummel-submit init
```

This creates, without overwriting existing files:

- `~/.config/hummel-submit/config.toml` — personal fallback settings
- `./.hummel-submit.toml` — project-specific settings

Precedence is:

1. built-in defaults
2. personal config
3. project config
4. selected `HUMMEL_*` environment overrides
5. explicit submit CLI overrides

Show the resolved configuration with:

```bash
hummel-submit config
```

## Example project config

```toml
[execution]
image = "/usw/u/${USER}/singularity_images/myproject-latest.sif"
command = ["my-train"]
auto_args = [
  "--run={RUN}",
  "--output={RUN_DIR}",
  "--strategy={STRATEGY}",
  "--checkpoint={CKPT}",
]
checkpoint_glob = "checkpoints/*.ckpt"
env_file = ".env"

[slurm]
job_name = "training"
partition = "gpu"
nodes = 1
gpus_per_node = 1
time_limit = "4:00:00"
max_hops = 20
extra_args = ["--cpus-per-task=8", "--mem=64G"]
```

`checkpoint_glob` is deliberately relative to `$OUTPUT_DIR/runs/<run-name>/`; absolute paths and `..` are rejected so one run cannot accidentally resume another run's checkpoint.

Available `auto_args` placeholders are:

- `{RUN}`: unique run name
- `{RUN_DIR}`: unique absolute run directory
- `{NGPU}`: GPUs visible to the payload
- `{STRATEGY}`: `ddp` for more than one visible GPU, otherwise `auto`
- `{CKPT}`: newest checkpoint, or the literal `null`

Prefer `--key=value` in `auto_args`. A user-supplied argument with the same `--key` after `--` suppresses the automatic one.

## Submitting

```bash
hummel-submit submit -- --epochs=100 --learning-rate=1e-3
```

Useful one-off overrides:

```bash
hummel-submit submit --time 12:00:00 -- --epochs=100
hummel-submit submit --reservation kasieczka -- --epochs=2
hummel-submit submit --sbatch-arg=--exclude=g002 -- --epochs=100
hummel-submit submit --no-resubmit -- --smoke-test
hummel-submit submit --dry-run -- --epochs=100
```

Common resource settings such as account, partition, time, reservation and GPU count belong in the TOML config. `slurm.extra_args` / `--sbatch-arg` are an escape hatch for options such as `--mem`, `--cpus-per-task`, `--constraint` and `--exclude`. Options that would break chain invariants (`--export`, `--dependency`, `--signal`, etc.) are rejected there.

### Courtesy slicing

For long training on shared GPUs, use e.g. `time_limit = "4:00:00"` or `"12:00:00"`. Each hop pre-queues an `afterany` follower, but that follower checks for a continuation marker written only when the current hop receives SLURM's pre-timeout `USR1` signal. The follower then goes through the scheduler again and resumes from the newest checkpoint.

An intentional time-slice handoff exits the batch worker successfully, so `--mail-type=FAIL` does not generate a failure email at every slice.

### Failures

By default, a nonzero application exit stops the chain and cancels the queued follower. This avoids repeating deterministic failures across all hops.

If a project explicitly wants retry behavior:

```toml
[slurm]
retry_on_failure = true
```

A failed hop will then continue only if a checkpoint already exists.

## Cancelling and inspecting a chain

Submission prints a chain id:

```text
[submit] chain id    20260918-140501-a1b2c3d4
```

Inspect it with:

```bash
hummel-submit status 20260918-140501-a1b2c3d4
```

and cancel the entire chain with:

```bash
hummel-submit cancel 20260918-140501-a1b2c3d4
```

A SLURM job id belonging to the chain can also be supplied instead of the chain id.

A plain `scancel <current-job>` is no longer dangerous: because no continuation marker is written, the already queued follower wakes only to mark the chain stopped and exit without launching the payload.

## Environment file

`env_file` is parsed as simple dotenv-style `KEY=VALUE` data. Blank lines, `#` comments, optional `export`, and simple single/double quotes around a whole value are accepted.

It is **not sourced as shell code**. This is intentional: project environment data should not be an implicit code-execution hook in a group-wide submission helper.

For Apptainer jobs these values are passed through `APPTAINERENV_*`, along with the runtime variables `HUMMEL_RUN_NAME`, `HUMMEL_RUN_DIR`, and `HUMMEL_CHAIN_ID`.

## Hummel-specific worker

The packaged `worker.sh` deliberately starts with:

```bash
source /sw/batch/init.sh
```

before any other command, as required on Hummel-2. It then executes the frozen Python worker snapshot. No `srun` is used.

## Development

The project has no runtime dependencies beyond Python >= 3.11.

Run tests with:

```bash
python3 -m unittest discover -s tests -v
```
