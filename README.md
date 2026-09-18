# hummel-submit

A small, standard-library-only Python helper for submitting restartable jobs on the UHH Hummel-2 cluster.

It replaces the pattern of copying and editing a large submission shell script. The launcher is installed once; project and personal policy live in layered TOML files.

## Design goals

- Keep Hummel-2 rules in one maintained implementation: `--export=NONE`, `/sw/batch/init.sh` first in the batch worker, and no `srun`.
- Separate personal defaults from project settings.
- Freeze both the resolved configuration and the worker code for each chain so edits/upgrades cannot change a run halfway through.
- Replay the same frozen SLURM options on every follower.
- Continue a chain only after the previous hop explicitly requests continuation on SLURM's pre-timeout `USR1` signal.
- Scope checkpoint discovery to the unique run directory.
- Stop on ordinary application failures by default; retrying is explicit.
- Follow Hummel-2 storage policy and catch common invalid paths before submission.
- Avoid shell parsing for payload commands and SLURM extra arguments.

## Installation

Requires Python 3.11 or newer. Hummel-2 documentation recommends installing downloaded Python software under `$USW`, because `/home` is for user-authored files and `/usw` is specifically intended for reinstallable software.

A lightweight installation is:

```bash
python3 -m venv "$USW/venvs/hummel-submit"
"$USW/venvs/hummel-submit/bin/pip" install .
mkdir -p "$HOME/.local/bin"
ln -sf "$USW/venvs/hummel-submit/bin/hummel-submit" "$HOME/.local/bin/hummel-submit"
```

The Python interpreter used to invoke `hummel-submit` must be visible at the same absolute path on compute nodes. `/usw` is read-only there, which is fine for an interpreter and installed package.

## First-time setup

Run in a project directory:

```bash
hummel-submit init
```

This creates, without overwriting existing files:

- `~/.config/hummel-submit/config.toml` — personal fallback settings
- `./.hummel-submit.toml` — project-specific settings

Precedence is built-in defaults → personal config → project config → selected `HUMMEL_*` environment overrides → explicit submit CLI overrides.

Show the resolved configuration with:

```bash
hummel-submit config
```

## Hummel-2 storage model

The defaults intentionally use Hummel's environment variables rather than spelling out paths. This avoids hard-coding an SSD number or the current directory layout.

```text
$HOME       source/config written by the user; backed up; READ-ONLY in batch jobs
$USW        installed software/containers;          READ-ONLY in batch jobs
$BEEGFS     large persistent data/checkpoints/logs; writable, good streaming I/O
$SSD        small-file/random-I/O scratch/caches;    writable, no backup/redundancy
/tmp        per-job RAM filesystem;                  writable, counts as job memory
/dev/shm    per-job RAM filesystem;                  writable, counts as job memory
```

Accordingly, the default user config is essentially:

```toml
[execution]
output_dir = "${BEEGFS}/jobs"
cache_dir = "${SSD}/.hummel-submit/cache"
binds = ["${BEEGFS}", "${USW}", "${SSD}"]
```

Large run results and checkpoints go to BeeGFS. Matplotlib, Triton and TorchInductor caches use a per-job directory under `cache_dir`; this directory is removed when the payload exits. The helper does **not** put those caches in `/tmp`, because Hummel-2 implements `/tmp` and `/dev/shm` as RAM-backed job-private filesystems.

The project itself may live under `$HOME`; it only needs to be read there. A batch payload should write results to `{RUN_DIR}` or another writable `$BEEGFS`/`$SSD` location rather than creating files next to the source checkout.

Relevant RRZ documentation:

- https://www.rrz.uni-hamburg.de/en/services/hpc/hummel2-2024/data.html
- https://www.rrz.uni-hamburg.de/en/services/hpc/hummel2-2024/data/tmpdir.html
- https://www.rrz.uni-hamburg.de/en/services/hpc/hummel2-2024/batch.html

## Example project config

```toml
[execution]
image = "${USW}/containers/myproject-latest.sif"
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
gpus = 1
time_limit = "4:00:00"
max_hops = 20
extra_args = ["--cpus-per-task=8"]

[validation]
# Add application-specific options whose values are known to be write targets.
# Common names such as --output, --output-dir, --save-dir, --log-file, etc.
# are recognized automatically.
writable_args = ["--tensorboard-dir"]
```

Hummel-2 determines GPU virtual-node allocation with `--gpus`; the helper therefore emits that form. Explicit SLURM memory requests such as `--mem`, `--mem-per-cpu` and `--mem-per-gpu` are rejected because Hummel-2 policy says memory must not be requested directly.

`checkpoint_glob` is relative to `$OUTPUT_DIR/runs/<run-name>/`; absolute paths and `..` are rejected so one run cannot accidentally resume another run's checkpoint.

Available `auto_args` placeholders are `{RUN}`, `{RUN_DIR}`, `{NGPU}`, `{STRATEGY}`, and `{CKPT}`. Prefer `--key=value` in `auto_args`: a user-supplied argument with the same `--key` after `--` suppresses the automatic one.

## Submission-time path validation

Before calling `sbatch`, `hummel-submit` validates paths that it knows must be writable and performs a conservative scan of path-like payload arguments.

It always checks `execution.output_dir` and `execution.cache_dir`. `/home` and `/usw` are hard errors for write targets because both are read-only in Hummel batch jobs. The persistent output directory must also be shared across hops, so `/tmp`, `/dev/shm`, Hummel-managed temporary BeeGFS directories and node-specific NVMe-oF storage are rejected for `output_dir`.

For application arguments, the helper cannot generally know whether a path is an input or output. The policy is therefore deliberately conservative:

- an existing path is treated as an input unless its option name is clearly output-like;
- a nonexistent path is treated as something the job is likely to create and is checked for compute-node writability;
- common output names such as `--output`, `--output-dir`, `--save-dir`, `--checkpoint-path`, `--log-file`, `--cache-dir`, and variants with `_` are recognized automatically;
- additional project-specific write options can be declared in `[validation].writable_args`;
- existing symlink prefixes are resolved before classifying the filesystem;
- an unrecognized filesystem produces a warning when the submission-node permissions look plausible, because the helper cannot prove its compute-node visibility.

Example:

```bash
# rejected: /home is writable on the frontend but read-only in the batch job
hummel-submit -- --output-dir="$HOME/results"

# accepted: existing input under /home is read-only but readable
hummel-submit -- --input="$HOME/config/model.yaml" --output-dir="$BEEGFS/jobs/result"
```

The check is intentionally not a security boundary and cannot understand arbitrary application semantics or every container-internal path. For an exceptional setup, bypass only the preflight check with:

```bash
hummel-submit --skip-path-checks -- --some-special-path=/custom/location
```

`--dry-run` still performs path validation, making it useful as a preflight command.

## Submitting

```bash
hummel-submit submit -- --epochs=100 --learning-rate=1e-3
```

For convenience, `submit` is the implicit default:

```bash
hummel-submit --dry-run -- --epochs=100
hummel-submit --no-resubmit -- --smoke-test
```

Useful one-off overrides include:

```bash
hummel-submit submit --time 12:00:00 -- --epochs=100
hummel-submit submit --reservation kasieczka -- --epochs=2
hummel-submit submit --sbatch-arg=--exclude=g002 -- --epochs=100
hummel-submit submit --no-resubmit -- --smoke-test
```

Account, partition, time, reservation and GPU count belong in TOML. `slurm.extra_args` / `--sbatch-arg` remain an escape hatch for options such as `--cpus-per-task`, `--constraint`, and `--exclude`. Options that would break chain invariants (`--export`, `--dependency`, `--signal`, etc.) are rejected there.

### Courtesy slicing and failures

For long training on shared GPUs, use e.g. `time_limit = "4:00:00"` or `"12:00:00"`. Each hop pre-queues an `afterany` follower, but that follower starts useful work only if the preceding hop wrote a continuation marker after receiving the pre-timeout `USR1` signal. A plain `scancel`, startup failure, or ordinary application failure therefore does not silently restart the workload.

By default a nonzero application exit stops the chain and cancels its waiting follower. Projects that explicitly want retry behavior can set:

```toml
[slurm]
retry_on_failure = true
```

A failed hop then continues only if a checkpoint exists.

## Cancelling and inspecting a chain

Submission prints a chain id. Inspect or cancel it with:

```bash
hummel-submit status 20260918-140501-a1b2c3d4
hummel-submit cancel 20260918-140501-a1b2c3d4
```

A SLURM job id belonging to the chain can also be supplied. A plain `scancel <current-job>` is safe: the waiting follower wakes without a continuation marker, marks the chain stopped, and exits without launching the payload.

## Environment file

`env_file` is parsed as simple dotenv-style `KEY=VALUE` data. Blank lines, `#` comments, optional `export`, and simple quotes around a complete value are accepted. It is **not sourced as shell code**.

For Apptainer jobs these values are passed through `APPTAINERENV_*`, together with `HUMMEL_RUN_NAME`, `HUMMEL_RUN_DIR`, and `HUMMEL_CHAIN_ID`.

## Frozen worker

Each chain stores its resolved JSON state plus a single ZIP snapshot of the installed Python worker. Later hops import directly from that ZIP. This keeps package upgrades from changing a running chain while avoiding a proliferation of tiny Python files on BeeGFS.

The packaged `worker.sh` starts with:

```bash
source /sw/batch/init.sh
```

before executing the frozen worker. No `srun` is used.

## Migration from the original shell launcher

| Original shell setting | New TOML setting |
| --- | --- |
| `IMAGE` | `execution.image` |
| `OUTPUT_DIR` | `execution.output_dir` |
| `COMMAND=(...)` | `execution.command = [...]` |
| `AUTO_ARGS=(...)` | `execution.auto_args = [...]` |
| `CKPT_GLOB` | `execution.checkpoint_glob` |
| `ENV_FILE` | `execution.env_file` |
| `BINDS` | `execution.binds` |
| `ACCOUNT` | `slurm.account` |
| `MAIL` | `slurm.mail` |
| `RESERVATION` | `slurm.reservation` |
| `TIME_LIMIT` | `slurm.time_limit` |
| `MAX_HOPS` | `slurm.max_hops` |
| `SBATCH_ARGS` | `slurm.extra_args` |

There is intentionally no `RUN_NAME_CMD` equivalent. Run names are generated without `eval`; use `hummel-submit submit --run-name NAME` when an explicit name is needed.

For the short `gputest` setup from the old launcher, for example:

```toml
[slurm]
partition = "gputest"
time_limit = "00:10:00"
signal_seconds = 60
gpus = 0
extra_args = ["--cpus-per-task=8"]
```

The group reservation remains opt-in. Leaving `reservation = ""` does not pin a job to `g002`; `extra_args = ["--exclude=g002"]` remains available for long runs that should stay off it.

## Development

The project has no runtime dependencies beyond Python >= 3.11.

```bash
python3 -m unittest discover -s tests -v
```
