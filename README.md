# hummel-submit (`humsub`)

`humsub` is a small Hummel-2 frontend and a law backend for restartable Slurm jobs.

Version 0.3 changes the architecture deliberately:

- **law** owns workflow semantics: tasks, branches, targets, dependencies, merging and normal failure handling.
- **humsub's Hummel contrib** behaves like a batch system from law's point of view.
- A remote law job is wrapped in an autonomous **chain** of real Slurm jobs.
- law only sees the stable chain id; changing inner Slurm job ids never leak into law.
- Hummel-specific path policy, Slurm options and pre-timeout continuation remain in humsub.

This is the autonomous continuation model. It is intentionally separate from a planned generic law feature where a special remote-job result can ask law itself to resubmit the job.

## Architecture

```text
humsub frontend
    |
    v
law workflow / remote job rendering
    |
    v
HummelJobManager
    |  submit() -> stable chain id
    |  query(chain id)
    |  cancel(chain id)
    v
humsub autonomous chain
    |
    +-- Slurm job A
    |      pre-timeout USR1
    |      explicit continuation marker
    |
    +-- Slurm job B
    |
    +-- Slurm job C ...
```

The law-generated remote-job script is stored persistently on BeeGFS. Each chain hop executes the same script. If a previous law branch already produced its target, law/Luigi naturally considers it complete when the payload is invoked again.

The chain layer does **not** understand law branches, application checkpoints or scientific outputs. It only understands scheduler lifetime. This keeps law as the single source of truth for workflow completion.

## law version

Development currently targets Marcel Rieger's major law rewrite on the public `release_prep` branch, not the legacy 0.1 API. `pyproject.toml` therefore depends directly on:

```text
git+https://github.com/riga/law.git@release_prep
```

The law-specific integration is isolated under `hummel_submit.contrib.hummel` so later API changes are confined to a small adapter.

## Installation

Requires Python 3.11 or newer. On Hummel, install downloaded Python software under `$USW`:

```bash
python3 -m venv "$USW/venvs/hummel-submit"
"$USW/venvs/hummel-submit/bin/pip" install .
mkdir -p "$HOME/.local/bin"
ln -sf "$USW/venvs/hummel-submit/bin/humsub" "$HOME/.local/bin/humsub"
```

Installing the package also installs law from `release_prep`. The Python interpreter and the resulting `law` executable must remain visible at the same absolute path on compute nodes. `/usw` being read-only there is fine.

## First-time setup

Run in a project directory:

```bash
humsub init
```

This creates, without overwriting existing files:

- `~/.config/hummel-submit/config.toml` — personal fallback settings
- `./.hummel-submit.toml` — project-specific settings

Precedence is built-in defaults → personal config → project config → selected `HUMMEL_*` environment overrides → explicit submit CLI overrides.

Show the resolved configuration with:

```bash
humsub config
```

## Hummel-2 storage model

```text
$HOME       source/config written by the user; backed up; READ-ONLY in batch jobs
$USW        installed software/containers;          READ-ONLY in batch jobs
$BEEGFS     large persistent data/checkpoints/logs; writable, good streaming I/O
$SSD        small-file/random-I/O scratch/caches;    writable, no backup/redundancy
/tmp        per-job RAM filesystem;                  writable, counts as job memory
/dev/shm    per-job RAM filesystem;                  writable, counts as job memory
```

Default personal settings are therefore essentially:

```toml
[execution]
output_dir = "${BEEGFS}/jobs"
cache_dir = "${SSD}/.hummel-submit/cache"
binds = ["${BEEGFS}", "${USW}", "${SSD}"]
```

A submission creates two kinds of persistent state below `output_dir`:

```text
.hummel-submit/
  submissions/<submission-id>/
    submission.json
    law/
      bootstrap.sh
      control/
      job-files/

  chains/<chain-id>/
    state.json
    worker.sh
    hummel-submit-worker.zip
```

The persistent law job-file directory is important: an autonomous successor can run long after the submission-side law process and its temporary directories have disappeared.

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
writable_args = ["--tensorboard-dir"]
```

`checkpoint_glob` remains an application-level convenience used by the built-in single-payload law workflow. The generic Hummel chain backend itself is checkpoint-agnostic.

`slurm.retry_on_failure` is retained for config compatibility but is deprecated in the law-backed architecture. Ordinary application/law failures terminate the autonomous chain and are exposed as failures to law. Failure retries belong at the law workflow level, not inside the chain.

## Submission-time path validation

Before invoking law, `humsub` validates paths that it knows must be writable and performs a conservative scan of path-like payload arguments.

It always checks `execution.output_dir` and `execution.cache_dir`. `/home` and `/usw` are hard errors for write targets because both are read-only in Hummel batch jobs. The persistent output directory must also be shared across hops, so `/tmp`, `/dev/shm`, Hummel-managed temporary BeeGFS directories and node-specific NVMe-oF storage are rejected for `output_dir`.

For application arguments:

- an existing path is treated as an input unless its option name is clearly output-like;
- a nonexistent path is treated as something the job is likely to create and is checked for compute-node writability;
- common output names such as `--output`, `--output-dir`, `--save-dir`, `--checkpoint-path`, `--log-file` and `--cache-dir` are recognized;
- additional write options can be declared in `[validation].writable_args`;
- existing symlink prefixes are resolved before filesystem classification.

Use `--skip-path-checks` only as an explicit escape hatch.

## Submitting

```bash
humsub submit -- --epochs=100 --learning-rate=1e-3
```

`submit` remains the implicit default:

```bash
humsub --dry-run -- --epochs=100
humsub --no-resubmit -- --smoke-test
```

The frontend now does the following:

1. resolves and validates the humsub configuration;
2. freezes a submission specification;
3. instantiates the built-in one-branch law workflow;
4. asks law to render and submit the remote job;
5. `HummelJobManager` wraps that rendered job in an autonomous chain;
6. law stores the **chain id** as its remote job id.

The CLI uses law with `no_poll=True` to retain the original non-blocking `humsub submit` behavior. More sophisticated projects can import `HummelWorkflow` and let law poll/control workflows normally.

One-off Slurm overrides are unchanged:

```bash
humsub submit --time 12:00:00 -- --epochs=100
humsub submit --reservation kasieczka -- --epochs=2
humsub submit --sbatch-arg=--exclude=g002 -- --epochs=100
```

Hummel-specific invariants remain enforced: `--export=NONE`, `/sw/batch/init.sh` first in the chain worker, no `srun`, no explicit memory request, and no user-supplied dependency/signal options that would break continuation semantics.

## Autonomous continuation

Each hop pre-queues one `afterany` follower. That follower starts useful work only when the preceding hop wrote an explicit continuation marker after receiving Slurm's pre-timeout `USR1` signal.

A normal sequence is:

```text
Slurm A: run law payload
         receive USR1
         mark continuation
         terminate payload process group
         exit successfully

Slurm B: verify marker
         execute the same law payload again
```

From law's point of view the job is still the same stable chain id and remains running. The inner Slurm ids are implementation details recorded in the chain state.

A plain application failure does **not** continue. The waiting follower is cancelled, the chain becomes failed, and `HummelJobManager.query(chain_id)` reports a law failure.

## Status and cancellation

Submission prints the stable chain id:

```bash
humsub status 20261001-140501-a1b2c3d4
humsub cancel 20261001-140501-a1b2c3d4
```

A known inner Slurm job id can still be supplied to the CLI for convenience; humsub resolves it back to its chain.

`HummelJobManager.cancel(chain_id)` cancels all known inner Slurm jobs and marks the chain terminal. law therefore never needs to know which inner job is currently active.

## Using the Hummel contrib from custom law workflows

The generic command-line workflow is intentionally small. Framework integrations should define normal law tasks and inherit the Hummel workflow backend directly:

```python
import law

from hummel_submit.contrib.hummel import HummelWorkflow


class MyTask(law.Task, HummelWorkflow, law.LocalWorkflow):
    # humsub_spec is supplied by the frontend/framework integration

    def create_branch_map(self):
        return {...}

    def output(self):
        ...

    def run(self):
        ...
```

This is the intended extension point for future DAS and ML/HPO drivers. Splitting, dependencies and merging belong in law tasks; Hummel-specific scheduling and autonomous continuation stay in the contrib.

## Frozen chain worker

Each chain still stores a ZIP snapshot of the installed humsub Python worker plus `worker.sh`. Package upgrades therefore cannot alter continuation mechanics halfway through an existing chain.

The law remote-job script and its rendered inputs are stored separately in the submission's persistent law job-file directory. The chain executes this immutable rendered payload on every hop.

## Testing

The scheduler-independent unit tests can be run with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v
```

A full integration test additionally requires the `release_prep` law dependency and a Slurm/Hummel environment.
