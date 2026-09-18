from __future__ import annotations

PROJECT_TEMPLATE = r'''# Project-specific settings for humsub.
# Commit this file with the project if these settings are shared by collaborators.

[execution]
# Use "none" to run directly on the compute node.
# $USW is the Hummel-2 location for user-installed software/containers.
image = "${USW}/containers/myproject-latest.sif"

# Command tokens are passed directly; no shell parsing is involved.
command = ["my-train"]

# Automatically added arguments. Prefer --key=value form so a matching argument
# supplied after `--` can override it. Available placeholders:
# {RUN}, {RUN_DIR}, {NGPU}, {STRATEGY}, {CKPT}
auto_args = []

# Relative to the unique run directory $OUTPUT_DIR/runs/<run-name>.
# Leave empty to disable automatic checkpoint discovery/resume.
checkpoint_glob = ""

# Project-relative unless absolute. Parsed as simple KEY=VALUE entries, not sourced
# as shell code.
env_file = ".env"

# Set false for a CPU-only Apptainer job.
nv = true

[slurm]
job_name = "job"
partition = "gpu"
nodes = 1
# Hummel-2 GPU allocations are requested with --gpus.
gpus = 1

# Courtesy slicing is recommended on the group's shared GPUs.
time_limit = "4:00:00"
signal_seconds = 600
max_hops = 20

# Default is deliberately conservative: application failures stop the chain.
# If true, a failure is retried only when a checkpoint exists.
retry_on_failure = false

# Escape hatch for SLURM options not modeled above, e.g.
# extra_args = ["--cpus-per-task=8", "--exclude=g002"]
# Hummel-2 forbids explicit memory requests such as --mem.
extra_args = []

[validation]
# humsub automatically recognizes common output/path option names. Add
# project-specific options here when their values must be writable in batch jobs.
# Example: writable_args = ["--tensorboard-dir", "--artifact-path"]
writable_args = []
'''

USER_TEMPLATE = r'''# Personal fallback settings for humsub.
# Project .hummel-submit.toml settings override these values.

[execution]
# Large persistent run output/checkpoints belong on BeeGFS.
output_dir = "${BEEGFS}/jobs"

# Small-file compiler/runtime caches belong on the assigned SSD. A per-job
# subdirectory is created and removed automatically.
cache_dir = "${SSD}/.hummel-submit/cache"

# Use Hummel's environment variables rather than hard-coding a particular SSD
# number or the pre-September-2026 directory layout.
binds = ["${BEEGFS}", "${USW}", "${SSD}"]

[slurm]
account = "kasieczka_gpu"
mail = ""
reservation = ""
'''
