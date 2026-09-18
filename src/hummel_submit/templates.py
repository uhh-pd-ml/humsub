from __future__ import annotations

PROJECT_TEMPLATE = r'''# Project-specific settings for hummel-submit.
# Commit this file with the project if these settings are shared by collaborators.

[execution]
# Use "none" to run directly on the compute node.
image = "/usw/u/${USER}/singularity_images/myproject-latest.sif"

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
gpus_per_node = 1

# Courtesy slicing is recommended on the group's shared GPUs.
time_limit = "4:00:00"
signal_seconds = 600
max_hops = 20

# Default is deliberately conservative: application failures stop the chain.
# If true, a failure is retried only when a checkpoint exists.
retry_on_failure = false

# Escape hatch for SLURM options not modeled above, e.g.
# extra_args = ["--cpus-per-task=8", "--mem=64G", "--exclude=g002"]
extra_args = []
'''

USER_TEMPLATE = r'''# Personal fallback settings for hummel-submit.
# Project .hummel-submit.toml settings override these values.

[execution]
output_dir = "/beegfs/u/${USER}/jobs"
binds = ["/beegfs", "/usw", "/nfs/ssd2.0"]

[slurm]
account = "kasieczka_gpu"
mail = ""
reservation = ""
'''
