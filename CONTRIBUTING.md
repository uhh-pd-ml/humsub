# Contributing to humsub

Thanks for helping.  Bug reports, questions and pull requests are welcome as
[issues](https://github.com/uhh-pd-ml/humsub/issues) and pull requests on GitHub.  For larger changes, open an
issue first so the design can be discussed before you invest the time.

When reporting a problem from Hummel, include the `humsub` version, the command line, and the output of
`humsub status CHAIN` (or `humsub submission-status SUBMISSION`); `humsub doctor` often already names the cause.

## Development setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e . pytest      # also installs law at the pinned commit (see "law" below)
.venv/bin/python -m pytest -q tests    # unit tests; the e2e tests are skipped outside their container
```

Python ≥ 3.11.  humsub has no runtime dependency besides law (and luigi through it); please discuss before adding
one, because every user installs it into a venv on the Hummel frontends.

## Tests

Every change needs tests, and both suites must pass:

* **Unit tests** (`tests/`, seconds, no cluster): `python -m pytest -q tests --ignore=tests/e2e`.
* **End-to-end tests** (`tests/e2e/`, minutes): real Slurm in the
  [hummel-slurm-ci](https://github.com/uhh-pd-ml/hummel-slurm-ci) Docker image, configured like Hummel-2
  (partitions, accounts, submit filter, storage variables).

```bash
tests/e2e/run.sh                         # all except the slow tests
tests/e2e/run.sh -m slow                 # only the slow ones (real Slurm time limits)
tests/e2e/run.sh -m "slow or not slow"   # everything, as CI does
tests/e2e/run.sh -k roundtrip -x         # any pytest arguments
```

`run.sh` needs Docker (the container runs `--privileged`).  It runs `E2E_WORKERS` pytest workers in parallel
(default 4; `0` = serial, easier to read when debugging); `KEEP_CONTAINER=1` leaves the container running and
`--keep-e2e` keeps the work directories; `COVERAGE=1` measures combined unit + e2e coverage.  The header of
`run.sh` documents all options.  The image is a package of the uhh-pd-ml organisation and its license limits it to
members of Universität Hamburg; if you cannot use it, CI runs the e2e suite on pull requests from branches of this
repository.

**Writing e2e tests.**  Use the `project` fixture and keep tests independent, because they run in parallel:

* every project submits under its own Slurm job name (`project.job_name`); look at the queue only through
  `project.squeue()` and never `scancel -u` or assert on the whole queue;
* keep files inside the project's directories (`project.dir`, `project.out`, `project.cache`);
* mark tests that wait for Slurm time limits with `@pytest.mark.slow`;
* if a test asserts on timing, remember that `sacct` has one-second resolution and slurmctld acts on a ~30 s tick.

A failing e2e test prints the project's queue, accounting, chain states and log tails; read those first.

## Compatibility rules

humsub runs unattended for days, so a few things must keep working across versions:

* **Running chains.**  Every chain runs from a ZIP snapshot of the worker code taken at submission, and its
  frozen state is read by the installed version (`status`, `follow`, `cancel`, `resume`).  Changes to the chain
  state or worker protocol must keep chains submitted by older versions working.
* **Older submissions.**  `resume`, `stats` and `prune` read frozen submission specs and law's control files of
  earlier submissions.  When a file format or name changes (e.g. `slurm_jobs_*.json` → `hummel_jobs_*.json` in
  0.5), keep reading the old one and add a test with the old format.
* **Hummel's rules.**  Hummel-2's submit filter rejects memory requests (`--mem*`), non-exclusive multi-node
  jobs and unwritable log directories, and the e2e image runs a copy of that filter.  humsub itself already
  refuses memory options before submission, with an explanation.  Chains run with a low priority (`--nice`) by default so productions do not crowd out other
  users; keep that default.

## law

humsub builds on [law](https://github.com/riga/law).  The integration lives in
`src/hummel_submit/contrib/hummel`: `HummelWorkflow` is a law workflow type of its own (`hummel`) modeled on law's
Slurm classes, and `HummelJobManager` follows the signatures of law's `BaseJobManager` (e.g. `retries=-1` meaning
"instance default").  Keep them aligned with law's own backends rather than reaching into law internals.

law is **pinned to a commit** in `pyproject.toml`.  To move to a newer law:

1. read law's commits since the pin for changes to the job manager and remote workflow APIs;
2. update the pin;
3. run the unit tests and the full e2e suite (`run.sh` builds its wheelhouse from the pin, cached per commit).

## Code and commits

* Match the surrounding code: naming, type hints, and comments that explain *why*, not what.
* User-visible changes go into `README.md` in the same pull request.
* One topic per commit; the message says why the change is needed, not only what changed.
* Pull requests target `main` and need green CI (unit + e2e).

## License

humsub is licensed under the [BSD 3-Clause License](LICENSE).  By contributing, you agree that your contributions
are licensed under the same terms.
