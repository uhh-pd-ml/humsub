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
tests/e2e/run.sh                         # everything, fast profile (what CI runs on every push)
E2E_PROFILE=realistic tests/e2e/run.sh   # everything with Hummel's timing (weekly, on main and by hand in CI)
tests/e2e/run.sh -m "not slow"           # skip what is slow in the chosen profile
tests/e2e/run.sh -k roundtrip -x         # any pytest arguments
```

`run.sh` needs Docker (the container runs `--privileged`).  It runs `E2E_WORKERS` pytest workers in parallel
(default 6; `0` = serial, easier to read when debugging); `KEEP_CONTAINER=1` leaves the container running and
`--keep-e2e` keeps the work directories; `COVERAGE=1` measures combined unit + e2e coverage.  The header of
`run.sh` documents all options; `[run.sh] Ns:` lines and `--durations=0` show where the time goes.  The image is a
package of the uhh-pd-ml organisation and its license limits it to members of Universität Hamburg; if you cannot use
it, CI runs the e2e suite on pull requests from branches of this repository.

**Profiles.**  Almost all e2e time is waiting for Slurm, so there are two (`E2E_PROFILE`):

* `fast` (default): the image's fast scheduler loop (`HUMMEL_CI_PROFILE=fast`), supervisor passes every 5 s (the
  tests lower humsub's 60 s floor through `HUMSUB_MIN_SUPERVISOR_INTERVAL`), and tests that need a hop to run out of
  time send the pre-timeout signal themselves instead of waiting for a real time limit.
* `realistic`: Hummel-2's scheduler timing, real time limits (slurmctld's own clock delivers `--signal`) and 60 s
  supervisor passes.  It catches timing assumptions the fast profile cannot; run it before releases.

**Writing e2e tests.**  Use the `project` fixture and keep tests independent and quick, because they run in parallel:

* every project submits under its own Slurm job name (`project.job_name`); look at the queue only through
  `project.squeue()` and never `scancel -u` or assert on the whole queue;
* keep files inside the project's directories (`project.dir`, `project.out`, `project.bulk`, `project.cache`);
* never sleep for a fixed time: wait for a condition with a generous timeout (`project.wait_for_log(...)`, a file);
* a test whose hop must run out of time uses `project.configure_deadline(time_limit=..., signal_seconds=...)` and
  `project.reach_deadline(chain, hop, after=...)`, and is marked `@pytest.mark.deadline`.  In the fast profile
  `reach_deadline` sends what slurmctld sends (`scancel --batch --signal=USR1`) once the payload has printed *after*
  (it must be a line from the payload: the worker installs its handler only right before starting it); in the
  realistic profile the real limit applies and the test counts as slow;
* a test that waits for supervisor passes uses `--supervisor-interval SUPERVISOR_INTERVAL` (from `conftest`) and is
  marked `@pytest.mark.supervisor`;
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
* **Locks and supervisors.**  Every process that runs a pass or changes the supervisor schedule holds the
  submission's control lock (`control.lock`, `O_EXCL`, because `flock` is node-local on BeeGFS); short critical
  sections use `locks.mutex`.  A submission records its lock generation (`control_lock`): bump
  `CONTROL_LOCK_GENERATION` when the protocol changes, so that `tick`/`resume` refuse submissions whose older
  supervisor may still run.
* **Hummel's rules.**  Hummel-2's submit filter rejects memory requests (`--mem*`), non-exclusive multi-node
  jobs and unwritable log directories, and the e2e image runs a copy of that filter.  humsub itself already
  refuses memory options before submission, with an explanation.  Chains run with a low priority (`--nice`) by default so productions do not crowd out other
  users; keep that default.

## law

humsub builds on [law](https://github.com/riga/law).  The integration lives in
`src/hummel_submit/contrib/hummel`: `HummelWorkflow` is a law workflow type of its own (`hummel`) modeled on law's
Slurm classes, and `HummelJobManager` follows the signatures of law's `BaseJobManager` (e.g. `retries=-1` meaning
"instance default").  Keep them aligned with law's own backends rather than reaching into law internals.

One known exception: humsub drives workflows by *passes* (law re-run with `no_poll` on an existing submission, see
`hummel_submit/tick.py`), and law keeps its per-job retry counter (`_job_retries`) only in memory while it persists
`attempts` in the jobs file.  `HummelWorkflowProxy.poll` therefore seeds `_job_retries` from `job_data.attempts`,
otherwise every pass would grant new retries.  `tests/test_tick.py` guards it; check it when moving the law pin, and
prefer a fix in law (seeding the counter from the persisted attempts) over keeping the override.

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
