"""``humsub stats``: what did a production cost?  Job counts, runtimes, core-hours, queue waits, failures, retries, peak concurrency.

Built from the chain states of a submission (every chain law ever started for it, retries included, plus the
supervisor jobs) and ``sacct``.  Everything the final report of a production needs, without writing a script for it.
"""
from __future__ import annotations

import collections
from datetime import datetime
import json
from pathlib import Path
import statistics
import subprocess
from typing import Any

from .slurm import slurm_command
from .state import chain_root, load_state

FIELDS = "JobIDRaw,JobName,State,Submit,Start,End,Elapsed,ExitCode,NCPUS,Partition"


def parse_elapsed(text: str) -> int:
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    h, m, s = (int(x) for x in text.split(":"))
    return days * 86400 + h * 3600 + m * 60 + s


def parse_time(text: str) -> datetime | None:
    return None if text in ("", "Unknown", "None") else datetime.fromisoformat(text)


def parse_sacct(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 10 or "." in parts[0]:
            continue
        job, name, state, submit, start, end, elapsed, exit_code, ncpus, partition = parts[:10]
        rows.append({
            "job": job, "name": name, "state": state.split()[0], "submit": parse_time(submit), "start": parse_time(start),
            "end": parse_time(end), "elapsed": parse_elapsed(elapsed), "exit": exit_code, "cpus": int(ncpus or 1), "partition": partition,
        })
    return rows


def query_sacct(job_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(0, len(job_ids), 400):          # keep the command line short
        proc = subprocess.run(
            [slurm_command("sacct"), "-X", "-P", "-n", "-j", ",".join(job_ids[i:i + 400]), "-o", FIELDS],
            check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        rows += parse_sacct(proc.stdout)
    return rows


def submission_job_ids(output_dir: Path, spec: dict[str, Any]) -> tuple[list[str], dict[str, int], int]:
    """(all Slurm job ids of the submission incl. supervisor jobs, jobs per chain, number of chains)."""
    jobs = [str(j) for j in spec.get("supervisor_job_ids", [])]
    per_chain: dict[str, int] = {}
    for chain_id in spec.get("chain_ids", []):
        try:
            state = load_state(chain_root(output_dir) / chain_id / "state.json")
        except OSError:
            continue
        ids = [str(j) for j in state.get("jobs", [])]
        jobs += ids
        per_chain[chain_id] = len(ids)
    return sorted(set(jobs), key=lambda j: (len(j), j)), per_chain, len(per_chain)


def _hours(seconds: float) -> float:
    return seconds / 3600


def summarize(rows: list[dict[str, Any]], per_chain: dict[str, int] | None = None, supervisor_name: str = "supervisor") -> dict[str, Any]:
    work = [r for r in rows if supervisor_name not in r["name"]]
    watchdog = [r for r in rows if supervisor_name in r["name"]]
    states = collections.Counter(r["state"] for r in work)
    # sacct's Elapsed has 1 s resolution: a job that finished within its first second shows 0 but did run.  A
    # follower cancelled while pending can also get a start time and 0 s; that one never ran.
    ran = [r for r in work if r["start"] is not None and (r["elapsed"] > 0 or r["state"] not in ("CANCELLED", "PENDING"))]
    done = [r["elapsed"] for r in work if r["state"] == "COMPLETED"]
    waits = [(r["start"] - r["submit"]).total_seconds() for r in work if r["start"] and r["submit"]]
    out: dict[str, Any] = {
        "jobs": len(work), "states": dict(states), "supervisor_jobs": len(watchdog),
        "supervisor_states": dict(collections.Counter(r["state"] for r in watchdog)),
    }
    if done:
        out["completed_hours"] = {"min": _hours(min(done)), "median": _hours(statistics.median(done)), "max": _hours(max(done)), "total": _hours(sum(done))}
    out["ran_jobs"] = len(ran)
    out["elapsed_hours"] = _hours(sum(r["elapsed"] for r in ran))
    out["core_hours"] = sum(r["elapsed"] * r["cpus"] for r in ran) / 3600
    bad = [r for r in ran if r["state"] not in ("COMPLETED", "RUNNING", "PENDING")]
    out["wasted_hours"] = _hours(sum(r["elapsed"] for r in bad))
    out["wasted_fraction"] = (out["wasted_hours"] / out["elapsed_hours"]) if out["elapsed_hours"] else 0.0
    if waits:
        out["queue_wait_hours"] = {"median": _hours(statistics.median(waits)), "max": _hours(max(waits))}
    starts = [r["start"] for r in ran]
    ends = [r["end"] for r in ran if r["end"]]
    if starts and ends:
        out["first_start"], out["last_end"] = min(starts).isoformat(timespec="seconds"), max(ends).isoformat(timespec="seconds")
        out["wall_hours"] = _hours((max(ends) - min(starts)).total_seconds())
    events = sorted([(r["start"], 1) for r in ran] + [(r["end"], -1) for r in ran if r["end"]])
    cur = peak = 0
    for _, d in events:
        cur += d
        peak = max(peak, cur)
    # a 0 s job's end sorts before its start; any job that ran means at least one ran at a time
    out["peak_concurrency"] = max(peak, 1 if ran else 0)
    if per_chain:
        out["chains"] = len(per_chain)
        out["chains_with_more_than_one_job"] = sum(1 for n in per_chain.values() if n > 1)
    out["failed_jobs"] = [
        {"job": r["job"], "state": r["state"], "elapsed_s": r["elapsed"], "exit": r["exit"]}
        for r in work if r["state"] not in ("COMPLETED", "RUNNING", "PENDING", "CANCELLED") and r["elapsed"] >= 0
    ][:200]
    return out


def format_stats(s: dict[str, Any]) -> list[str]:
    lines = [f"jobs:          {s['jobs']} " + "(" + ", ".join(f"{n} {k}" for k, n in sorted(s["states"].items())) + ")"
             + (f"; {s['supervisor_jobs']} supervisor job(s)" if s["supervisor_jobs"] else "")]
    if "chains" in s:
        lines.append(f"chains:        {s['chains']} ({s['chains_with_more_than_one_job']} used more than one job: continuation or retry hops)")
    if "completed_hours" in s:
        c = s["completed_hours"]
        lines.append(f"completed:     min {c['min']:.2f} h / median {c['median']:.2f} h / max {c['max']:.2f} h / total {c['total']:.1f} h")
    lines.append(f"all that ran:  {s['ran_jobs']} jobs, {s['elapsed_hours']:.1f} h elapsed = {s['core_hours']:,.0f} core-hours")
    lines.append(f"not completed: {s['wasted_hours']:.1f} h ({100 * s['wasted_fraction']:.0f} % of the elapsed time) in failed, cancelled or timed-out jobs")
    if "queue_wait_hours" in s:
        lines.append(f"queue wait:    median {s['queue_wait_hours']['median']:.2f} h, max {s['queue_wait_hours']['max']:.2f} h")
    if "wall_hours" in s:
        lines.append(f"wall-clock:    {s['first_start']} -> {s['last_end']} ({s['wall_hours']:.1f} h)")
    lines.append(f"peak running:  {s['peak_concurrency']} jobs at the same time")
    if s["failed_jobs"]:
        lines.append("failed jobs:   " + ", ".join(f"{f['job']} {f['state']} after {f['elapsed_s']} s (exit {f['exit']})" for f in s["failed_jobs"][:8])
                     + (" ..." if len(s["failed_jobs"]) > 8 else ""))
    return lines


def stats_for(output_dir: Path, spec: dict[str, Any], job_name: str) -> dict[str, Any]:
    ids, per_chain, _ = submission_job_ids(output_dir, spec)
    rows = query_sacct(ids) if ids else []
    return summarize(rows, per_chain, supervisor_name=f"{job_name}-supervisor")


def to_json(summary: dict[str, Any]) -> str:
    return json.dumps(summary, indent=2, sort_keys=True, default=str)
