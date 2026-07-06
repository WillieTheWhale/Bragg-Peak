#!/usr/bin/env python3
"""Compact watcher for the BraggTransporter Spot GPU training VM."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass
class CmdResult:
    code: int
    out: str
    err: str


def run(cmd: list[str]) -> CmdResult:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return CmdResult(proc.returncode, proc.stdout.strip(), proc.stderr.strip())


def parse_time(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def instance(project: str, zone: str, name: str) -> dict | None:
    result = run(
        [
            "gcloud",
            "compute",
            "instances",
            "describe",
            name,
            f"--project={project}",
            f"--zone={zone}",
            "--format=json",
        ]
    )
    if result.code != 0:
        return None
    return json.loads(result.out)


def gcs_summary(run_gcs: str) -> tuple[str, str, bool, bool]:
    result = run(["gcloud", "storage", "ls", "--recursive", "--long", run_gcs])
    if result.code != 0:
        return "ckpt=none", "metrics=none", False, False

    latest_ckpt: tuple[str, str] | None = None
    latest_metrics: tuple[str, str] | None = None
    done = False
    failed = False

    for line in result.out.splitlines():
        parts = line.split()
        if not parts:
            continue
        uri = parts[-1]
        stamp = " ".join(parts[-3:-1]) if len(parts) >= 3 else ""
        base = uri.rstrip("/").rsplit("/", 1)[-1]
        if base == "DONE":
            done = True
        if base == "FAILED":
            failed = True
        if "checkpoint" in base or "ckpt" in base or base.endswith((".pt", ".pth")):
            latest_ckpt = (stamp, uri)
        if "metric" in base or base.endswith((".csv", ".json", ".npz")):
            latest_metrics = (stamp, uri)

    ckpt_text = "ckpt=none" if latest_ckpt is None else f"ckpt={latest_ckpt[1].rsplit('/', 1)[-1]}"
    metrics_text = (
        "metrics=none"
        if latest_metrics is None
        else f"metrics={latest_metrics[1].rsplit('/', 1)[-1]}"
    )
    return ckpt_text, metrics_text, done, failed


def billing_summary(project: str) -> str:
    result = run(["gcloud", "billing", "projects", "describe", project, "--format=json"])
    if result.code != 0:
        return "spend=n/a"
    try:
        payload = json.loads(result.out)
    except json.JSONDecodeError:
        return "spend=n/a"
    if not payload.get("billingEnabled"):
        return "spend=billing-disabled"
    account = payload.get("billingAccountName", "").rsplit("/", 1)[-1]
    return f"spend=n/a billing={account or 'enabled'}"


def runtime_text(created: dt.datetime | None, now: dt.datetime) -> tuple[str, float]:
    if created is None:
        return "runtime=n/a", 0.0
    hours = max((now - created).total_seconds() / 3600.0, 0.0)
    return f"runtime={hours:.2f}h", hours


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="braggtransporter")
    parser.add_argument("--zone", default="us-central1-f")
    parser.add_argument("--instance", default="braggtransport-gpu")
    parser.add_argument("--run-gcs", default="gs://braggtransporter-braggtransporter/runs/braggtransport-gpu")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--max-hours", type=float, default=10.0)
    args = parser.parse_args()

    billing = billing_summary(args.project)

    while True:
        now = dt.datetime.now(dt.timezone.utc)
        ckpt, metrics, done, failed = gcs_summary(args.run_gcs)
        vm = instance(args.project, args.zone, args.instance)

        if vm is None:
            state = "vm=gone"
            runtime = "runtime=n/a"
            hours = 0.0
        else:
            state = f"vm={vm.get('status', 'UNKNOWN')}"
            runtime, hours = runtime_text(parse_time(vm.get("creationTimestamp")), now)

        guard = ""
        if hours > args.max_hours:
            guard = f" WARN=max-hours-exceeded({args.max_hours:g}h)"

        marker = " done" if done else " failed" if failed else ""
        print(
            f"{now.isoformat(timespec='seconds')} {state} {runtime} {ckpt} {metrics} {billing}{guard}{marker}",
            flush=True,
        )

        if done:
            return 0
        if failed:
            return 1
        if vm is None:
            return 0

        time.sleep(max(args.interval, 5))


if __name__ == "__main__":
    sys.exit(main())
