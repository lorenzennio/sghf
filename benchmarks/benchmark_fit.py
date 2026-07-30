from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable

import pyhf

import sghf


def median_latency(operation: Callable[[], object], repeats: int) -> float:
    operation()
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        durations.append(time.perf_counter() - start)
    return statistics.median(durations)


parser = argparse.ArgumentParser(
    description="Compare repeated SGHF and pyhf fit latency"
)
parser.add_argument("workspace")
parser.add_argument("--repeats", type=int, default=20)
parser.add_argument("--require-speedup", type=float)
args = parser.parse_args()
if args.repeats < 1:
    parser.error("--repeats must be positive")

with open(args.workspace, encoding="utf-8") as stream:
    workspace_spec = json.load(stream)

linear_model = sghf.load(workspace_spec)
pyhf_spec = {
    key: workspace_spec[key]
    for key in ("channels", "measurements", "observations", "version")
}
exact_workspace = pyhf.Workspace(pyhf_spec)
exact_model = exact_workspace.model()
exact_data = exact_workspace.data(exact_model)

sghf_seconds = median_latency(linear_model.fit, args.repeats)
pyhf_seconds = median_latency(
    lambda: pyhf.infer.mle.fit(exact_data, exact_model),
    args.repeats,
)
speedup = pyhf_seconds / sghf_seconds

print(f"repeats: {args.repeats}")
print(f"SGHF median fit: {sghf_seconds:.6f} s")
print(f"pyhf median fit: {pyhf_seconds:.6f} s")
print(f"speedup: {speedup:.2f}x")

if args.require_speedup is not None and speedup < args.require_speedup:
    raise SystemExit(
        f"Required {args.require_speedup:g}x speedup, measured {speedup:.2f}x"
    )
