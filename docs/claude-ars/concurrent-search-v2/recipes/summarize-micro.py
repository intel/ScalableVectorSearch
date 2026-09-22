#!/usr/bin/env python3
"""Aggregate micro-benchmark results per (config, point, searchers) group.

Parses gzipped harness logs and produces a summary table with latency percentiles,
achieved rates, and memory usage across windows and replicates.
"""
import glob
import gzip
import os
import re
import statistics
import sys

POINT = re.compile(r"--- point \d+ \((\d+) searchers\) / window (\d+) ---")
PCT = re.compile(
    r"^(searches|updates)\s+count=(\d+).*?p99=\s*([\d.]+)s p99\.9=\s*([\d.]+)s max=\s*([\d.]+)s"
)
ACHIEVED = re.compile(
    r"\[point \d+ / window \d+\] ACHIEVED RATES -- searches/s = ([\d.]+) updates/s = ([\d.]+)"
)
MEMORY = re.compile(
    r"\[end of run\] MEMORY -- .*?total=(\d+)"
)

# Configuration order and point order
CONFIG_ORDER = {"ce": 0, "cf": 1, "a": 2, "b": 3}
POINT_ORDER = {"r100": 0, "r10": 1}

rows = []
for path in sorted(glob.glob(sys.argv[1])):
    basename = os.path.basename(path)
    key = re.search(r"(ce|cf|a|b)_(r\d+)_rep(\d+)", basename)
    if not key:
        continue

    config = key.group(1)
    point = key.group(2)
    rep = key.group(3)

    file_memory_mb = None
    cur = None

    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            # Check for memory line (once per file)
            m = MEMORY.search(line)
            if m:
                total_bytes = int(m.group(1))
                file_memory_mb = total_bytes / 1048576.0

            # Check for point header
            m = POINT.search(line)
            if m:
                cur = {
                    "config": config,
                    "point": point,
                    "rep": rep,
                    "searchers": int(m.group(1)),
                    "window": int(m.group(2)),
                }
                rows.append(cur)
                continue

            if cur is None:
                continue

            # Check for latency percentiles
            m = PCT.match(line)
            if m and m.group(1) == "searches":
                cur["n"] = int(m.group(2))
                cur["p99"] = float(m.group(3)) * 1000
                cur["p999"] = float(m.group(4)) * 1000
                cur["max"] = float(m.group(5)) * 1000

            # Check for achieved rates
            m = ACHIEVED.search(line)
            if m:
                cur["achieved_searches"] = float(m.group(1))
                cur["achieved_updates"] = float(m.group(2))

    # Associate file's memory with all groups from this file
    for r in rows:
        if r["rep"] == rep and r["config"] == config and r["point"] == point and "memory_mb" not in r:
            r["memory_mb"] = file_memory_mb

# Group rows by (config, point, searchers)
groups = {}
for r in rows:
    if "p99" not in r:
        continue

    key = (r["config"], r["point"], r["searchers"])
    if key not in groups:
        groups[key] = []
    groups[key].append(r)

# Sort groups by config order, point order, then searchers ascending
sorted_keys = sorted(
    groups.keys(),
    key=lambda k: (
        CONFIG_ORDER.get(k[0], 999),
        POINT_ORDER.get(k[1], 999),
        k[2],
    )
)

# Build output table
hdr = (
    f"{'config':<7} {'point':<6} {'searchers':>9} {'n':>5} "
    f"{'p99 med':>10} {'p99 range':>20} {'p99.9 med':>10} {'max med':>10} "
    f"{'sch/s med':>10} {'upd/s med':>10} {'mem MB':>8}"
)
print(hdr)
print("-" * len(hdr))

for key in sorted_keys:
    config, point, searchers = key
    g = groups[key]

    if not g:
        continue

    # Extract values, using None for missing fields
    p99_vals = [r.get("p99") for r in g if "p99" in r]
    p999_vals = [r.get("p999") for r in g if "p999" in r]
    max_vals = [r.get("max") for r in g if "max" in r]
    sch_vals = [r.get("achieved_searches") for r in g if "achieved_searches" in r]
    upd_vals = [r.get("achieved_updates") for r in g if "achieved_updates" in r]
    mem_vals = [r.get("memory_mb") for r in g if "memory_mb" in r]

    # Format output row
    p99_med = statistics.median(p99_vals) if p99_vals else None
    p99_min = min(p99_vals) if p99_vals else None
    p99_max = max(p99_vals) if p99_vals else None
    p99_range = f"{p99_min:.2f}-{p99_max:.2f}" if p99_min is not None else "-"

    p999_med = statistics.median(p999_vals) if p999_vals else None
    max_med = statistics.median(max_vals) if max_vals else None
    sch_med = statistics.median(sch_vals) if sch_vals else None
    upd_med = statistics.median(upd_vals) if upd_vals else None
    mem_med = statistics.median(mem_vals) if mem_vals else None

    p99_med_str = f"{p99_med:.2f}" if p99_med is not None else "-"
    p999_med_str = f"{p999_med:.2f}" if p999_med is not None else "-"
    max_med_str = f"{max_med:.2f}" if max_med is not None else "-"
    sch_med_str = f"{sch_med:.2f}" if sch_med is not None else "-"
    upd_med_str = f"{upd_med:.2f}" if upd_med is not None else "-"
    mem_med_str = f"{mem_med:.0f}" if mem_med is not None else "-"

    print(
        f"{config:<7} {point:<6} {searchers:>9} {len(g):>5} "
        f"{p99_med_str:>10} {p99_range:>20} {p999_med_str:>10} {max_med_str:>10} "
        f"{sch_med_str:>10} {upd_med_str:>10} {mem_med_str:>8}"
    )
