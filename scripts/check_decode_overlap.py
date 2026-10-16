#!/usr/bin/env python3
"""Report per-worker decode overlap from cluster_bench trace_*.csv files.

Usage:
  scripts/check_decode_overlap.py results_10/nccl_bg8/migrate/nccl/trace_migrate_7.csv
  scripts/check_decode_overlap.py results_dir/   # walks for trace_*.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter


def analyze(path: str) -> list[str]:
    rows = list(csv.DictReader(open(path, newline="")))
    evs = []
    for r in rows:
        if r["kind"] not in ("DecodeStart", "DecodeEnd"):
            continue
        evs.append((int(r["t_ns"]), r["kind"], int(r["req_id"]), int(r["worker"])))
    evs.sort()

    lines = [f"== {path} =="]
    if not evs:
        lines.append("  (no decode events)")
        return lines

    workers = sorted({w for *_, w in evs})
    ok = True
    for w in workers:
        inflight = 0
        max_inf = 0
        at_start: list[int] = []
        for _t, kind, _rid, ww in evs:
            if ww != w:
                continue
            if kind == "DecodeStart":
                at_start.append(inflight)
                inflight += 1
                max_inf = max(max_inf, inflight)
            else:
                inflight -= 1
        overlap_pct = 100.0 * sum(1 for x in at_start if x >= 1) / max(len(at_start), 1)
        lines.append(
            f"  worker={w}: max_inflight={max_inf}  decode_starts={len(at_start)}  "
            f"starts_with_prior_inflight={overlap_pct:.1f}%"
        )
        if max_inf < 2:
            ok = False
    lines.append("  PASS" if ok and workers else "  FAIL (need max_inflight >= 2 on some worker)")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="trace CSV or directory to walk")
    args = ap.parse_args()

    paths: list[str] = []
    if os.path.isfile(args.path):
        paths = [args.path]
    else:
        for root, _, files in os.walk(args.path):
            for f in files:
                if f.startswith("trace_") and f.endswith(".csv"):
                    paths.append(os.path.join(root, f))
        paths.sort()

    if not paths:
        print(f"no traces under {args.path}", file=sys.stderr)
        return 2

    any_pass = False
    for p in paths:
        for line in analyze(p):
            print(line)
            if line.strip() == "PASS":
                any_pass = True
        print()
    return 0 if any_pass or len(paths) > 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
