#!/usr/bin/env python3
"""Summarise transport_bench / cluster_bench results into a markdown report.

Usage:
    scripts/analyze.py --root results/<timestamp> [--out report.md] [--no-plots]

Standard library only. Plots (PNG next to the report) are produced when
matplotlib is installed; otherwise the markdown tables are still written.

Directory layout understood (any nesting; a directory is a "variant" named by
its path relative to --root):
    transport.csv / transport_setup.csv         -> transport_bench
    summary.csv / migrations.csv / requests.csv -> cluster_bench
    meta.json / transport_meta.json             -> run metadata
"""

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict


# ---------------------------------------------------------------- helpers --

def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else 0.0


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0.0
    idx = p / 100.0 * (len(xs) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (idx - lo)


def human_bytes(n):
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def md_table(header, rows):
    if not rows:
        return "_no data_\n"
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def find_files(root, name):
    hits = []
    for d, _, files in os.walk(root):
        if name in files:
            hits.append(d)
    return sorted(hits)


def variant_of(root, d):
    rel = os.path.relpath(d, root)
    return "." if rel == "." else rel


def try_import_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


# -------------------------------------------------------------- transport --

def report_transport(root, out_lines, plt, out_dir):
    dirs = find_files(root, "transport.csv")
    if not dirs:
        return
    out_lines.append("## Transport microbenchmark\n")
    out_lines.append(
        "Latency is first post to last completion across both endpoints; GB/s counts "
        "bytes in both directions for `bidir`. NCCL/NIXL Send agents ack at post time, "
        "so the recv-side column (`recv_lat`) is the true completion time.\n"
    )

    series = defaultdict(dict)  # (variant, backend, mode) -> {bytes: (lat, gbps)}
    for d in dirs:
        v = variant_of(root, d)
        rows = read_csv(os.path.join(d, "transport.csv"))
        table = []
        for r in rows:
            b, mode, nbytes = r["backend"], r["mode"], int(r["bytes"])
            lat, gbps = fnum(r["lat_us_median"]), fnum(r["gbps_median"])
            series[(v, b, mode)][nbytes] = (lat, gbps)
            table.append([
                b, mode, human_bytes(nbytes), f"{lat:.1f}", f"{fnum(r['lat_us_p95']):.1f}",
                f"{fnum(r['recv_lat_us_median']):.1f}" if r["recv_lat_us_median"] else "",
                f"{gbps:.2f}", f"{fnum(r['gbps_best']):.2f}",
                "yes" if r["verified"] == "1" else ("n/a" if r["verified"] == "na" else "NO"),
            ])
        out_lines.append(f"### Variant `{v}`\n")
        out_lines.append(md_table(
            ["backend", "mode", "size", "lat median us", "lat p95 us", "recv_lat us",
             "GB/s median", "GB/s best", "verified"], table))

        setup_path = os.path.join(d, "transport_setup.csv")
        if os.path.exists(setup_path):
            srows = [[r["backend"], r["phase"], f"{fnum(r['us']):.0f}"]
                     for r in read_csv(setup_path)]
            out_lines.append("One-time setup cost (us):\n")
            out_lines.append(md_table(["backend", "phase", "us"], srows))

    if plt:
        for mode in ("uni", "bidir"):
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            any_data = False
            for (v, b, m), pts in sorted(series.items()):
                if m != mode or not pts:
                    continue
                xs = sorted(pts)
                label = f"{v}:{b}" if v != "." else b
                axes[0].plot([x / 2**20 for x in xs], [pts[x][0] for x in xs], marker="o", label=label)
                axes[1].plot([x / 2**20 for x in xs], [pts[x][1] for x in xs], marker="o", label=label)
                any_data = True
            if not any_data:
                plt.close(fig)
                continue
            axes[0].set(xscale="log", yscale="log", xlabel="size (MiB)", ylabel="median latency (us)",
                        title=f"transport latency ({mode})")
            axes[1].set(xscale="log", xlabel="size (MiB)", ylabel="GB/s", title=f"bandwidth ({mode})")
            axes[0].legend(fontsize=7)
            fig.tight_layout()
            path = os.path.join(out_dir, f"transport_{mode}.png")
            fig.savefig(path, dpi=120)
            plt.close(fig)
            out_lines.append(f"![transport {mode}]({os.path.basename(path)})\n")


# ---------------------------------------------------------------- cluster --

def load_cluster(root):
    data = []
    for d in find_files(root, "summary.csv"):
        v = variant_of(root, d)
        entry = {"variant": v, "dir": d, "summary": read_csv(os.path.join(d, "summary.csv"))}
        mp = os.path.join(d, "migrations.csv")
        entry["migrations"] = read_csv(mp) if os.path.exists(mp) else []
        data.append(entry)
    return data


def report_scale(cluster, out_lines):
    rows = []
    for e in cluster:
        for r in e["summary"]:
            if r["scenario"] == "scale" and r["warmup"] == "0":
                rows.append((e["variant"], r))
    if not rows:
        return
    out_lines.append("## A. Replica scaling\n")
    groups = defaultdict(list)
    for v, r in rows:
        groups[(v, r["model"], int(r["workers"]), int(r["requests"]))].append(r)
    table = []
    tput = {}
    for (v, model, w, n), rs in sorted(groups.items()):
        t = med([fnum(r["throughput_tok_s"]) for r in rs])
        tput[(v, model, w, n)] = t
        table.append([model, w, n, f"{t:.1f}",
                      f"{med([fnum(r['ttft_p50_us']) for r in rs]) / 1e3:.1f}",
                      f"{med([fnum(r['itl_p50_us']) for r in rs]) / 1e3:.2f}",
                      f"{med([fnum(r['itl_p95_us']) for r in rs]) / 1e3:.2f}",
                      f"{med([fnum(r['itl_p99_us']) for r in rs]) / 1e3:.2f}", len(rs)])
    out_lines.append(md_table(
        ["model", "workers", "sessions", "tok/s", "TTFT p50 ms", "ITL p50 ms", "ITL p95 ms",
         "ITL p99 ms", "n runs"], table))

    eff = []
    for (v, model, w, n), t in sorted(tput.items()):
        if w == 2 and (v, model, 1, n) in tput and tput[(v, model, 1, n)] > 0:
            s = t / tput[(v, model, 1, n)]
            eff.append([model, n, f"{s:.2f}x", f"{100 * s / 2:.0f}%"])
    if eff:
        out_lines.append("Scaling 1 -> 2 workers (tok/s ratio, efficiency vs ideal 2x):\n")
        out_lines.append(md_table(["model", "sessions", "speedup", "efficiency"], eff))


def report_migrate(cluster, out_lines, plt, out_dir):
    rows = []
    for e in cluster:
        for m in e["migrations"]:
            if m["scenario"] == "migrate":
                rows.append((e["variant"], m))
    if not rows:
        return
    out_lines.append("## B. Migration cost\n")
    out_lines.append(
        "`wait_boundary` = request -> begin (waits for the in-flight decode); `xfer` = first post -> "
        "recv-side completion (host timestamps, ~1 ms poll resolution in the worker loop); "
        "`commit_ack` = recv done -> commit; `stall_excess` = ITL gap across the migration minus the "
        "request's steady ITL.\n")

    steady = [(v, m) for v, m in rows if m["warmup"] == "0" and m["committed"] == "1"]
    groups = defaultdict(list)
    for v, m in steady:
        groups[(v, m["model"], m["backend"], int(m["max_seq"]), m["load"])].append(m)
    table = []
    points = defaultdict(list)  # (variant, model, load) -> [(bytes, stall_excess_ms)]
    for (v, model, backend, max_seq, load), ms in sorted(groups.items()):
        nbytes = int(med([fnum(m["bytes"]) for m in ms]))
        xfer = med([fnum(m["xfer_us"]) for m in ms])
        stall = med([fnum(m["stall_excess_us"]) for m in ms])
        table.append([v, model, backend, max_seq, load, human_bytes(nbytes),
                      f"{med([fnum(m['wait_boundary_us']) for m in ms]) / 1e3:.2f}",
                      f"{xfer / 1e3:.3f}",
                      f"{med([fnum(m['commit_ack_us']) for m in ms]) / 1e3:.3f}",
                      f"{med([fnum(m['total_us']) for m in ms]) / 1e3:.2f}",
                      f"{stall / 1e3:.2f}",
                      f"{(nbytes / (xfer * 1e3)) if xfer > 0 else 0:.2f}", len(ms)])
        points[(v, model, load)].append((nbytes, stall / 1e3, backend))
    out_lines.append(md_table(
        ["variant", "model", "backend", "max_seq", "load", "slot bytes", "wait_boundary ms",
         "xfer ms", "commit_ack ms", "total ms", "stall_excess ms", "xfer GB/s", "n"], table))

    # first-migration (warmup) overhead vs steady state
    first = defaultdict(list)
    for v, m in rows:
        if m["warmup"] == "1" and m["committed"] == "1":
            first[(v, m["model"], m["backend"], int(m["max_seq"]), m["load"])].append(fnum(m["xfer_us"]))
    if first:
        steady_xfer = defaultdict(list)
        for v, m in steady:
            steady_xfer[(v, m["model"], m["backend"], int(m["max_seq"]), m["load"])].append(fnum(m["xfer_us"]))
        ftable = []
        for k, xs in sorted(first.items()):
            s = med(steady_xfer.get(k, []))
            ftable.append([k[0], k[1], k[2], k[3], k[4], f"{xs[0] / 1e3:.3f}", f"{s / 1e3:.3f}"])
        out_lines.append("First migration (warmup run) vs steady-state xfer time:\n")
        out_lines.append(md_table(["variant", "model", "backend", "max_seq", "load", "first xfer ms",
                                   "steady xfer ms"], ftable))

    # interference
    itable = []
    for e in cluster:
        for r in e["summary"]:
            if r["scenario"] == "migrate" and r["warmup"] == "0" and r["load"] == "busy":
                during, outside = fnum(r["bg_itl_during_p50_us"]), fnum(r["bg_itl_outside_p50_us"])
                if during > 0 and outside > 0:
                    itable.append((e["variant"], r["model"], r["backend"], int(r["max_seq"]),
                                   during, outside))
    if itable:
        g = defaultdict(list)
        for v, model, backend, ms, d, o in itable:
            g[(v, model, backend, ms)].append((d, o))
        rows_i = []
        for k, vals in sorted(g.items()):
            d, o = med([x[0] for x in vals]), med([x[1] for x in vals])
            rows_i.append([k[0], k[1], k[2], k[3], f"{o / 1e3:.2f}", f"{d / 1e3:.2f}", f"{d / o:.2f}x"])
        out_lines.append("Interference on background sessions' ITL (median) during vs outside "
                         "migration windows:\n")
        out_lines.append(md_table(["variant", "model", "backend", "max_seq", "outside ms",
                                   "during ms", "slowdown"], rows_i))

    if plt and points:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for (v, model, load), pts in sorted(points.items()):
            pts = sorted(pts)
            label = f"{v}:{model}:{load}"
            ax.plot([p[0] / 2**20 for p in pts], [p[1] for p in pts], marker="o", label=label)
        ax.set(xlabel="slot bytes (MiB)", ylabel="stall excess (ms)", xscale="log",
               title="Migration stall vs state size")
        ax.legend(fontsize=6)
        fig.tight_layout()
        path = os.path.join(out_dir, "migration_stall.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        out_lines.append("![migration stall](migration_stall.png)\n")


def report_rebalance(cluster, out_lines):
    rows = []
    for e in cluster:
        for r in e["summary"]:
            if r["scenario"] == "rebalance" and r["warmup"] == "0":
                rows.append((e["variant"], r))
    if not rows:
        return
    out_lines.append("## C. Rebalancing\n")
    groups = defaultdict(list)
    for v, r in rows:
        groups[(v, r["model"], r["backend"], r["config"], int(r["threshold"]))].append(r)
    base = {}
    for (v, model, backend, conf, th), rs in groups.items():
        if conf == "pinned_off":
            base[(v, model, backend)] = med([fnum(r["makespan_s"]) for r in rs])
    table = []
    for (v, model, backend, conf, th), rs in sorted(groups.items()):
        mk = med([fnum(r["makespan_s"]) for r in rs])
        b = base.get((v, model, backend))
        table.append([v, model, backend, conf, th if conf.endswith("_on") else "-",
                      f"{mk:.3f}", f"{b / mk:.2f}x" if b and mk else "-",
                      f"{med([fnum(r['throughput_tok_s']) for r in rs]):.1f}",
                      f"{med([fnum(r['ttft_p95_us']) for r in rs]) / 1e3:.1f}",
                      f"{med([fnum(r['itl_p99_us']) for r in rs]) / 1e3:.1f}",
                      f"{med([fnum(r['ctr_migrates_completed']) for r in rs]):.0f}",
                      f"{med([fnum(r['ctr_bytes_migrated']) for r in rs]) / 2**20:.0f}",
                      len(rs)])
    out_lines.append(md_table(
        ["variant", "model", "backend", "config", "threshold", "makespan s", "vs pinned_off",
         "tok/s", "TTFT p95 ms", "ITL p99 ms", "migrations", "MiB moved", "n"], table))

    # break-even view from the per-migration rows
    be = []
    for e in cluster:
        ms = [m for m in e["migrations"]
              if m["scenario"] == "rebalance" and m["warmup"] == "0" and m["committed"] == "1"]
        if not ms:
            continue
        stall = [fnum(m["stall_excess_us"]) / 1e3 for m in ms]
        remaining = [fnum(m["tokens_remaining"]) for m in ms]
        be.append([e["variant"], len(ms), f"{med(stall):.2f}", f"{pct(stall, 95):.2f}",
                   f"{med(remaining):.0f}"])
    if be:
        out_lines.append("Per-migration cost during rebalancing (a migration pays off only if the "
                         "session has enough tokens left to repay the stall):\n")
        out_lines.append(md_table(["variant", "migrations", "stall_excess median ms",
                                   "stall_excess p95 ms", "tokens remaining median"], be))


def report_logcheck(root, cluster, out_lines):
    runs = {}
    for e in cluster:
        base = os.path.basename(e["dir"])
        if os.path.basename(os.path.dirname(e["dir"])) == "logcheck" and base in ("off", "debug"):
            rs = [r for r in e["summary"] if r["warmup"] == "0"]
            runs[base] = rs
    if len(runs) != 2:
        return
    out_lines.append("## Logging overhead check\n")
    rows = []
    for metric in ("throughput_tok_s", "itl_p50_us", "itl_p99_us", "ttft_p50_us"):
        o = med([fnum(r[metric]) for r in runs["off"]])
        d = med([fnum(r[metric]) for r in runs["debug"]])
        rows.append([metric, f"{o:.1f}", f"{d:.1f}", f"{(d - o) / o * 100:+.1f}%" if o else "-"])
    out_lines.append(md_table(["metric", "MAMBASERVE_LOG=off", "MAMBASERVE_LOG=debug", "delta"], rows))
    out_lines.append("Large deltas mean logging perturbs the run; keep timed runs at warn/off.\n")


def report_meta(root, out_lines):
    metas = []
    for name in ("meta.json", "transport_meta.json"):
        for d in find_files(root, name):
            metas.append((variant_of(root, d), os.path.join(d, name)))
    if not metas:
        return
    out_lines.append("## Run metadata\n")
    _, first = metas[0]
    with open(first) as f:
        m = json.load(f)
    keys = ["git_hash", "timestamp", "hostname", "kernel", "nproc", "cpu_model", "nvcc_version",
            "nccl_version", "ucx_version", "built_with_nccl", "built_with_nixl", "scope_note"]
    out_lines.append(md_table(["key", "value"], [[k, m.get(k, "")] for k in keys if k in m]))
    # Worker mode differs per variant (thread for memcpy, process for nccl/nixl), so list
    # it for every cluster_bench run rather than only the first.
    wm_rows = []
    for v, path in metas:
        if os.path.basename(path) != "meta.json":
            continue
        with open(path) as f:
            mv = json.load(f)
        wm_rows.append([v, mv.get("arg.backends", ""), mv.get("arg.worker-mode", "thread")])
    if wm_rows:
        out_lines.append("Worker mode per cluster_bench run:\n")
        out_lines.append(md_table(["variant", "backends", "worker-mode"], wm_rows))
    if m.get("nvidia_smi_gpus"):
        out_lines.append("GPUs:\n\n```\n" + m["nvidia_smi_gpus"] + "\n```\n")
    if m.get("nvidia_smi_topo"):
        out_lines.append("Topology:\n\n```\n" + m["nvidia_smi_topo"] + "\n```\n")
    envs = {k: v for k, v in m.items() if k.startswith("env.") and v}
    if envs:
        out_lines.append("Transport-related environment (first run; per-variant env is set by "
                         "`run_matrix.sh`):\n")
        out_lines.append(md_table(["var", "value"], [[k[4:], v] for k, v in envs.items()]))


# ------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    out_path = args.out or os.path.join(args.root, "report.md")
    out_dir = os.path.dirname(os.path.abspath(out_path))
    plt = None if args.no_plots else try_import_plt()
    if plt is None and not args.no_plots:
        print("matplotlib not available: writing tables only")

    lines = ["# MambaServe benchmark report\n",
             f"Source: `{os.path.abspath(args.root)}`\n",
             "> Scope: intra-node. cluster_bench uses each backend's native worker mode: "
             "memcpy runs thread-per-worker, NCCL and NIXL run process-per-worker. NIXL uses UCX "
             "between local GPUs (cuda_ipc or host staged); results are not network RDMA "
             "numbers.\n"]
    report_meta(args.root, lines)
    report_transport(args.root, lines, plt, out_dir)
    cluster = load_cluster(args.root)
    report_scale(cluster, lines)
    report_migrate(cluster, lines, plt, out_dir)
    report_rebalance(cluster, lines)
    report_logcheck(args.root, cluster, lines)

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
