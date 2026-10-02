#!/usr/bin/env python3
"""Summarize a gem5 ROB campaign (P1.1) and seed the offset calibration (P0.1).

Reads results/gem5/<prefix>-rob<R>-d<D>/, fits truth = slope * knee + offset per depth and
checks |error| <= max(2, 2% of truth). Writes results/gem5/<prefix>-summary/.

  tools/rob_summary.py [--prefix p11]
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# palette slots 1-2; marker is the secondary encoding
STYLE = {1: dict(color="#2a78d6", marker="o"), 2: dict(color="#eb6834", marker="s")}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def tolerance(truth):
    return max(2.0, 0.02 * truth)


def load(prefix):
    rows = []
    for path in sorted(
        glob.glob(os.path.join(ROOT, "results", "gem5", f"{prefix}-rob*-d[0-9]", "knees.json"))
    ):
        m = re.search(r"-rob(\d+)-d(\d+)$", os.path.dirname(path))
        rob, depth = int(m.group(1)), int(m.group(2))
        with open(path) as fh:
            kj = json.load(fh)
        k = kj["knees"]["nop"]
        with open(os.path.join(os.path.dirname(path), "sweep.csv")) as fh:
            curve = [
                (int(r["n"]), float(r["cycles_per_iter"])) for r in csv.DictReader(fh) if r["filler"] == "nop"
            ]
        rows.append(
            dict(
                rob=rob,
                depth=depth,
                truth=kj["truth"]["rob"],
                knee=k["knee"],
                ci=k["ci"],
                quality=k["quality"],
                jump=k["jump_ratio"],
                anomaly=k["anomaly"],
                old_estimate=kj["estimates"].get("rob", {}).get("value"),
                elapsed_s=kj.get("elapsed_s"),
                curve=sorted(curve),
            )
        )
    return rows


def calibrate(rows):
    cal = {}
    for d in sorted({r["depth"] for r in rows}):
        pts = [(r["knee"], r["truth"]) for r in rows if r["depth"] == d and r["knee"] is not None]
        if len(pts) < 3:
            continue
        knee, truth = np.array(pts, dtype=float).T
        slope, intercept = np.polyfit(knee, truth, 1)
        offset = float(np.median(truth - knee))  # integer-valued if slope is 1
        cal[d] = dict(
            slope=round(float(slope), 4),
            intercept=round(float(intercept), 2),
            offset=offset,
            offset_sd=round(float(np.std(truth - knee)), 3),
            n_configs=len(pts),
            slope_ok=bool(0.98 <= slope <= 1.02),
        )
    return cal


def plot(rows, cal, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
        }
    )
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))

    # (a) curves aligned on the configured ROB: x = N - ROB, y normalized to the plateau
    for r in rows:
        ns, ys = zip(*r["curve"])
        base = np.median([y for n, y in r["curve"] if n < r["rob"] - 8] or ys[:3])
        ax[0].plot(
            np.array(ns) - r["rob"], np.array(ys) / base, lw=1.2, alpha=0.7, color=STYLE[r["depth"]]["color"]
        )
    for d, st in STYLE.items():
        ax[0].plot([], [], color=st["color"], lw=2, label=f"depth {d}")
    ax[0].legend(frameon=False, loc="upper left")
    ax[0].axvline(0, color=INK2, lw=0.8, ls="--")
    ax[0].set(
        xlabel="fillers N minus configured ROB",
        ylabel="cycles per iteration / plateau",
        title="Each curve steps up just before N = ROB",
        xlim=(-40, 24),
    )

    # (b) diagonal with the calibrated offset; depths nudged apart in x so coincident points stay visible
    lim = [0, max(r["truth"] for r in rows) * 1.08]
    ax[1].plot(lim, lim, color=INK2, lw=0.8)
    nudge = {1: -3, 2: 3}
    for d, st in STYLE.items():
        sel = [r for r in rows if r["depth"] == d and r["knee"] is not None]
        if not sel or d not in cal:
            continue
        ax[1].scatter(
            [r["truth"] + nudge.get(d, 0) for r in sel],
            [r["knee"] + cal[d]["offset"] for r in sel],
            s=36,
            color=st["color"],
            marker=st["marker"],
            edgecolor="white",
            linewidth=1,
            label=f"depth {d} (offset {cal[d]['offset']:g})",
            zorder=3,
        )
    ax[1].set(
        xlim=lim,
        ylim=lim,
        xlabel="configured ROB entries",
        ylabel="detected ROB entries",
        title="Detected vs configured, calibrated offset",
    )
    ax[1].legend(frameon=False, loc="upper left")

    # (c) error vs configured, with the tolerance band
    xs = np.linspace(0, lim[1], 200)
    tol = np.maximum(2, 0.02 * xs)
    ax[2].fill_between(xs, -tol, tol, color=GRID, lw=0, label="tolerance max(2, 2%)")
    ax[2].axhline(0, color=INK2, lw=0.8)
    for d, st in STYLE.items():
        sel = [r for r in rows if r["depth"] == d and r["knee"] is not None]
        if not sel or d not in cal:
            continue
        ax[2].scatter(
            [r["truth"] for r in sel],
            [r["knee"] + cal[d]["offset"] - r["truth"] for r in sel],
            s=36,
            color=st["color"],
            marker=st["marker"],
            edgecolor="white",
            linewidth=1,
            label=f"depth {d}",
            zorder=3,
        )
        old = [r for r in sel if r["old_estimate"] is not None]
        ax[2].scatter(
            [r["truth"] for r in old],
            [r["old_estimate"] - r["truth"] for r in old],
            s=36,
            facecolor="none",
            edgecolor=st["color"],
            marker=st["marker"],
            linewidth=1.2,
            label=f"depth {d}, old 2·depth offset",
            zorder=2,
        )
    ax[2].set(
        xlim=lim,
        xlabel="configured ROB entries",
        ylabel="detected minus configured",
        title="Error against the pass tolerance",
    )
    ax[2].legend(frameon=False, fontsize=8, loc="lower left")

    for a in ax:
        a.grid(color=GRID, lw=0.6)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="p11")
    args = ap.parse_args()
    rows = load(args.prefix)
    if not rows:
        sys.exit(f"no results/gem5/{args.prefix}-rob*-d[0-9]/knees.json yet")
    cal = calibrate(rows)
    out = os.path.join(ROOT, "results", "gem5", f"{args.prefix}-summary")
    os.makedirs(out, exist_ok=True)

    print(
        f"{'ROB':>4} {'d':>2} {'knee':>6} {'q':>5} {'jump':>5} {'old est':>8} {'old err':>8} "
        f"{'cal est':>8} {'cal err':>8} {'tol':>5}  pass  anomaly"
    )
    table = []
    for r in sorted(rows, key=lambda r: (r["depth"], r["rob"])):
        c = cal.get(r["depth"])
        est = r["knee"] + c["offset"] if (c and r["knee"] is not None) else None
        err = est - r["truth"] if est is not None else None
        old_err = r["old_estimate"] - r["truth"] if r["old_estimate"] is not None else None
        ok = err is not None and abs(err) <= tolerance(r["truth"]) and not r["anomaly"]
        table.append(
            dict(
                rob=r["rob"],
                depth=r["depth"],
                knee=r["knee"],
                ci=r["ci"],
                quality=r["quality"],
                jump=r["jump"],
                old_estimate=r["old_estimate"],
                old_error=old_err,
                calibrated_estimate=est,
                calibrated_error=err,
                tolerance=tolerance(r["truth"]),
                passed=ok,
                anomaly=r["anomaly"],
                elapsed_s=r["elapsed_s"],
            )
        )
        f = lambda v, w=8: f"{v:>{w}.1f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"
        print(
            f"{r['rob']:>4} {r['depth']:>2} {f(r['knee'], 6)} {r['quality']:>5.2f} {r['jump']:>5.2f} "
            f"{f(r['old_estimate'])} {f(old_err)} {f(est)} {f(err)} {tolerance(r['truth']):>5.1f}  "
            f"{'yes' if ok else 'NO ':>4}  {r['anomaly'] or ''}"
        )
    print("\ncalibration (truth = slope * knee + intercept; offset = median(truth - knee)):")
    for d, c in cal.items():
        print(
            f"  depth {d}: slope {c['slope']}  intercept {c['intercept']}  offset {c['offset']:g} "
            f"(sd {c['offset_sd']})  n={c['n_configs']}  slope in [0.98, 1.02]: {c['slope_ok']}"
        )
    n_pass = sum(t["passed"] for t in table)
    print(f"\n{n_pass}/{len(table)} configurations within tolerance")

    with open(os.path.join(out, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    with open(os.path.join(out, "calibration.json"), "w") as fh:
        json.dump(
            {
                "emitter": "riscv64",
                "platform": "gem5-o3-v25.1",
                "filler": "nop",
                "delay": "miss",
                "by_depth": cal,
            },
            fh,
            indent=2,
        )
    plot(rows, cal, os.path.join(out, "rob.png"))
    print(f"wrote {os.path.relpath(out, ROOT)}/")


if __name__ == "__main__":
    main()
