#!/usr/bin/env python3
"""README figures from finished runs in results/.

Detected sizes are knee + offset: leave-one-out fits on gem5, the toy's offset on BOOM and x86.

  python3 tools/figures.py    # writes docs/figures/*.png
"""

import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from uarchre import viz
from uarchre.analysis import calibrate as cal

RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs", "figures")
BLUE, ORANGE, AQUA, VIOLET = (viz.SERIES[i] for i in (0, 1, 2, 6))
plt = viz.setup()


def curve(path, filler):
    with open(path) as fh:
        pts = [
            (int(r["n"]), float(r["cycles_per_iter"])) for r in csv.DictReader(fh) if r["filler"] == filler
        ]
    return np.array(sorted(pts)).T


def read_json(*parts):
    with open(os.path.join(RES, *parts)) as fh:
        return json.load(fh)


def log2_axis(ax, ticks, axis="x"):
    getattr(ax, f"set_{axis}scale")("log", base=2)
    getattr(ax, f"set_{axis}ticks")(ticks, [str(t) for t in ticks])
    getattr(ax, f"{axis}axis").set_minor_locator(plt.NullLocator())


def tolerance_band(ax, xs, center=0, label="tolerance max(2, 2%)"):
    t = viz.tolerance(xs)
    ax.fill_between(xs, center - t, center + t, color=viz.BAND, lw=0, label=label)


def dots(ax, x, y, color, marker, **kw):
    ax.scatter(x, y, color=color, marker=marker, s=34, edgecolor=viz.SURFACE, lw=0.8, zorder=3, **kw)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=150, metadata={"Software": None})
    plt.close(fig)


def window():
    fig, ax = plt.subplots(figsize=(6.5, 3.5), layout="constrained")
    for c, rob in zip((BLUE, ORANGE, AQUA, VIOLET), (32, 64, 128, 256)):
        n, y = curve(os.path.join(RES, "gem5", f"p11-rob{rob}-d1", "sweep.csv"), "nop")
        ax.plot(n, y, color=c, marker="o", ms=2.5, lw=1.4)
        ax.axvline(rob, color=c, lw=1, ls=(0, (2, 2)))
        ax.annotate(
            f"ROB {rob}",
            (rob, 1),
            xycoords=("data", "axes fraction"),
            xytext=(3, -2),
            textcoords="offset points",
            va="top",
            fontsize=8.5,
            color=viz.INK2,
        )
    log2_axis(ax, [16, 32, 64, 128, 256])
    ax.set(xlim=(14, 300), ylim=(150, 420), xlabel="NOP fillers N", ylabel="cycles per iteration")
    ax.set_title("Window probe on gem5 O3")
    save(fig, "window.png")


def gem5_backend():
    rob = cal.load_points("p11-rob*-d[12]") + cal.load_points("p11-rob*-fp12")
    lq, sq = cal.load_points("p13-lq*"), cal.load_points("p13-sq*")
    sets = [  # label, color, marker, groups that each get their own offset
        ("ROB", BLUE, "o", [[p for p in rob if p["tag"].endswith(s)] for s in ("-d1", "-d2", "-fp12")]),
        ("int register headroom", ORANGE, "^", [cal.load_points("p12-int*")]),
        ("FP register headroom", ORANGE, "v", [cal.load_points("p12-fp*")]),
        ("load queue", AQUA, "s", [[p for p in lq if p["delay"] == d] for d in ("miss", "fp")]),
        ("store queue", AQUA, "D", [[p for p in sq if p["delay"] == d] for d in ("miss", "fp")]),
    ]
    fig, (ax, ex) = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    xs = np.geomspace(10, 320, 200)
    tolerance_band(ax, xs, xs)
    ax.plot(xs, xs, color=viz.AXIS, lw=1)
    tolerance_band(ex, xs, label=None)
    ex.axhline(0, color=viz.AXIS, lw=1)
    counts = {}
    for i, (label, c, m, groups) in enumerate(sets):
        loo = [r for g in groups for r in cal.fit(g)["loo"]]
        t = np.array([r["truth"] for r in loo])
        e = np.array([r["estimate"] for r in loo])
        dots(ax, t, e, c, m, label=label)
        dots(ex, t * 2 ** (0.07 * (i - 2)), e - t, c, m)  # nudged apart in x
        counts[label] = (int(np.sum(np.abs(e - t) <= viz.tolerance(t))), len(t))
    for a in (ax, ex):
        log2_axis(a, [16, 32, 64, 128, 256])
        a.set(xlim=(12, 300), xlabel="configured size (entries)")
    log2_axis(ax, [16, 32, 64, 128, 256], "y")
    ax.set(ylim=(12, 300), ylabel="detected size (entries)")
    ex.set(ylim=(-6, 6), ylabel="detected minus configured (entries)")
    ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Back-end sizes on gem5 RISC-V O3", x=0.01, ha="left", fontsize=11, fontweight="semibold")
    save(fig, "gem5-backend.png")
    return counts


def rob_platforms():
    pts = {"gem5 ARM O3": [], "BOOM RTL": [], "x86 i5-14600": []}  # (true, detected, depth)
    for d in (1, 2):
        f = cal.fit([p for p in cal.load_points(f"arm-rob*-d{d}") if p["filler"] == "nop"])
        pts["gem5 ARM O3"] += [(r["truth"], r["estimate"], d) for r in f["loo"]]
    for cfg in ("small", "medium", "large", "mega"):
        k = read_json("boom", f"{cfg}-d2", "knees.json")
        pts["BOOM RTL"].append((k["truth"]["rob"], k["estimates"]["rob"]["value"], 2))
    x86 = {
        (core, d): read_json("native", f"x86-{core}-miss-d{d}", "knees.json") for core in "pe" for d in (1, 2)
    }
    pts["x86 i5-14600"] = [
        (k["truth"]["rob"], k["estimates"]["rob"]["value"], d) for (_, d), k in x86.items()
    ]

    fig, ax = plt.subplots(figsize=(6.5, 3.6), layout="constrained")
    tolerance_band(ax, np.geomspace(20, 800, 200))
    ax.axhline(0, color=viz.AXIS, lw=1)
    spread = {"gem5 ARM O3": 0.09, "BOOM RTL": 0, "x86 i5-14600": 0.06}  # depth 1 left, depth 2 right
    for (label, p), c, m in zip(pts.items(), (BLUE, ORANGE, AQUA), ("o", "s", "D")):
        t, e, d = np.array(p).T
        dots(ax, t * 2 ** (spread[label] * (2 * d - 3)), e - t, c, m, label=label)
    for core, name in (("p", "P-core"), ("e", "E-core")):
        v = [x86[core, d]["estimates"]["rob"]["value"] for d in (1, 2)]
        t = x86[core, 1]["truth"]["rob"]
        txt = f"{name}: {v[0]:g}" + (f", {v[1]:g}" if v[1] != v[0] else "") + f" vs {t}"
        ax.annotate(
            txt,
            (t * 2**-0.06, max(v) - t),
            xytext=(-8, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8.5,
            color=viz.INK2,
        )
    log2_axis(ax, [32, 64, 128, 256, 512])
    ax.set(
        xlim=(24, 700), ylim=(-6, 6), xlabel="true ROB size (entries)", ylabel="detected minus true (entries)"
    )
    ax.legend(loc="upper left", ncol=2, fontsize=8.5)
    ax.set_title("ROB on other platforms")
    save(fig, "rob-platforms.png")
    return pts


def anomaly():
    toy = read_json("toy", "mech-p42", "data.json")["issue"]["curves"]
    panels = [
        (
            "gem5 O3: LQ allocated at dispatch",
            [curve(os.path.join(RES, "gem5", t, "sweep.csv"), "load") for t in ("p13-lq32", "p13-lq32-fp12")],
        ),
        (
            "toy core: LQ allocated at issue",
            [np.array([toy[k]["n"], toy[k]["y"]]) for k in ("load/1/miss", "load/8/fp")],
        ),
    ]
    labels = ("miss delay (pointer chase)", "ALU delay (dependent FP sqrt)")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True, layout="constrained")
    for ax, (title, curves) in zip(axes, panels):
        for (n, y), c, label in zip(curves, (BLUE, ORANGE), labels):
            ax.plot(
                n,
                y / np.median(y[:4]),
                color=c,
                marker="o",
                ms=2.5,
                lw=1.4,
                label=label if ax is axes[0] else None,
            )
        ax.axvline(32, color=viz.INK2, lw=1, ls=(0, (2, 2)))
        ax.annotate(
            "LQ 32",
            (32, 1),
            xycoords=("data", "axes fraction"),
            xytext=(3, -2),
            textcoords="offset points",
            va="top",
            fontsize=8.5,
            color=viz.INK2,
        )
        ax.set(xlim=(0, 64), ylim=(0.9, 2.2), xlabel="load fillers N", title=title)
    axes[0].set_ylabel("cycles per iteration / low plateau")
    fig.legend(loc="outside upper center", ncol=2, fontsize=8.5)
    save(fig, "anomaly.png")


def main():
    window()
    for label, (n_in, n) in gem5_backend().items():
        print(f"gem5 {label}: {n_in}/{n} within max(2, 2%)")
    for label, p in rob_platforms().items():
        print(f"{label} ROB: " + ", ".join(f"{t:g} -> {e:g}" for t, e, _ in sorted(p)))
    anomaly()


if __name__ == "__main__":
    main()
