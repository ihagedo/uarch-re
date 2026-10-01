#!/usr/bin/env python3
"""Window probe on the toy model (P1.1-P1.4).

Sweep the ROB with nop fillers, then int PRF / LQ / SQ one at a time under a
large ROB, then a near-tie (int headroom within 5% of the ROB) that must come
out ambiguous and a separated pair that must not.

  python3 validation/toy/window_sweep.py   # writes results/toy/window-sweep/{diagonal.png, summary.json}
"""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from uarchre.kernels.window import window_probe
from uarchre.backends.platform.toy import run_toy, ToyConfig
from uarchre.analysis.knee import attribute
from uarchre.analysis.sweep import sweep_refine

OUT = os.path.join(ROOT, "results", "toy", "window-sweep")
os.makedirs(OUT, exist_ok=True)

BIG = dict(rob=512, int_prf=32 + 480, fp_prf=32 + 480, lq=256, sq=256)


def measure(cfg, fillers, ns):
    knees = {}
    for f in fillers:
        _, _, knees[f] = sweep_refine(lambda n: run_toy(window_probe(f, n), cfg), ns)  # noqa: B023
    return attribute(knees)


def ns_around(v):
    step = max(1, v // 24)
    lo = max(2, int(v * 0.6))
    hi = int(v * 1.3) + 4
    return list(range(lo, hi + 1, step))


points = []  # (structure, configured, detected, status)

# ROB diagonal
for rob in [32, 48, 64, 96, 128, 192, 256]:
    cfg = ToyConfig(**{**BIG, "rob": rob})
    est = measure(cfg, ["nop"], ns_around(rob))
    e = est["rob"]
    points.append(("rob", rob, e["value"], e["status"]))
    print(f"rob={rob:4d} -> {e['value']}  {e['status']}")

# one structure at a time
for struct, key, filler in [
    ("int_prf_headroom", "int_prf", "int"),
    ("lq", "lq", "load"),
    ("sq", "sq", "store"),
]:
    for v in [16, 32, 48, 64, 96]:
        over = {key: v + (32 if key == "int_prf" else 0)}
        cfg = ToyConfig(**{**BIG, **over})
        est = measure(cfg, ["nop", filler], ns_around(v))
        e = est[struct]
        points.append((struct, v, e["value"], e["status"]))
        print(f"{struct}={v:4d} -> {e['value']}  {e['status']}")

# a near-tie must be flagged ambiguous
cfg = ToyConfig(**{**BIG, "rob": 96, "int_prf": 32 + 94})
est = measure(cfg, ["nop", "int"], ns_around(96))
tie = est["int_prf_headroom"]
print(f"near-tie rob=96 int_headroom=94 -> {tie['value']}  {tie['status']}")
assert tie["status"].startswith("ambiguous"), "near-tie was not flagged"

# a clearly separated pair must not be
cfg = ToyConfig(**{**BIG, "rob": 96, "int_prf": 32 + 60})
est = measure(cfg, ["nop", "int"], ns_around(96))
sep = est["int_prf_headroom"]
print(f"separated rob=96 int_headroom=60 -> {sep['value']}  {sep['status']}")
assert sep["status"] == "measured" and abs(sep["value"] - 60) <= 1

with open(os.path.join(OUT, "summary.json"), "w") as fh:
    json.dump({"points": points, "near_tie": tie, "separated": sep}, fh, indent=2)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5.5, 5))
mark = {"rob": "o", "int_prf_headroom": "s", "lq": "^", "sq": "v"}
for s in mark:
    xs = [p[1] for p in points if p[0] == s and p[2] is not None]
    ys = [p[2] for p in points if p[0] == s and p[2] is not None]
    ax.scatter(xs, ys, marker=mark[s], label=s, s=45)
lim = max(p[1] for p in points) * 1.1
ax.plot([0, lim], [0, lim], "k--", lw=1, label="detected = configured")
ax.set_xlabel("configured size")
ax.set_ylabel("detected size")
ax.set_title("Window probe on the toy model")
ax.legend()
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "diagonal.png"), dpi=130)
errs = [abs(p[2] - p[1]) for p in points if p[2] is not None]
print(f"\nmax |error| = {max(errs)}, mean = {sum(errs) / len(errs):.2f} over {len(errs)} points")
print(f"wrote {OUT}/diagonal.png")
