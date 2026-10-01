#!/usr/bin/env python3
"""Calibrate the toy model's knee-to-size offsets (P0.1).

Per filler and miss depth 1-3 (or --delay fp|int, --depths links), sweep the filler's
structure over five sizes and regress truth on knee; a fit passes with slope ~1 and a
constant offset (the toy is deterministic). --write merges passing fits into offsets.yaml.

  validation/toy/calibrate_offsets.py [--delay fp] [--write]
"""

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from uarchre.kernels.window import window_probe
from uarchre.backends.platform.toy import run_toy, ToyConfig
from uarchre.analysis.knee import find_knee, STRUCT
from uarchre.analysis import offsets

BIG = dict(rob=512, int_prf=32 + 480, fp_prf=32 + 480, lq=256, sq=256)
KNOB = {"nop": "rob", "int": "int_prf", "fp": "fp_prf", "load": "lq", "store": "sq", "mix": "rob"}
TRUTH = {"mix": "rob"}


def one(job):
    filler, depth, size, delay = job
    knob = KNOB[filler]
    cfg = ToyConfig(**{**BIG, knob: size + (32 if knob.endswith("prf") else 0)})
    lo = size - (12 if delay == "miss" else 30)
    ns = list(range(max(2, lo), size + 7))
    ys = [run_toy(window_probe(filler, n, depth=depth, delay=delay), cfg) for n in ns]
    k = find_knee(
        ns, ys, min_jump=1 + 0.4 / depth if delay == "miss" else 1.3, expected_steps=1 if depth == 1 else 2
    )
    return filler, depth, size, cfg.headroom()[TRUTH.get(filler, STRUCT[filler])], k.knee, k.anomaly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--delay", default="miss", choices=["miss", "fp", "int"])
    ap.add_argument("--depths", default=None, help="comma list; default 1,2,3 (miss) or 12 (ALU)")
    ap.add_argument("--fillers", default="nop,int,fp,load,store")
    args = ap.parse_args()
    depths = [int(d) for d in (args.depths or ("1,2,3" if args.delay == "miss" else "12")).split(",")]
    sizes = [24, 32, 48, 64, 96] if args.delay == "miss" else [32, 48, 64, 96, 128]
    fillers = args.fillers.split(",")
    jobs = [(f, d, s, args.delay) for f in fillers for d in depths for s in sizes]
    with Pool(args.jobs) as pool:
        res = pool.map(one, jobs, chunksize=1)

    entries, ok = [], True
    for f in fillers:
        for d in depths:
            pts = [(r[4], r[3]) for r in res if r[0] == f and r[1] == d and r[4] is not None]
            bad = [r for r in res if r[0] == f and r[1] == d and (r[4] is None or r[5])]
            knee, truth = np.array(pts, dtype=float).T
            slope = float(np.polyfit(knee, truth, 1)[0])
            diff = truth - knee
            off, sd = float(np.median(diff)), float(np.std(diff))
            good = 0.98 <= slope <= 1.02 and sd <= 0.5 and not bad
            ok &= good
            print(
                f"{f:5s} d={d}  offset {off:g}  sd {sd:.2f}  slope {slope:.3f}  "
                f"knees {[int(k) for k in knee]}  {'ok' if good else 'FAIL'}"
                + "".join(f"\n      size {r[2]}: knee {r[4]} {r[5] or ''}" for r in bad)
            )
            if not good:
                continue
            entries.append(
                {
                    "id": f"abstract.toy.{f}.d{d}.{args.delay}",
                    "emitter": "abstract",
                    "platform": "toy",
                    "calibrated_on": "toy (uarchre/backends/platform/toy.py defaults)",
                    "filler": f,
                    "depth": d,
                    "delay": args.delay,
                    "offset": round(off),
                    "offset_sd": round(sd, 3),
                    "slope": round(slope, 4),
                    "n_configs": len(pts),
                    "evidence": "validation/toy/calibrate_offsets.py",
                }
            )
    if args.write:
        if not ok:
            print("some fits failed: writing only the passing ones")
        ids = {e["id"] for e in entries}
        table = [e for e in offsets.load() if e["id"] not in ids]
        with open(offsets.PATH) as fh:
            header = "".join(line for line in fh if line.startswith("#"))
        with open(offsets.PATH, "w") as fh:
            fh.write(header)
            yaml.safe_dump(table + entries, fh, sort_keys=False)
        print(f"merged {len(entries)} toy entries into {os.path.relpath(offsets.PATH, ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
