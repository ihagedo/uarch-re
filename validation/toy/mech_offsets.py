#!/usr/bin/env python3
"""Toy offsets the anomaly fixtures need beyond offsets.yaml: the branch filler and ALU delays.

Same fit as calibrate_offsets.py, for every filler under an 8- and 12-link FP delay (the FP
filler under an int delay). Writes tests/fixtures/toy_offsets_extra.yaml, only for ids
offsets.yaml does not already have.

  taskset -c 4-19 python3 validation/toy/mech_offsets.py [--jobs 6]
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
from uarchre.analysis.knee import find_knee
from uarchre.analysis.anomaly import probe_params
from uarchre.analysis import offsets

OUT = os.path.join(ROOT, "tests", "fixtures", "toy_offsets_extra.yaml")
BIG = dict(rob=512, int_prf=32 + 480, fp_prf=32 + 480, lq=256, sq=256)
KNOB = {"nop": "rob", "int": "int_prf", "fp": "fp_prf", "load": "lq", "store": "sq", "branch": "rob"}
TRUTH = {
    "nop": "rob",
    "int": "int_prf_headroom",
    "fp": "fp_prf_headroom",
    "load": "lq",
    "store": "sq",
    "branch": "rob",
}
SIZES = [24, 32, 48, 64, 96]
ALU_DEPTHS = (8, 12)  # 8 x 24-cycle links ~ the toy's 200-cycle miss; 12 = gem5 campaign's fsqrt x 12


def contexts(alu_depths):
    c = [("branch", d, "miss") for d in (1, 2, 3)]
    c += [(f, d, "int" if f == "fp" else "fp") for d in alu_depths for f in KNOB]
    return c


def one(job):
    filler, depth, delay, size = job
    knob = KNOB[filler]
    cfg = ToyConfig(**{**BIG, knob: size + (32 if knob.endswith("prf") else 0)})
    ns = list(range(max(2, size - 14), size + 7))
    ys = [run_toy(window_probe(filler, n, depth=depth, delay=delay), cfg) for n in ns]
    min_jump, steps = probe_params(depth, delay)
    k = find_knee(ns, ys, min_jump=min_jump, expected_steps=steps)
    return job, cfg.headroom()[TRUTH[filler]], k.knee, k.anomaly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--alu-depths", default=",".join(map(str, ALU_DEPTHS)))
    args = ap.parse_args()
    try:
        os.sched_setaffinity(0, set(range(4, 20)) & os.sched_getaffinity(0) or os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    ctxs = contexts([int(d) for d in args.alu_depths.split(",")])
    jobs = [(f, d, dl, s) for f, d, dl in ctxs for s in SIZES]
    with Pool(args.jobs) as pool:
        res = pool.map(one, jobs, chunksize=1)
    entries, ok = [], True
    for f, d, dl in ctxs:
        rows = [r for r in res if r[0][:3] == (f, d, dl)]
        pts = np.array([(r[2], r[1]) for r in rows if r[2] is not None], float)
        good = len(pts) == len(SIZES)
        slope = float(np.polyfit(pts[:, 0], pts[:, 1], 1)[0]) if len(pts) > 1 else float("nan")
        diff = pts[:, 1] - pts[:, 0]
        off, sd = float(np.median(diff)), float(np.std(diff))
        good &= 0.98 <= slope <= 1.02 and sd <= 0.5
        ok &= good
        print(
            f"{f:6s} d={d} {dl:4s} offset {off:g} sd {sd:.2f} slope {slope:.3f} "
            f"knees {[int(k) for k in pts[:, 0]]} {'ok' if good else 'FAIL'}"
        )
        entries.append(
            {
                "id": f"abstract.toy.{f}.d{d}.{dl}",
                "emitter": "abstract",
                "platform": "toy",
                "calibrated_on": "toy (uarchre/backends/platform/toy.py defaults)",
                "filler": f,
                "depth": d,
                "delay": dl,
                "offset": round(off),
                "offset_sd": round(sd, 3),
                "slope": round(slope, 4),
                "n_configs": len(pts),
                "evidence": "validation/toy/mech_offsets.py",
            }
        )
    if not ok:
        sys.exit("a fit failed; not writing")
    have = {e["id"] for e in offsets.load()}
    dup = [e["id"] for e in entries if e["id"] in have]
    if dup:
        print(f"already in offsets.yaml (kept there, not written here): {', '.join(dup)}")
    entries = [e for e in entries if e["id"] not in have]
    with open(OUT, "w") as fh:
        fh.write(
            "# Toy offsets for the branch filler and ALU delays (validation/toy/mech_offsets.py).\n"
            "# Same format as uarchre/analysis/offsets.yaml; merge there once calibrate_offsets.py\n"
            "# covers these contexts. Used by the anomaly fixtures (tests/test_anomaly.py).\n"
        )
        yaml.safe_dump(entries, fh, sort_keys=False)
    print(f"wrote {len(entries)} entries to {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
