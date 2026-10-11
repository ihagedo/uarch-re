#!/usr/bin/env python3
"""P3.6 return-address-stack depth on gem5 O3 (RISC-V, or ARM with --isa aarch64).

One run per RAS size R and placement seed sweeps call depth D = 1..DMAX (ras_source).
M(D) = T(D) - T(D-1) is ~2 cycles while every return is predicted and one mispredict
more once D > R; depth = last D on the low segment (offset 0: the harness's call is level 1).

  . tools/env.sh && python3 validation/gem5/fp_ras.py [--isa aarch64]   # campaign + summary
  python3 validation/gem5/fp_ras.py --analyze                           # summary only
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp_common import RESULTS, run_points, config_value, pmap, save_json, load_json, log, g5
from fp_common import depth_marginal as marginal
from uarchre.kernels.predictors import ras_source
from uarchre.analysis.knee import find_knee
from uarchre.analysis.anomaly import band_knee
from uarchre.viz import tolerance

OUT = os.path.join(RESULTS, "fp-ras")
SIZES = [8, 12, 16, 24, 32, 48, 64]
SEEDS = [0, 1, 2]
DMAX = 128
ITERS, REPS, WARM = 32, 3, 4
ISA = "riscv64"
ARM_OUT, ARM_SIZES, ARM_SEEDS = os.path.join(RESULTS, "arm-fp-ras"), [8, 16, 32, 48], [0, 1]


def run_one(job):
    r, seed = job
    src = ras_source(dmax=DMAX, iters=ITERS, reps=REPS, warm=WARM, seed=seed, isa=ISA)
    pts, od = run_points(src, f"fp_ras_s{seed}", sets=[f"cpu.branchPred.ras.numEntries={r}"], isa=ISA)
    took = config_value(od, "system.cpu.branchPred.ras", "numEntries")
    st = {k: g5.stat(od, f"system.cpu.branchPred.ras.{k}") for k in ("correct", "incorrect", "used")}
    log(
        os.path.basename(OUT),
        f"R={r} seed={seed} config numEntries={took} ras.incorrect={st['incorrect']} outdir={od}",
    )
    return {
        "ras": r,
        "seed": seed,
        "config_numEntries": int(took),
        "outdir": od,
        "stats": st,
        "T": {d: v for d, v in pts.items()},
    }


def analyze(raw):
    out = []
    for r in sorted({x["ras"] for x in raw}):
        runs = [x for x in raw if x["ras"] == r]
        per_seed = []
        for x in runs:
            T = {int(d): float(np.median(v)) for d, v in x["T"].items()}
            per_seed.append(marginal(T))
        ds = sorted(per_seed[0])
        reps = {d: [m[d] for m in per_seed] for d in ds}
        kn = band_knee(reps)
        med = {d: float(np.median(reps[d])) for d in ds}
        pt = find_knee(ds, [med[d] for d in ds])
        # cross-check from gem5's own counter: expected mispredicted returns
        calls = ITERS * REPS + WARM
        expect_incorrect = sum(max(0, d - r) for d in range(1, DMAX + 1)) * calls
        lo = float(np.median([med[d] for d in ds if d <= (kn.knee or 0)])) if kn.knee else None
        hi = float(np.median([med[d] for d in ds if d > (kn.knee or 0)])) if kn.knee else None
        err = None if kn.knee is None else kn.knee - r
        ok = kn.knee is not None and abs(err) <= float(tolerance(r)) and kn.ci[0] <= r <= kn.ci[1]
        out.append(
            {
                "truth": r,
                "detected": kn.knee,
                "ci": kn.ci,
                "ci_method": kn.ci_method,
                "n_reps": kn.n_reps,
                "error": err,
                "pass": bool(ok),
                "jump_ratio": pt.jump_ratio,
                "quality": pt.quality,
                "flags": kn.flags,
                "cost_low": lo,
                "cost_high": hi,
                "marginal_median": {str(d): med[d] for d in ds},
                "ras_incorrect_stat": [x["stats"]["incorrect"] for x in runs],
                "ras_incorrect_expected_min": expect_incorrect,
                "config_numEntries": sorted({x["config_numEntries"] for x in runs}),
            }
        )
    return out


def main():
    global ISA, OUT, SIZES, SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--isa", choices=["riscv64", "aarch64"], default="riscv64")
    a = ap.parse_args()
    if a.isa == "aarch64":
        ISA, OUT, SIZES, SEEDS = a.isa, ARM_OUT, ARM_SIZES, ARM_SEEDS
    raw_path = os.path.join(OUT, "raw.json")
    if not a.analyze:
        raw = pmap(run_one, [(r, s) for r in SIZES for s in SEEDS])
        save_json(raw_path, raw)
    raw = load_json(raw_path)
    summ = analyze(raw)
    save_json(os.path.join(OUT, "summary.json"), summ)
    for s in summ:
        log(
            os.path.basename(OUT),
            f"R={s['truth']:3d} detected={s['detected']} ci={s['ci']} err={s['error']} "
            f"pass={s['pass']} jump={s['jump_ratio']:.2f} low={s['cost_low']:.2f} high={s['cost_high']:.2f} "
            f"ras.incorrect={s['ras_incorrect_stat']} (>= {s['ras_incorrect_expected_min']} expected)",
        )


if __name__ == "__main__":
    main()
