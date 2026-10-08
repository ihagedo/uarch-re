#!/usr/bin/env python3
"""P3.4 direction-predictor history length on gem5 O3 (RISC-V; ARM: --isa aarch64), TAGE.

history_source(N): branch A on a random bit, N taken jumps, then branch B, with B = A
(corr), B random (rand) or B taken (const). f(N) = (T_corr - T_const) / (T_rand - T_const)
is 0 while A is still in the history at B and 1 once it has left; history length = last
N on the low segment + 1. Truth: TAGE maxHist in {64, 130 (default), 256}.

  . tools/env.sh && python3 validation/gem5/fp_history.py [--analyze] [--isa aarch64]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp_common import RESULTS, run_points, config_value, pmap, save_json, load_json, log
from uarchre.kernels.predictors import history_source
from uarchre.analysis.bootstrap import bootstrap_knee
from uarchre.viz import tolerance

OUT = os.path.join(RESULTS, "fp-history")
HISTS = [64, 130, 256]
SEEDS = [0, 1]
GRID = [0, 16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 288, 320, 384]
ITERS, WARM = 1000, 1000
ISA = "riscv64"
ARM_OUT = os.path.join(RESULTS, "arm-fp-history")  # same maxHist values and grid


def tage(h):
    return f"cpu.branchPred.conditionalBranchPred=TAGE(tage=TAGEBase(maxHist={h}))"


def run_one(job):
    """One gem5 run: all N of one (maxHist, seed) round, 3 variants each."""
    h, seed, ns = job
    pts, od = run_points(
        history_source(list(ns), iters=ITERS, warm=WARM, seed=seed, isa=ISA),
        f"fp_hist_s{seed}",
        sets=[tage(h)],
        isa=ISA,
    )
    took = int(config_value(od, "system.cpu.branchPred.conditionalBranchPred.tage", "maxHist"))
    return [
        {
            "maxHist": h,
            "seed": seed,
            "n": n,
            "outdir": od,
            "config_maxHist": took,
            "pts": {k: v for k, v in pts.items() if int(k.split(":")[1]) == n},
        }
        for n in ns
    ]


def runs(todo):
    groups = {}
    for h, s, n in todo:
        groups.setdefault((h, s), []).append(n)
    jobs = [(h, s, tuple(sorted(ns))) for (h, s), ns in sorted(groups.items())]
    return [r for rows in pmap(run_one, jobs) for r in rows]


def frac(rows):
    """{N: f(N)} for one seed."""
    out = {}
    for r in rows:
        p = {k.split(":")[0]: float(np.median(v)) for k, v in r["pts"].items()}
        den = p["rand"] - p["const"]
        out[r["n"]] = (p["corr"] - p["const"]) / den if den > 0.5 else float("nan")
    return out


def knee(raw, h):
    per_seed = [frac([r for r in raw if r["maxHist"] == h and r["seed"] == s]) for s in SEEDS]
    ns = sorted(set.intersection(*[set(f) for f in per_seed]))
    ns = [n for n in ns if all(np.isfinite(f[n]) for f in per_seed)]
    return bootstrap_knee({n: [1 + f[n] for f in per_seed] for n in ns}), per_seed, ns


def campaign():
    raw_path = os.path.join(OUT, "raw.json")
    raw = load_json(raw_path) if os.path.exists(raw_path) else []
    todo = [(h, s, n) for h in HISTS for s in SEEDS for n in GRID]
    for _ in range(4):  # coarse, then refine the bracket until it is step 1
        have = {(r["maxHist"], r["seed"], r["n"]) for r in raw}
        todo = [j for j in todo if j not in have]
        raw += runs(todo)
        save_json(raw_path, raw)
        todo = []
        for h in HISTS:
            k, _, ns = knee(raw, h)
            if k.knee is None:
                log(os.path.basename(OUT), f"maxHist={h}: no knee yet ({k.anomaly})")
                continue
            a = int(k.knee)
            b = ns[ns.index(a) + 1]
            if b - a <= 1:
                continue
            step = 1 if b - a <= 10 else max(1, (b - a) // 8)
            todo += [(h, s, n) for s in SEEDS for n in range(a + step, b, step)]
        if not todo:
            break
    return raw


def analyze(raw):
    out = []
    for h in HISTS:
        k, per_seed, ns = knee(raw, h)
        det = None if k.knee is None else k.knee + 1
        ci = None if k.ci is None else [k.ci[0] + 1, k.ci[1] + 1]
        ok = det is not None and abs(det - h) <= float(tolerance(h)) and ci[0] <= h <= ci[1]
        out.append(
            {
                "truth": h,
                "knee_n": k.knee,
                "detected": det,
                "ci": ci,
                "error": None if det is None else det - h,
                "pass": bool(ok),
                "jump_ratio": k.jump_ratio,
                "quality": k.quality,
                "flags": k.flags,
                "f": {str(n): [f[n] for f in per_seed] for n in ns},
                "cycles": {
                    str(r["n"]): {kk.split(":")[0]: float(np.median(v)) for kk, v in r["pts"].items()}
                    for r in raw
                    if r["maxHist"] == h and r["seed"] == SEEDS[0]
                },
                "config_maxHist": sorted({r["config_maxHist"] for r in raw if r["maxHist"] == h}),
            }
        )
    return out


def main():
    global ISA, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--isa", choices=["riscv64", "aarch64"], default="riscv64")
    a = ap.parse_args()
    if a.isa == "aarch64":
        ISA, OUT = a.isa, ARM_OUT
    raw = load_json(os.path.join(OUT, "raw.json")) if a.analyze else campaign()
    summ = analyze(raw)
    save_json(os.path.join(OUT, "summary.json"), summ)
    for s in summ:
        log(
            os.path.basename(OUT),
            f"maxHist={s['truth']} knee_n={s['knee_n']} detected={s['detected']} ci={s['ci']} "
            f"err={s['error']} pass={s['pass']} jump={s['jump_ratio']:.2f}",
        )


if __name__ == "__main__":
    main()
