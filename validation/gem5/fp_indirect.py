#!/usr/bin/env python3
"""P3.5 indirect-predictor capacity on gem5 O3 (RISC-V; ARM: --isa aarch64).

indirect_source: S indirect call sites per round, each target rotating through T leaves,
so the run needs S*T entries; once they no longer fit each site costs about one more
mispredict. Capacity = last S on the low segment of the marginal cost, times T. Truth:
indirectSets x indirectWays; one config keeps the default instShiftAmt=0 (every 4th set).

  . tools/env.sh && python3 validation/gem5/fp_indirect.py [--analyze] [--isa aarch64]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp_common import RESULTS, run_points, config_value, pmap, save_json, load_json, log, g5
from fp_common import site_marginal as marginal, site_knee as knee_cfg
from uarchre.kernels.predictors import indirect_source
from uarchre.viz import tolerance

OUT = os.path.join(RESULTS, "fp-indirect")
# (sets, ways, targets, instShiftAmt)
CONFIGS = [
    (64, 2, 1, 2),
    (128, 2, 1, 2),
    (256, 2, 1, 2),
    (64, 4, 1, 2),
    (128, 2, 2, 2),
    (256, 2, 2, 2),
    (128, 2, 1, 0),
]
ISA = "riscv64"
ARM_OUT, ARM_CONFIGS = os.path.join(RESULTS, "arm-fp-indirect"), CONFIGS[:3]


def cfg_sets(sets, ways, shift):
    return [
        f"cpu.branchPred.indirectBranchPred.indirectSets={sets}",
        f"cpu.branchPred.indirectBranchPred.indirectWays={ways}",
        f"cpu.branchPred.instShiftAmt={shift}",
    ]


def run_sizes(cfg, sizes, phase, rounds=8, reps=2):
    sets, ways, t, shift = cfg
    smax = max(sizes)
    pts, od = run_points(
        indirect_source(smax, sorted(sizes), targets=t, rounds=rounds, reps=reps, isa=ISA),
        f"fp_ind_{phase}_T{t}",
        sets=cfg_sets(sets, ways, shift),
        isa=ISA,
    )
    took = (
        config_value(od, "system.cpu.branchPred.indirectBranchPred", "indirectSets"),
        config_value(od, "system.cpu.branchPred.indirectBranchPred", "indirectWays"),
        config_value(od, "system.cpu.branchPred", "instShiftAmt"),
    )
    return {
        "cfg": list(cfg),
        "phase": phase,
        "pts": pts,
        "outdir": od,
        "config": took,
        "indirect_mispredicted": g5.stat(od, "system.cpu.branchPred.indirectMispredicted"),
    }


def merged(runs, cfg, phases=("coarse", "fine")):
    """{S: [cycles per round per rep]} over the given phases of one configuration."""
    T = {}
    for r in runs:
        if r["cfg"] == list(cfg) and r["phase"] in phases:
            for s, v in r["pts"].items():
                T[int(s)] = v
    return T


def campaign():
    raw_path = os.path.join(OUT, "raw.json")
    raw = load_json(raw_path) if os.path.exists(raw_path) else []
    have = {(tuple(r["cfg"]), r["phase"]) for r in raw}
    jobs = []
    for cfg in CONFIGS:
        sets, ways, t, shift = cfg
        cap = sets * ways // t // (4 if shift == 0 else 1)
        step = max(2, cap // 16)
        coarse = list(range(step, int(cap * 2.2) + 1, step))
        if (cfg, "coarse") not in have:
            jobs.append((cfg, coarse))
    raw += pmap(lambda j: run_sizes(j[0], j[1], "coarse"), jobs)
    save_json(raw_path, raw)
    jobs = []
    for cfg in CONFIGS:
        if (cfg, "fine") in {(tuple(r["cfg"]), r["phase"]) for r in raw}:
            continue
        T = merged(raw, cfg)
        k, _ = knee_cfg(T)
        if k.knee is None:
            log(os.path.basename(OUT), f"{cfg}: no coarse knee")
            continue
        ss = sorted(T)
        i = ss.index(int(k.knee))
        lo, hi = ss[max(0, i - 1)], ss[min(len(ss) - 1, i + 2)]
        fine = [s for s in range(lo, hi + 1) if s not in T]
        if fine:
            jobs.append((cfg, sorted(set(fine) | {lo, hi})))
    raw += pmap(lambda j: run_sizes(j[0], j[1], "fine"), jobs)
    save_json(raw_path, raw)
    # fine2: +-10 sites around the knee at 32 rounds x 4 reps (8 x 2 is noisy at step 1);
    # context only, analyze() reports coarse + fine
    jobs = []
    for cfg in CONFIGS:
        if (cfg, "fine2") in {(tuple(r["cfg"]), r["phase"]) for r in raw}:
            continue
        k, _ = knee_cfg(merged(raw, cfg))
        if k.knee is not None:
            jobs.append((cfg, list(range(int(k.knee) - 10, int(k.knee) + 11))))
    raw += pmap(lambda j: run_sizes(j[0], j[1], "fine2", rounds=32, reps=4), jobs)
    save_json(raw_path, raw)
    return raw


def analyze(raw):
    """Reported value: coarse + fine phases, step-1 marginal knee, bootstrap over
    the 2 inner reps. fine2 is context: the transition is ~10 sites wide."""
    out = []
    for cfg in CONFIGS:
        T = merged(raw, cfg)
        if not T:
            continue
        sets, ways, t, shift = cfg
        k, margs = knee_cfg(T)
        truth = sets * ways
        det = None if k.knee is None else k.knee * t
        ci = None if k.ci is None else [k.ci[0] * t, k.ci[1] * t + (t - 1)]
        ok = det is not None and abs(det - truth) <= float(tolerance(truth)) and ci[0] <= truth <= ci[1]
        ss = sorted(margs[0])
        med = {s: float(np.median([m[s] for m in margs])) for s in ss}
        Tc = merged(raw, cfg, ("coarse",))
        mc = marginal({s: float(np.median(Tc[s])) for s in Tc})
        T2 = merged(raw, cfg, ("fine2",))
        k2 = knee_cfg(T2)[0] if T2 else None
        out.append(
            {
                "sets": sets,
                "ways": ways,
                "targets": t,
                "instShiftAmt": shift,
                "truth": truth,
                "knee_sites": k.knee,
                "detected": det,
                "ci": ci,
                "error": None if det is None else det - truth,
                "within_tolerance": bool(det is not None and abs(det - truth) <= float(tolerance(truth))),
                "pass": bool(ok),
                "jump_ratio": k.jump_ratio,
                "quality": k.quality,
                "flags": k.flags,
                "marginal": {str(s): med[s] for s in ss},
                "marginal_coarse": {str(s): v for s, v in mc.items()},
                "round_cycles": {str(s): float(np.median(T[s])) for s in sorted(T)},
                "fine2_round_cycles": {str(s): float(np.median(T2[s])) for s in sorted(T2)},
                "fine2_knee_sites": None if k2 is None else k2.knee,
                "config": sorted({tuple(r["config"]) for r in raw if r["cfg"] == list(cfg)}),
            }
        )
    return out


def main():
    global ISA, OUT, CONFIGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--isa", choices=["riscv64", "aarch64"], default="riscv64")
    a = ap.parse_args()
    if a.isa == "aarch64":
        ISA, OUT, CONFIGS = a.isa, ARM_OUT, ARM_CONFIGS
    raw = load_json(os.path.join(OUT, "raw.json")) if a.analyze else campaign()
    summ = analyze(raw)
    save_json(os.path.join(OUT, "summary.json"), summ)
    for s in summ:
        log(
            os.path.basename(OUT),
            f"sets={s['sets']} ways={s['ways']} T={s['targets']} shift={s['instShiftAmt']} "
            f"truth={s['truth']} knee_sites={s['knee_sites']} detected={s['detected']} ci={s['ci']} "
            f"pass={s['pass']} jump={s['jump_ratio']:.2f}",
        )


if __name__ == "__main__":
    main()
