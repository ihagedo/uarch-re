#!/usr/bin/env python3
"""P3.8 decoupling and fetch-directed prefetch on gem5 O3 (RISC-V; ARM: --isa aarch64).

fdip_source(0): 7168 predictable jumps, one per code line (448 KiB, all DRAM fetches).
Only decoupled + FDP should beat one memory latency per line: with F FTQ entries ~F-1
misses overlap until DRAM bandwidth binds. --barrier (a mispredicted return every P
lines) gives no knee on gem5, which keeps returns out of the BTB.

  . tools/env.sh && python3 validation/gem5/fp_fdip.py [--analyze] [--barrier] [--isa aarch64]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp_common import RESULTS, run_points, config_value, pmap, save_json, load_json, log, g5
from uarchre.kernels.predictors import fdip_source
from uarchre.analysis.knee import find_knee

OUT = os.path.join(RESULTS, "fp-fdip")
# instShiftAmt stays 0: with 2 the 2-byte BAC search aliases onto 4-byte BTB slots and gem5 asserts
BASE = [
    "cpu.branchPred.conditionalBranchPred=TAGE()",  # the decoupled frontend needs branchPlaceholder
    "cpu.branchPred.btb.numEntries=32768",  # decoupled fetch sees only BTB hits (default: 1024 branches)
    "cpu.fetchTargetWidth=64",  # one line = one fetch target
    "cpu.minInstSize=2",  # RVC in the harness
    "cpu.icache.mshrs=32",  # MSHRs never cap the overlap
]
# name, decoupled, FTQ entries, FDP attached to the I-cache
CONFIGS = (
    [("coupled", False, 8, False), ("coupled+FDP", False, 8, True)]
    + [(f"ftq-only{f}", True, f, False) for f in (2, 8, 16)]
    + [(f"fdp-ftq{f}", True, f, True) for f in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16)]
)
SEEDS = [1, 2]  # traversal orders
BARRIER_PERIODS = [[1, 2, 3, 4, 5, 6, 7, 8], [10, 12, 14, 16, 20, 24, 32, 0]]
ISA = "riscv64"
ARM_OUT = os.path.join(RESULTS, "arm-fp-fdip")
ARM_BASE = [b.replace("minInstSize=2", "minInstSize=4") for b in BASE]  # fixed 4 B instructions
ARM_CONFIGS = (
    CONFIGS[:2] + [("ftq-only8", True, 8, False)] + [(f"fdp-ftq{f}", True, f, True) for f in (2, 3, 5)]
)


def run_one(job):
    name, dec, f, fdp, periods, seed = job
    sets = BASE + [f"cpu.decoupledFrontEnd={dec}", f"cpu.numFTQEntries={f}"]
    pts, od = run_points(
        fdip_source(periods, seed=seed, isa=ISA),
        f"fp_fd_s{seed}",
        sets=sets,
        fdp_args=["--fdp"] if fdp else [],
        isa=ISA,
    )
    return {
        "config": name,
        "decoupled": dec,
        "ftq": f,
        "fdp": fdp,
        "seed": seed,
        "pts": pts,
        "outdir": od,
        "config_decoupled": config_value(od, "system.cpu", "decoupledFrontEnd"),
        "config_ftq": config_value(od, "system.cpu", "numFTQEntries"),
        "prefetcher": config_value(od, "system.cpu.icache", "prefetcher"),
        "pf_issued": g5.stat(od, "system.cpu.icache.prefetcher.pfIssued"),
        "icache_misses": g5.stat(od, "system.cpu.icache.overallMisses::total"),
    }


def analyze(raw):
    """Steady state (no barrier): cycles per line per config; overlap = coupled / config."""
    base = float(np.median([np.median(r["pts"]["0"]) for r in raw if r["config"] == "coupled"]))
    out = []
    for name, dec, f, fdp in CONFIGS:
        rows = [r for r in raw if r["config"] == name]
        if not rows:
            continue
        per_seed = [float(np.median(r["pts"]["0"])) for r in rows]
        cost = float(np.median(per_seed))
        out.append(
            {
                "config": name,
                "decoupled": dec,
                "ftq": f,
                "fdp": fdp,
                "cycles_per_line": cost,
                "cycles_per_line_seeds": per_seed,
                "overlap": base / cost,
                "overlap_range": [base / max(per_seed), base / min(per_seed)],
                "pf_issued": sum((r["pf_issued"] or 0) for r in rows),
                "config_check": sorted(
                    {(r["config_decoupled"], r["config_ftq"], r["prefetcher"]) for r in rows}
                ),
            }
        )
    return {"coupled_cycles_per_line": base, "configs": out}


def barrier_analyze(raw):
    """The --barrier sweep over the barrier period P (a negative result)."""
    out = []
    for name in sorted({r["config"] for r in raw}):
        per_line = {}
        for r in raw:
            if r["config"] == name:
                for p, v in r["pts"].items():
                    per_line[int(p)] = float(np.median(v))
        ps = sorted(p for p in per_line if p > 0)
        seg = {p: p * per_line[p] for p in ps}
        marg = {b: (seg[b] - seg[a]) / (b - a) for a, b in zip(ps, ps[1:])}
        mp = sorted(marg)
        kn = find_knee(mp, [marg[p] for p in mp])
        out.append(
            {
                "config": name,
                "per_line": {str(p): per_line[p] for p in sorted(per_line)},
                "marginal": {str(p): marg[p] for p in mp},
                "knee": kn.knee,
                "jump_ratio": kn.jump_ratio,
            }
        )
    return out


def main():
    global ISA, OUT, BASE, CONFIGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument(
        "--barrier", action="store_true", help="re-run the barrier-period sweep (negative finding)"
    )
    ap.add_argument("--isa", choices=["riscv64", "aarch64"], default="riscv64")
    a = ap.parse_args()
    if a.isa == "aarch64":
        ISA, OUT, BASE, CONFIGS = a.isa, ARM_OUT, ARM_BASE, ARM_CONFIGS
    if a.barrier:
        cfgs = [
            c
            for c in CONFIGS
            if c[0] in ("coupled", "ftq-only8")
            or c[0] in ("fdp-ftq2", "fdp-ftq4", "fdp-ftq6", "fdp-ftq8", "fdp-ftq12")
        ]
        raw = pmap(run_one, [(*c, ps, 1) for c in cfgs for ps in BARRIER_PERIODS])
        save_json(os.path.join(OUT, "barrier-raw.json"), raw)
    if a.barrier or os.path.exists(os.path.join(OUT, "barrier-raw.json")):
        save_json(
            os.path.join(OUT, "barrier-summary.json"),
            barrier_analyze(load_json(os.path.join(OUT, "barrier-raw.json"))),
        )
    raw_path = os.path.join(OUT, "raw.json")
    if not a.analyze:
        raw = pmap(run_one, [(*c, [0], s) for c in CONFIGS for s in SEEDS])
        save_json(raw_path, raw)
    raw = load_json(raw_path)
    summ = analyze(raw)
    save_json(os.path.join(OUT, "summary.json"), summ)
    for s in summ["configs"]:
        log(
            os.path.basename(OUT),
            f"{s['config']:12s} cyc/line={s['cycles_per_line']:.1f} seeds={s['cycles_per_line_seeds']} "
            f"overlap={s['overlap']:.2f} pf_issued={s['pf_issued']}",
        )


if __name__ == "__main__":
    main()
