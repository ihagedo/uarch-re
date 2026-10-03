#!/usr/bin/env python3
"""P2.3 on gem5 O3 (RISC-V): memory-level parallelism.

K = 1..48 independent random chains, no fillers. While K <= L1D MSHRs all misses
overlap and T(K) stays at one miss latency; past it T grows by L/MSHRs per chain.
Truth: cpu.dcache.mshrs in {4, 8, 16, 32}, ROB, LSQ, PRF and L2 MSHRs set large. Far
cache: only the L1D MSHRs can bind; DDR5-8400 and DDR3-1600: the memory side may first.

    python3 validation/gem5/mem_mlp.py [far] [ddr5] [ddr3] [holdout]
"""

import os

import numpy as np

from mem_common import run_jobs, RESULTS

OUT = os.path.join(RESULTS, "mem-mlp", "results.json")
MSHRS = [4, 8, 16, 32]
KS = list(range(1, 13)) + [14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40, 44, 48]
SEEDS = [0, 1, 2]
F, ITERS, REPS = 8 << 20, 400, 2
PARAMS = dict(numROBEntries=512, LQEntries=128, SQEntries=64, numPhysIntRegs=512, numPhysFloatRegs=512)
MEM = {
    "ddr5": ["mem_ctrl.dram=DDR5_8400_4x8(range=AddrRange('1GB'))"],
    "ddr3": [],
}


# far cache: 2 MiB of chains in a 4 MiB L2 with 100-cycle hits, ~50 misses in flight, so only
# the L1D MSHRs bind. x=8 MiB streams the chain out and reads it back clean (with dirty lines
# every fill is a writeback and the 8 L1D write buffers bind first)
FAR = dict(F=2 << 20, ITERS=200, sets=["l2.size=4096kB", "l2.tag_latency=100", "l2.data_latency=100"])


def jobs(mems=("far", "ddr5", "ddr3")):
    out = []
    if "far" in mems:
        for m in MSHRS:
            for s in SEEDS:
                out.append(
                    {
                        "key": f"far/m{m}/s{s}",
                        "mem": "far",
                        "mshrs": m,
                        "seed": s,
                        "argv": ["mlp", FAR["F"], FAR["ITERS"], REPS, s, f"x={8 << 20}", *KS],
                        "params": PARAMS,
                        "sets": [f"cpu.dcache.mshrs={m}", "l2.mshrs=64", *FAR["sets"]],
                    }
                )
    for mem in [m for m in mems if m != "far"]:
        seeds = SEEDS if mem == "ddr5" else SEEDS[:1]
        for m in MSHRS:
            for s in seeds:
                out.append(
                    {
                        "key": f"{mem}/m{m}/s{s}",
                        "mem": mem,
                        "mshrs": m,
                        "seed": s,
                        "argv": ["mlp", F, ITERS, REPS, s, *KS],
                        "params": PARAMS,
                        "sets": [f"cpu.dcache.mshrs={m}", "l2.mshrs=64", *MEM[mem]],
                    }
                )
    return out


OUT_FAR = os.path.join(RESULTS, "mem-mlp", "far.json")
# held-out MSHR counts: not used to choose the reading rule
HOLDOUT = [12, 24]
OUT_HO = os.path.join(RESULTS, "mem-mlp", "far_holdout.json")


def holdout_jobs():
    return [
        {
            "key": f"far/m{m}/s0",
            "mem": "far",
            "mshrs": m,
            "seed": 0,
            "holdout": True,
            "argv": ["mlp", FAR["F"], FAR["ITERS"], REPS, 0, f"x={8 << 20}", *KS],
            "params": PARAMS,
            "sets": [f"cpu.dcache.mshrs={m}", "l2.mshrs=64", *FAR["sets"]],
        }
        for m in HOLDOUT
    ]


# flags on the shape of T: T up > 10% at the hinge knee, or saturated MLP > 25% off the knee
FLAT = 1.10
CONSIST = 0.25


def hinge(ks, ts):
    """Continuous hinge T = a + b max(0, K - c), c over integers; returns (c, (a, b))."""
    ks, ts = np.asarray(ks, float), np.asarray(ts, float)
    best = None
    for c in range(int(ks.min()), int(ks.max()) - 2):
        X = np.column_stack([np.ones_like(ks), np.maximum(0, ks - c)])
        coef, *_ = np.linalg.lstsq(X, ts, rcond=None)
        sse = float(((X @ coef - ts) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, c, coef)
    return best[1], best[2]


if __name__ == "__main__":
    import sys

    want = sys.argv[1:] or ("far", "ddr5", "ddr3")
    if "holdout" in want:
        run_jobs("mem_mlp", holdout_jobs(), OUT_HO)
        want = tuple(m for m in want if m != "holdout")
    if "far" in want:
        run_jobs("mem_mlp", jobs(("far",)), OUT_FAR)
    rest = tuple(m for m in want if m != "far")
    if rest:
        run_jobs("mem_mlp", jobs(rest), OUT)
