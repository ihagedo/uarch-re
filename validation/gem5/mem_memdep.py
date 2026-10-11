#!/usr/bin/env python3
"""P2.5 on gem5 O3 (RISC-V or ARM): store-to-load forwarding and memory disambiguation.

N blocks per iteration, each a store whose address waits on a divide and a load that aliases it (none
in the control). gem5's SSIT is direct mapped on PC/4 and a 16-byte block is 4 slots, so SSIT entries
= 4 x the last N before the cost per block rises. Truth: cpu.SSITSize in {64, 128, 256, 512}, set as
"128B" (an int is rejected); ARM_SSIT and no holdout on ARM.

    python3 validation/gem5/mem_memdep.py [--isa aarch64] coarse fine [holdout]
"""

import os
import sys

from mem_common import run_jobs, load, med, RESULTS, log, isa_args, for_isa

OUT = os.path.join(RESULTS, "mem-memdep")
ARM_OUT = os.path.join(RESULTS, "arm-memory-memdep")
ARM_SSIT = [64, 128, 256]
SSIT = [64, 128, 256, 512]
COARSE = sorted({int(round(4 * 2 ** (k / 4))) for k in range(0, 25)})  # 4 .. 256
ITERS, REPS = 100, 3


# held-out sizes: not used to choose the reading rule
HOLDOUT = [32, 1024]
COARSE_HO = sorted({int(round(4 * 2 ** (k / 4))) for k in range(0, 27)})  # 4 .. 362


def coarse_jobs(sizes=SSIT, grid=COARSE, extras=True):
    jobs = [
        {
            "key": f"ssit{n}",
            "probe": "memdep",
            "ssit": n,
            "ns": grid,
            "alias": True,
            "argv": ["memdep", ITERS, REPS, *grid],
            "params": {},
            "sets": [f"cpu.SSITSize={n}B"],
        }
        for n in sizes
    ]
    if not extras:
        return jobs
    jobs.append(
        {
            "key": "ssit64-noalias",
            "probe": "memdep",
            "ssit": 64,
            "ns": COARSE,
            "alias": False,
            "argv": ["memdep", ITERS, REPS, *COARSE],
            "params": {},
            "sets": ["cpu.SSITSize=64B"],
        }
    )
    jobs.append(
        {
            "key": "fwd",
            "probe": "memdep",
            "ns": [4],
            "alias": True,
            "argv": ["fwd", 2000, 3],
            "params": {},
            "sets": [],
        }
    )
    return jobs


def bracket(ns, ys, frac=0.1):
    """Coarse bracket around the onset (10% of the step): the ramp runs from
    SSITSize/4 to twice that, so a 50% crossing would put the fine grid mid-ramp."""
    lo, hi = min(ys[:3]), max(ys)
    thr = lo + frac * (hi - lo)
    for i in range(1, len(ns)):
        if ys[i] >= thr:
            return ns[i - 1], ns[i], lo, hi
    return None


def change_point(xs, ys):
    """Change point c of the best continuous hinge (flat, then linear), c over the interior xs."""
    import numpy as np

    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    best = None
    for c in xs[1:-1]:
        X = np.column_stack([np.ones_like(xs), np.maximum(0, xs - c)])
        coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
        sse = float(((X @ coef - ys) ** 2).sum())
        if best is None or sse < best[0] - 1e-12:
            best = (sse, int(c))
    return best[1] if best else None


def fine_jobs(sizes=SSIT, coarse="coarse.json", out=OUT):
    co = load(os.path.join(out, coarse))
    jobs = []
    for n in sizes:
        k = f"ssit{n}"
        if k not in co:
            continue
        d = co[k]["runs"]["memdep"]
        ns = sorted(int(x) for x in d)
        ys = [med(d[str(x)]) for x in ns]
        br = bracket(ns, ys)
        if not br:
            log("mem_memdep", f"{k}: no step")
            continue
        lo, hi, ylo, yhi = br
        fine = list(range(max(2, lo - 2), hi + 2))
        jobs.append(
            {
                "key": k,
                "probe": "memdep",
                "ssit": n,
                "ns": fine,
                "alias": True,
                "bracket": [lo, hi],
                "levels": [ylo, yhi],
                "argv": ["memdep", ITERS, REPS, *fine],
                "params": {},
                "sets": [f"cpu.SSITSize={n}B"],
            }
        )
    return jobs


if __name__ == "__main__":
    isa, want = isa_args(sys.argv[1:])
    want = want or ["coarse", "fine"]
    if isa == "aarch64":
        if "coarse" in want:
            run_jobs(
                "arm_mem_memdep", for_isa(coarse_jobs(ARM_SSIT), isa), os.path.join(ARM_OUT, "coarse.json")
            )
        if "fine" in want:
            jobs = for_isa(fine_jobs(ARM_SSIT, out=ARM_OUT), isa)
            run_jobs("arm_mem_memdep", jobs, os.path.join(ARM_OUT, "fine.json"))
        sys.exit()
    if "coarse" in want:
        run_jobs("mem_memdep", coarse_jobs(), os.path.join(OUT, "coarse.json"))
    if "fine" in want:
        run_jobs("mem_memdep", fine_jobs(), os.path.join(OUT, "fine.json"))
    if "holdout" in want:
        run_jobs(
            "mem_memdep",
            coarse_jobs(HOLDOUT, COARSE_HO, extras=False),
            os.path.join(OUT, "holdout_coarse.json"),
        )
        run_jobs(
            "mem_memdep", fine_jobs(HOLDOUT, "holdout_coarse.json"), os.path.join(OUT, "holdout_fine.json")
        )
