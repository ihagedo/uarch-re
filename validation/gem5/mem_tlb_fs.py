#!/usr/bin/env python3
"""P2.2 on gem5 O3 (RISC-V), bare-metal full system: dTLB and iTLB capacity, walk cost.

One node per 4 KiB page over P pages, every line an L1D hit, so only dtb residency
changes with P (SE mode never consults the TLB; mem_tlb.py is that control). Truth:
cpu.mmu.dtb.size / itb.size in {8, 16, 32, 64, 128}. Phases: coarse (P at 4 per
octave), fine (from the coarse bracket only), walk (L2 hit latency sweep), itlb.

    python3 validation/gem5/mem_tlb_fs.py coarse fine walk itlb
"""

import os
import sys

from mem_common import run_jobs, load, med, RESULTS, log

OUT = os.path.join(RESULTS, "mem-tlb")
SIZES = [8, 16, 32, 64, 128]
NLOADS, REPS = 1024, 3
COARSE = sorted({int(round(4 * 2 ** (k / 4))) for k in range(0, 29)})  # 4 .. 512


def coarse_jobs():
    return [
        {
            "key": f"dtb{n}",
            "fs": True,
            "dtb_size": n,
            "plist": COARSE,
            "nloads": NLOADS,
            "reps": REPS,
            "sets": [f"cpu.mmu.dtb.size={n}"],
        }
        for n in SIZES
    ]


def step_bracket(ps, ys, frac=0.5):
    """First coarse transition: (last P below lo + frac*(hi-lo), first P above)."""
    lo, hi = ys[0], max(ys)
    thr = lo + frac * (hi - lo)
    for i in range(1, len(ps)):
        if ys[i] >= thr:
            return ps[i - 1], ps[i], lo, hi
    return None


def fine_jobs():
    co = load(os.path.join(OUT, "fs_coarse.json"))
    jobs = []
    for n in SIZES:
        k = f"dtb{n}"
        if k not in co:
            continue
        d = co[k]["runs"]["tlb"]
        ps = sorted(int(p) for p in d)
        ys = [med(d[str(p)]) for p in ps]
        br = step_bracket(ps, ys)
        if not br:
            log("mem_tlb_fs", f"{k}: no step in coarse data")
            continue
        lo, hi, ylo, yhi = br
        plist = list(range(max(2, lo - 2), hi + 2))
        jobs.append(
            {
                "key": k,
                "fs": True,
                "dtb_size": n,
                "plist": plist,
                "nloads": NLOADS,
                "reps": REPS,
                "bracket": [lo, hi],
                "levels": [ylo, yhi],
                "sets": [f"cpu.mmu.dtb.size={n}"],
            }
        )
    return jobs


WALK_L2 = [5, 10, 20, 30]
WALK_P = [8, 64]  # dtb.size 16: P = 8 all hits, P = 64 every load walks


def walk_jobs():
    """Page-walk cost: Sv39 walks read 3 PTEs through the L2 (walker ports on
    the L2 bus), so the miss-minus-hit cost should move by 3x any change of
    the L2 hit latency (tag = data latency here)."""
    return [
        {
            "key": f"walk-l2lat{x}",
            "fs": True,
            "dtb_size": 16,
            "l2_latency": x,
            "plist": WALK_P,
            "nloads": NLOADS,
            "reps": REPS,
            "sets": ["cpu.mmu.dtb.size=16", f"l2.tag_latency={x}", f"l2.data_latency={x}"],
        }
        for x in WALK_L2
    ]


ITLB_P = list(range(2, 161))  # every P: resolution 1 entry, no second round needed


def itlb_jobs():
    """iTLB: P code pages, each a 5-instruction stub jumping to the next page
    in random cyclic order (every jump fetches from a new page); truth
    cpu.mmu.itb.size. NLOADS = jumps per rep."""
    return [
        {
            "key": f"itb{n}",
            "fs": True,
            "itlb": True,
            "itb_size": n,
            "plist": ITLB_P,
            "nloads": NLOADS,
            "reps": REPS,
            "sets": [f"cpu.mmu.itb.size={n}"],
        }
        for n in SIZES
    ]


if __name__ == "__main__":
    want = sys.argv[1:] or ["coarse", "fine"]
    if "coarse" in want:
        run_jobs("mem_tlb_fs", coarse_jobs(), os.path.join(OUT, "fs_coarse.json"))
    if "fine" in want:
        run_jobs("mem_tlb_fs", fine_jobs(), os.path.join(OUT, "fs_fine.json"))
    if "itlb" in want:
        run_jobs("mem_tlb_fs", itlb_jobs(), os.path.join(OUT, "fs_itlb.json"))
    if "walk" in want:
        run_jobs("mem_tlb_fs", walk_jobs(), os.path.join(OUT, "fs_walk.json"))
